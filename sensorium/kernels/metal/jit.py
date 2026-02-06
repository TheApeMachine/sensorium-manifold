"""Just-In-Time (JIT) compilation and loading of Apple Metal GPU extensions.

This module provides infrastructure to compile Metal shaders (.metal files) into
GPU-executable code and load them as Python-callable extensions at runtime. This
enables high-performance GPU acceleration for physics simulation kernels on macOS.

Architecture Overview:
----------------------
Metal shaders follow a two-stage compilation pipeline:
  1. .metal source -> .air (Apple Intermediate Representation) via `metal` compiler
  2. .air files -> .metallib (Metal Library archive) via `metallib` linker

The resulting .metallib is then loaded by an Objective-C++ wrapper (ops.mm) which
exposes the GPU kernels to Python through PyTorch's C++ extension mechanism.

Why JIT Compilation:
--------------------
- Avoids shipping pre-compiled binaries for every macOS/Xcode version combination
- Enables automatic recompilation when shader sources change
- Allows the same codebase to work across different Metal feature sets

Platform Isolation:
-------------------
This module is intentionally kept separate from the main import path so that
`sensorium` can be imported and type-checked on non-macOS platforms (Linux, Windows)
without requiring Xcode toolchains. The actual compilation only happens when
explicitly invoked on a macOS system with proper tooling installed.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Any

from sensorium.kernels.runtime import metal_build_tools_available, metal_supported


def _this_dir() -> Path:
    """Return the directory containing this module.
    
    Used to locate sibling files like Metal shader sources (manifold_physics.metal)
    and the Objective-C++ wrapper (ops.mm) that live alongside this Python file.
    """
    return Path(__file__).resolve().parent


# Module-level cache for the compiled extension to avoid redundant compilation.
# Once successfully built, the extension is reused for the lifetime of the process.
_CACHED_MOD: Any | None = None

# Cache for any compilation/loading error to fail fast on subsequent attempts.
# This prevents repeated expensive failure attempts if the toolchain is broken.
_CACHED_ERR: Exception | None = None


def _xcrun_find(tool: str) -> str:
    """Locate an Xcode developer tool using Apple's xcrun utility.
    
    Apple's xcrun is the canonical way to find tools within the active Xcode
    toolchain. This handles the complexity of multiple Xcode installations,
    command-line tools vs full Xcode, and SDK path resolution.
    
    Args:
        tool: Name of the tool to find (e.g., "metal", "metallib", "clang")
        
    Returns:
        Absolute path to the requested tool executable
        
    Raises:
        RuntimeError: If the tool cannot be located, with actionable fix instructions
        
    Why xcrun instead of hardcoded paths:
    -------------------------------------
    Tool locations vary based on:
      - Whether Xcode.app or just CommandLineTools is installed
      - Which Xcode version is selected via xcode-select
      - Custom developer directory configurations
    
    Using xcrun ensures we always find the correct tool for the active configuration.
    """
    try:
        # Query xcrun with the macOS SDK to find the tool path.
        # The -sdk macosx flag ensures we get macOS tools, not iOS/tvOS variants.
        out = subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "--find", str(tool)],
            stderr=subprocess.STDOUT,
        )
        p = out.decode("utf-8", errors="replace").strip()
        if not p:
            raise RuntimeError(f"xcrun returned empty path for tool {tool!r}")
        return p
    except Exception as e:
        # Provide actionable diagnostics: most failures are missing SDK/toolchain.
        # Capture the current developer directory to help users debug.
        try:
            devdir = subprocess.check_output(
                ["xcode-select", "-p"], stderr=subprocess.STDOUT
            ).decode("utf-8", errors="replace").strip()
        except Exception:
            devdir = "<unknown>"
        
        raise RuntimeError(
            f"Unable to locate required Xcode tool {tool!r} via xcrun.\n"
            f"Active developer dir: {devdir}\n\n"
            "Fix:\n"
            "  - Install Xcode Command Line Tools: `xcode-select --install`\n"
            "  - OR select Xcode.app:\n"
            "      `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
            "      `sudo xcodebuild -license accept`\n\n"
            "Verify:\n"
            "  `xcrun -sdk macosx --find metal`\n"
            "  `xcrun -sdk macosx --find metallib`\n"
        ) from e


def _compile_metallib(*, out_dir: Path, verbose: bool) -> Path:
    """Compile Metal shader sources into a Metal library archive.
    
    This implements the two-stage Metal compilation pipeline:
      1. Compile each .metal source to .air (Apple Intermediate Representation)
      2. Link all .air files into a single .metallib archive
    
    The .metallib format is Apple's archive format for GPU shader code, similar
    to how .a or .so files work for CPU code. It contains optimized GPU bytecode
    that can be loaded directly by the Metal runtime.
    
    Args:
        out_dir: Directory where intermediate (.air) and output (.metallib) files
                 are written. This is typically PyTorch's extension build directory.
        verbose: If True, print compilation commands for debugging
        
    Returns:
        Path to the compiled .metallib file
        
    Raises:
        RuntimeError: If shader compilation or linking fails
        
    Incremental Compilation:
    ------------------------
    The function implements timestamp-based caching: if the .metallib already
    exists and is newer than all source files, compilation is skipped. This
    makes repeated imports fast after the initial build.
    """
    # List of Metal shader source files to compile.
    # Additional shaders can be added here as the kernel library grows.
    sources = [
        _this_dir() / "manifold_physics.metal",
    ]
    
    # Intermediate .air files - one per source file.
    # AIR (Apple Intermediate Representation) is LLVM bitcode optimized for Metal.
    airs = [out_dir / f"{src.stem}.air" for src in sources]
    
    # Final output: a single .metallib containing all compiled shaders
    metallib = out_dir / "manifold_ops.metallib"

    # Locate the Metal compiler and linker from the active Xcode toolchain
    metal = _xcrun_find("metal")
    metallib_tool = _xcrun_find("metallib")

    # Incremental build check: skip compilation if output is up-to-date.
    # This compares modification times to avoid unnecessary recompilation.
    if metallib.exists():
        mt = metallib.stat().st_mtime
        if all(mt >= src.stat().st_mtime for src in sources):
            return metallib

    # Ensure the output directory exists (PyTorch may not create it yet)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Compile each .metal source to .air intermediate representation.
    # The -c flag means "compile only" (don't link), similar to gcc -c.
    for src, air in zip(sources, airs, strict=True):
        cmd = [
            metal,      # Metal compiler (similar to clang for GPU code)
            "-c",       # Compile only, produce .air intermediate file
            str(src),   # Input .metal source file
            "-o",       # Output flag
            str(air),   # Output .air file path
        ]
        if verbose:
            print("[manifold] compiling Metal shader:", " ".join(cmd))
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to compile Metal shaders with the active toolchain.\n\n"
                f"Command:\n  {' '.join(cmd)}\n\n"
                f"stdout:\n{proc.stdout}\n\n"
                f"stderr:\n{proc.stderr}\n\n"
                "If the error mentions SDKs, verify:\n"
                "  `xcrun --sdk macosx --show-sdk-path`\n"
                "If the error mentions missing tools, verify:\n"
                "  `xcrun -sdk macosx --find metallib`\n"
            )

    # Stage 2: Link all .air files into a single .metallib archive.
    # This is similar to using `ar` or `ld` for CPU object files.
    cmd2 = [
        metallib_tool,                  # Metal library archiver/linker
        *[str(air) for air in airs],    # All .air input files
        "-o",                           # Output flag
        str(metallib),                  # Output .metallib path
    ]
    if verbose:
        print("[manifold] linking Metal metallib:", " ".join(cmd2))
    
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            "Failed to link Metal metallib (`metallib`).\n\n"
            f"Command:\n  {' '.join(cmd2)}\n\n"
            f"stdout:\n{proc2.stdout}\n\n"
            f"stderr:\n{proc2.stderr}\n"
        )
    
    return metallib


def load_manifold_metal_ops(*, verbose: bool = False) -> Any:
    """Build and load the Metal GPU operations extension module.
    
    This is the main entry point for obtaining access to Metal-accelerated
    operations. It handles the complete pipeline:
      1. Verify Metal/MPS runtime support
      2. Verify Xcode build tools are available
      3. Compile Metal shaders to .metallib (with caching)
      4. Build the Objective-C++ Python extension (with caching)
      5. Load and return the extension module
    
    The returned module provides Python-callable functions that dispatch
    computation to GPU kernels defined in the Metal shaders.
    
    Args:
        verbose: If True, print compilation commands and progress
        
    Returns:
        The loaded Python extension module containing Metal operations
        
    Raises:
        RuntimeError: If Metal is not supported or build tools are missing
        
    Caching Behavior:
    -----------------
    - First call: Full compilation (may take several seconds)
    - Subsequent calls in same process: Instant (returns cached module)
    - Subsequent process runs: Fast if sources unchanged (timestamp check)
    
    Extension Architecture:
    -----------------------
    The extension uses PyTorch's cpp_extension loader which:
      - Compiles ops.mm (Objective-C++ wrapper) into a shared library
      - Links against Metal.framework and Foundation.framework
      - Provides pybind11 bindings for Python interoperability
    
    The ops.mm wrapper loads the .metallib at runtime and creates
    MTLFunction handles for each kernel, which can then be dispatched
    via MTLComputeCommandEncoder.
    """
    global _CACHED_MOD, _CACHED_ERR
    
    # Fast path: return previously loaded module
    if _CACHED_MOD is not None:
        return _CACHED_MOD
    
    # Fast fail: re-raise cached error from previous failed attempt.
    # This avoids repeated expensive failure attempts.
    if _CACHED_ERR is not None:
        raise _CACHED_ERR

    # Runtime check: verify Metal Performance Shaders (MPS) is available.
    # This fails on non-Apple hardware or very old macOS versions.
    if not metal_supported():
        err = RuntimeError("Metal/MPS is not supported on this runtime")
        _CACHED_ERR = err
        raise err
    
    # Build tools check: verify Xcode's Metal toolchain is installed.
    # Users need either Xcode.app or Command Line Tools with Metal support.
    if not metal_build_tools_available():
        err = RuntimeError(
            "Metal build tools unavailable.\n\n"
            "manifold's fused Metal kernels require Xcode's Metal toolchain "
            "(`metal`, `metallib`).\n"
            "Install/select it:\n"
            "  - `xcode-select --install`\n"
            "  - or install Xcode.app then:\n"
            "      `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
            "      `sudo xcodebuild -license accept`\n\n"
            "Verify:\n"
            "  `xcrun -sdk macosx --find metal`\n"
            "  `xcrun -sdk macosx --find metallib`\n"
        )
        _CACHED_ERR = err
        raise err

    # Import PyTorch's C++ extension utilities for JIT compilation.
    # This provides ninja-based parallel compilation and caching.
    import torch.utils.cpp_extension as ce

    try:
        name = "manifold_metal_ops"
        
        # Get PyTorch's build directory for this extension.
        # This is typically ~/.cache/torch_extensions/ or similar.
        build_dir = Path(ce._get_build_directory(name, verbose=verbose))

        # Compile Metal shaders first (must exist before ops.mm loads them).
        # This can take a while on the first run; print a minimal progress line
        # even when verbose=False so users don't think the process hung.
        t0 = time.perf_counter()
        print("[manifold] JIT: ensuring metallib is built…", flush=True)
        _compile_metallib(out_dir=build_dir, verbose=verbose)
        print(f"[manifold] JIT: metallib ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        # Path to the Objective-C++ wrapper that bridges Metal to Python.
        # ops.mm contains:
        #   - Metal device and command queue initialization
        #   - .metallib loading and kernel function lookup
        #   - pybind11 bindings exposing kernels to Python
        src_ops = str(_this_dir() / "ops.mm")
        
        # Compiler flags for the Objective-C++ wrapper:
        extra_cflags = [
            "-O3",          # Maximum optimization for performance-critical code
            "-std=c++17",   # C++17 for modern language features
            "-fobjc-arc",   # Automatic Reference Counting for Objective-C objects
            "-fblocks",     # Enable Clang blocks (used by Metal completion handlers)
        ]
        
        # Linker flags to link against Apple frameworks:
        extra_ldflags = [
            "-framework", "Metal",       # Metal GPU compute framework
            "-framework", "Foundation",  # Core Objective-C runtime
        ]
        
        # Build and load the extension using PyTorch's JIT loader.
        # This compiles ops.mm, links it into a .so, and imports it.
        t1 = time.perf_counter()
        print("[manifold] JIT: building/loading ops extension…", flush=True)
        mod = ce.load(
            name=name,                    # Extension module name
            sources=[src_ops],            # Objective-C++ source files
            extra_cflags=extra_cflags,    # Compiler flags
            extra_ldflags=extra_ldflags,  # Linker flags
            with_cuda=False,              # Not a CUDA extension
            is_python_module=True,        # Creates importable Python module
            build_directory=str(build_dir),  # Where to put build artifacts
            verbose=verbose,              # Print build commands if requested
        )
        print(f"[manifold] JIT: ops extension loaded ({time.perf_counter() - t1:.1f}s)", flush=True)
    except Exception as e:
        # Cache the error so subsequent calls fail fast
        _CACHED_ERR = e
        raise

    # Cache the successfully loaded module for future calls
    _CACHED_MOD = mod
    return mod
