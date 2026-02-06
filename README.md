# The Sensorium Manifold

**Native Multimodality via Isomorphism**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Metal](https://img.shields.io/badge/Metal-Apple_Silicon-black.svg)](https://developer.apple.com/metal/)
[![Triton](https://img.shields.io/badge/Triton-CUDA-76B900.svg)](https://triton-lang.org/)

---

## Abstract

The Sensorium Manifold explores a direct question: if physics is already a system that manages information—through entropy, conservation, and equilibration—how much of it do you need to simulate before useful capabilities emerge?

Rather than engineering a learning algorithm, we simulate a thermodynamic substrate governed by *Hamiltonian dynamics*. Data enters as energy distributed across coupled oscillators. Structure emerges not from optimization, but from the system finding equilibrium. What we call "learning" is the crystallization of resonant modes; what we call "inference" is the system's response to perturbation.

The system operates via three principles:

1. **Coherence Coupling** — Distant oscillators couple via shared ω-modes
2. **Metabolic Gating** — Modes persist only if energetically maintained by resonance
3. **Crystallization** — Generation is a Boundary Value Problem, not serial autoregression

This crystallization implements a *Holographic Content Addressable Memory*: partial inputs retrieve complete patterns because information is distributed across the entire resonant field. The physics handles self-regulation (homeostasis prevents runaway dynamics), adaptation (the system re-equilibrates when distributions shift), and parallelism (all oscillators relax simultaneously, achieving O(k) latency independent of sequence length).

---

## Core Insight

> *Physics is already a system that manages information.*

This project does not attempt to build a learning system that borrows metaphors from physics. The premise is simpler and stranger: physics already solves the problem of information—through entropy, conservation, diffusion, and equilibration. These are not analogies to computation; they *are* computation, operating at the most fundamental level.

The question then becomes: how much physics do you need to simulate before useful capabilities emerge? Not a gratuitous amount. Not a simplified toy model. Just enough to be functional—no more, no less. The hypothesis is that a system with sufficient thermodynamic structure will naturally equilibrate into states that exhibit what we recognize as intelligence: pattern completion, associative recall, generalization, adaptation.

We do not train this system. We do not optimize a loss function. We simulate physics, and the system settles into configurations that happen to be useful. What an outside observer calls "learning" is the system finding equilibrium. What they call "inference" is the system responding to perturbation.

The theoretical framework maps onto machine learning concepts, but the causality is reversed:

| Physics Term | ML Analogue | Key Difference |
|-------------|-------------|----------------|
| Oscillator | Input Token | Has phase/frequency; exists in continuous time |
| Mode (well) | Hidden State / Weight | A standing wave pattern that couples oscillators |
| Hamiltonian (H) | Loss Function | Conserved quantity; system minimizes potential V |
| Spectral Coupling | Attention Mechanism | Non-local entanglement via frequency resonance |
| Crystallization | Inference | Global parallel relaxation, not serial generation |
| Holographic CAM | Associative Memory | Content-addressable; partial input retrieves full pattern |
| Metabolism | Regularization | Modes decay if they do not receive energy |
| Universal Tokenizer | Embedding Layer | Deterministic hashing of raw bytes; no training |
| Phase Locking | Pattern Matching | Information encoded in relative phase angles |
| Symplectic Integrator | Optimizer | Preserves energy phase-space; no gradient descent |

---

## Architecture

The implementation consists of two primary layers:

```
thermo_manifold/
├── sensorium/
│   ├── kernels/            # GPU physics engine (the "substrate")
│   │   ├── gas_dynamics.py # Compressible ideal-gas Navier–Stokes (reference)
│   │   ├── metal/          # Apple Silicon (Metal Shading Language)
│   │   └── triton/         # CUDA (Triton JIT)
│   ├── dataset/            # Universal byte-stream datasets
│   ├── tokenizer/          # Universal byte→token hashing
│   ├── observers/          # Observer pattern (the "interface")
│   └── experiments/        # Paper/benchmark pipelines
│
└── optimizer/              # Temporary legacy shim for experiments (will be removed)
```

### The Kernels: GPU Physics Engine

The physics engine has two coupled domains:

**ThermodynamicsDomain (spatial / classical)**
- Particle↔grid coupling (PIC) into a periodic Eulerian grid
- Compressible **ideal-gas Navier–Stokes** update on the grid
- Grid→particle gather updates particle motion + heat proxy

**OmegaWaveDomain (ω-space / quantum-inspired coherence layer)**
- A fixed ω-lattice complex field \(\Psi(\omega_k)\)
- A dissipative **Gross–Pitaevskii-style** update with a kinetic (Laplacian) tunneling term
- Phase-torque feedback updates oscillator phases from \(\Psi\)

These two domains share the same oscillator frequencies \(\omega_i\) (excitations) and energies.

### The Sensorium: Observer Pattern

The `sensorium/` module provides a composable, fluent interface for observing simulation state. Think of it as a query language for inference:

```python
from sensorium.observers import Modes, Crystallized, TopK, Statistics, MODE_CRYSTALLIZED

# SQL-like composition:
# SELECT amplitude, phase WHERE state == CRYSTALLIZED
# ORDER BY amplitude DESC LIMIT 5

result = (
    Modes()
    .observe(state)
    .where(lambda m: m["state"] == MODE_CRYSTALLIZED)
    .sort_by("amplitude", descending=True)
    .take(5)
    .statistics("phase")
)
```

This decouples *what you observe* from *how the system evolves*. The observer decides what matters; the physics proceeds regardless.

---

## The Universal Tokenizer

All sensory modalities can be represented as spectral distributions:

- **Audio**: Energy over temporal frequencies (Hz)
- **Images**: Energy over 2D spatial frequencies (u, v)
- **Video**: Energy over 3D spatiotemporal frequencies (u, v, t)
- **Text**: Energy over semantic embedding dimensions

The Universal Tokenizer maps raw bytes to oscillator frequencies via deterministic hashing:

```
ID = h(Byte, Index) mod N
```

No learned embeddings. No tokenizer training. Adding a new modality requires only a spectral encoder (decomposition) and decoder (reconstruction). Cross-modal relationships emerge from Hebbian co-activation, not architectural coupling.

---

## Physics Implementation

### Spatial Dynamics

The spatial substrate is a compressible ideal-gas model (periodic grid) coupled to particles via PIC.

At a high level, the grid evolves conserved quantities \((\rho, \rho \mathbf{u}, E)\) under an ideal-gas EOS and explicit time stepping; particles deposit/gather these fields with trilinear/CIC weights.

### Hydrodynamic ω-Field (Quantum Coherence)

We evolve a complex wavefunction over a fixed ω-lattice:

$$
i\hbar \,\partial_t \Psi(\omega) =
\left(
-\frac{\hbar^2}{2m}\nabla_\omega^2
 + V_{\text{ext}}(\omega)
 + g|\Psi(\omega)|^2
 - \mu
\right)\Psi(\omega)
$$

Discretely, \(\nabla_\omega^2\) is a 1D Laplacian on neighboring ω-bins. This provides:
- **superposition & interference** (complex phase dynamics)
- **tunneling through ω-space** (kinetic term couples neighbors)
- **soliton-like attractors** when the nonlinearity and kinetic term balance

Implementation detail: the Metal and Triton kernels use a **symmetric split-step** (Strang-style) ordering to improve phase fidelity.

### Thermal-Oscillator Equilibrium

Heat Q and oscillator energy E_osc exchange bidirectionally toward the Planck distribution:

$$E_{\text{osc,eq}}(\omega, T) = \frac{\hbar\omega}{\exp(\hbar\omega / k_B T) - 1}$$

This quantum-inspired equilibrium ensures that high-frequency modes require more energy to excite—a natural regularization.

### Homeostatic Regulation

The homeostatic ratio prevents runaway dynamics:

$$\rho = \frac{\log(1 + E_{\text{total}})}{\log(1 + \mathcal{B}) + \varepsilon}$$

where B is an exponential moving average baseline. When ρ > 1, the system is "overheated" and damping increases. When ρ < 1, the system is "cold" and damping decreases. No learned parameters—self-regulation emerges from the dynamics.

---

## Inference as Boundary Value Problem

Classical autoregressive inference: clamp prefix, generate suffix token-by-token.

Crystallization inference: clamp *any* subset of oscillators (the "boundary conditions"), relax *all others* in parallel toward the nearest energy minimum.

```python
from sensorium.observers import infer, Modes, TopK

# Inject query as "dark particles" (invisible to regular observers)
# System responds; observe the crystallized modes
result = infer(
    b"The capital of France is",
    Modes(),
    TopK(5, by="amplitude"),
    steps=10,
).observe(state, manifold=manifold)
```

**Dark particles** perturb the system but do not couple to modes. They are filtered from all observations. The observable response *is* the inference result.

This enables:
- **Causal generation**: Clamp past, relax future
- **Inpainting**: Clamp start/end, relax middle
- **Super-resolution**: Clamp low-frequency, relax high-frequency
- **Semantic constraint**: Clamp specific mode to high amplitude

All modes use the same physics. The observer decides which constraints to impose.

---

## GPU Backends

### Metal (Apple Silicon)

`sensorium/kernels/metal/manifold_physics.metal` — Metal Shading Language kernels:

- Particle-to-field scatter (gravity potential ρ, temperature field T)
- FFT-based Poisson solver for gravitational forces
- Hardware texture3D sampler for trilinear field interpolation
- Velocity Verlet integration with Strang-split drag
- Adaptive 2-pass statistics reduction (mean, std, energy)
- Fail-fast NaN detection

### Triton (CUDA)

`sensorium/kernels/triton/` — Triton JIT kernels:

- Periodic boundary conditions (torus topology)
- Poisson-Jacobi iteration for gravity
- Temperature-dependent viscosity: μ(T) = μ_ref · √T
- Stefan-Boltzmann radiation: P = εσAT⁴
- Newton's law of cooling: Q̇ = 4πκr(T_env - T_particle)
- Physics-based thermalization timescale: τ = (m·c_v)/(4πκr)
- GPU parallel reduction for statistics

Both backends share the same physics; dispatch is determined at configuration time.

---

## Installation

### Requirements

- Python 3.10+ (3.12 recommended)
- PyTorch 2.0+
- Apple Silicon (Metal) or CUDA GPU (Triton)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Quick Setup

```bash
git clone https://github.com/theapemachine/thermo_manifold.git
cd thermo_manifold
make install
```

### Manual Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv sync --extra dev --extra viz
```

---

## Usage

### Basic Simulation

```python
from optimizer.manifold import Manifold, GeometricSimulationConfig

config = GeometricSimulationConfig(
    grid_dims=(64, 64, 64),
    particle_count=10000,
    dt=0.001,
)

manifold = Manifold(config)
state = manifold.build_initial_state()

for _ in range(1000):
    state = manifold.step(state)
```

### Observation

```python
from sensorium.observers import Particles, Modes, Statistics

# Observe particle energy distribution
energy_stats = (
    Particles()
    .observe(state)
    .statistics("energy")
)

# Observe crystallized modes
crystallized = (
    Modes()
    .observe(state)
    .where(lambda m: m["state"] == 2)
    .top_k(10, by="amplitude")
)
```

### Inference via Dark Particle Injection

```python
from sensorium.observers import infer, Modes, TopK

result = infer(
    b"query input bytes",
    Modes(),
    TopK(5, by="amplitude"),
    steps=10,
).observe(state, manifold=manifold)

# Result contains top-5 coupled modes by amplitude
for mode in result.get():
    print(f"ω={mode['omega']:.2f}, A={mode['amplitude']:.4f}")
```

---

## Experiments

The `sensorium/experiments/` directory contains 20+ experiment modules:

| Experiment | Description |
|-----------|-------------|
| `kernel_next_token.py` | Next-byte prediction (text) |
| `kernel_image_gen.py` | Image generation |
| `kernel_audio_gen.py` | Audio synthesis from coherence modes |
| `kernel_text_diffusion.py` | Text diffusion via boundary conditions |
| `kernel_cross_modal.py` | Text ↔ Audio transduction |
| `kernel_rule_shift.py` | Online adaptation to distribution shift |
| `kernel_cocktail_party.py` | Source separation |
| `kernel_timeseries.py` | Time series prediction |

---

## Theoretical Foundation

See `The_Sensorium_Manifold.pdf` for the complete theoretical treatment, including:

- Spectral Isomorphism Principle (Principle 1)
- Hamiltonian formulation and symplectic integration
- Proof of holographic content-addressability
- Crystallization convergence analysis
- Experimental validation across modalities

---

## Citation

```bibtex
@article{vandommelen2026sensorium,
  title={The Sensorium Manifold: Native Multimodality via Isomorphism},
  author={van Dommelen, Daniel Owen},
  journal={Independent Research},
  year={2026},
  url={https://github.com/theapemachine/thermo_manifold}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>"Physics already manages information. Simulate enough of it, and what emerges is what we've been calling AI."</i>
</p>
