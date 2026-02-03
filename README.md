# Thermo Manifold

**Thermodynamic Primitives for Neural Computation Without Backpropagation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

Thermo Manifold is a reference implementation exploring an alternative paradigm for machine learning: **thermodynamic computation**. Instead of relying on backpropagation and gradient descent, this framework models learning and inference as physical processes—energy flow, heat diffusion, and homeostatic regulation—operating on sparse graph structures.

The core insight is that learning can emerge from local, Hebbian-style dynamics governed by thermodynamic principles, enabling:

- **Online, continuous learning** from streaming data
- **Emergent structure formation** without explicit supervision
- **Multi-modal transduction** between semantic and spectral domains
- **Self-organization** through energy minimization and homeostasis

This repository accompanies our paper and serves as the canonical implementation for reproducing experimental results.

---

## Key Concepts

### Thermodynamic Learning

Traditional neural networks compute gradients through the entire network via backpropagation. Thermo Manifold takes a different approach: each component operates as a thermodynamic system where:

- **Energy** represents activation strength and information salience
- **Heat** captures uncertainty and exploratory noise
- **Homeostasis** maintains stable operating regimes through adaptive baselines
- **Surprise** modulates plasticity, enabling rapid adaptation to novel patterns

### Sparse Bond Graphs

Rather than dense weight matrices, relationships are encoded in sparse directed graphs:

```
Token A ──[bond strength]──▶ Token B
```

Bonds strengthen through co-activation and weaken through disuse, implementing a form of Hebbian learning. This representation scales efficiently and naturally captures the sparsity of real-world sequential patterns.

### Hierarchical Abstraction

The system discovers compositional structure through **chunk formation**:

1. Frequently co-occurring token sequences condense into chunks
2. Chunks participate in higher-level bond graphs
3. Top-down biases from chunks guide token-level predictions

This emerges from binding energy dynamics rather than explicit segmentation algorithms.

---

## Architecture

Thermo Manifold consists of four interconnected manifolds:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT (Tokens)                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC MANIFOLD                            │
│  • Sparse bond graphs (token → token)                           │
│  • Thermodynamic flow propagation                               │
│  • Surprise-driven plasticity                                   │
│  • Idle pondering (transitive closure, dream rollouts)          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               HIERARCHICAL SEMANTIC MANIFOLD                    │
│  • Variable-length chunks (2-4 tokens)                          │
│  • Chunk ↔ token bipartite bonds                                │
│  • Binding energy condensation                                  │
│  • Multi-resolution representation                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BRIDGE MANIFOLD                             │
│  • Semantic vectors ↔ spectral frequencies                      │
│  • Carrier population (not lookup tables)                       │
│  • Co-activation learning                                       │
│  • Event-horizon locality                                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SPECTRAL MANIFOLD                            │
│  • 1D thermodynamic diffusion                                   │
│  • Frequency attractor dynamics                                 │
│  • Audio synthesis from spectral energy                         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT (Audio)                            │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

| Manifold | Purpose | Key Innovation |
|----------|---------|----------------|
| **SemanticManifold** | Token-level sequential grammar | Sparse bond graphs with thermodynamic flow |
| **HierarchicalSemanticManifold** | Multi-scale abstraction | Emergent chunk formation via binding energy |
| **BridgeManifold** | Cross-modal transduction | Carrier-based coupling without lookup tables |
| **SpectralManifold** | Audio generation | Particle diffusion toward frequency attractors |

---

## Installation

### Requirements

- Python 3.10+ (3.12 recommended)
- PyTorch 2.0+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/theapemachine/thermo_manifold.git
cd thermo_manifold

# Install using make (creates venv and syncs dependencies)
make install
```

### Installation Options

```bash
make install          # Base dependencies only
make install-dev      # Include development tools (pytest, black, mypy, etc.)
make install-viz      # Include visualization (matplotlib)
make install-all      # Install everything
```

### Manual Setup

If you prefer not to use `make`:

```bash
python3.12 -m venv .venv
. .venv/bin/activate && uv sync

# With optional dependencies
. .venv/bin/activate && uv sync --extra dev --extra viz
```

---

## Quick Start

### Basic Usage

```python
from thermo_manifold import SemanticManifold, PhysicsConfig

# Initialize configuration
config = PhysicsConfig(
    dt=0.02,                        # Integration timestep
    tau=0.95,                       # Homeostasis time constant
    dream_sampling_temperature=1.0, # Exploration temperature
    dream_energy_budget=100,        # Pondering compute budget
)

# Create semantic manifold
manifold = SemanticManifold(
    vocab_size=10000,
    embed_dim=256,
    config=config,
)

# Training: observe token sequences
for sequence in training_data:
    manifold.observe(sequence)

# Inference: predict next token
context = [token_a, token_b, token_c]
prediction = manifold.predict(context)
```

### End-to-End Demo

```bash
python -m thermo_manifold.demos.unified_demo
```

This demonstrates the complete pipeline: text tokens → semantic manifold → bridge → spectral manifold → audio synthesis.

### Rule-Shift Benchmark

```bash
python -m thermo_manifold.demos.rule_shift_demo
```

Evaluates adaptation to distributional shifts, with diagnostic output for analysis.

---

## Project Structure

```
thermo_manifold/
├── __init__.py                 # Package exports
├── core/
│   ├── config.py              # PhysicsConfig, PhysicsMedium
│   ├── state.py               # BatchState container
│   ├── scatter.py             # Scatter operations (sum, max, softmax)
│   ├── diagnostics.py         # Logging (CSV/JSONL)
│   └── viz.py                 # Visualization utilities
├── physics/
│   └── engine.py              # ThermodynamicEngine base class
├── semantic/
│   ├── manifold.py            # SemanticManifold (token-level)
│   ├── hierarchical.py        # HierarchicalSemanticManifold
│   ├── bond_graph.py          # SparseBondGraph
│   ├── bipartite_graph.py     # SparseBipartiteBondGraph
│   ├── chunk_store.py         # ChunkStore (variable-length sequences)
│   └── radix_trie.py          # RadixTrie for efficient chunk lookup
├── bridge/
│   └── manifold.py            # BridgeManifold (semantic ↔ spectral)
├── spectral/
│   └── manifold.py            # SpectralManifold (audio synthesis)
├── demos/
│   ├── unified_demo.py        # End-to-end demonstration
│   └── rule_shift_demo.py     # Benchmark with diagnostics
└── tests/
    ├── test_bridge_heat.py
    ├── test_hunger.py
    ├── test_multi_resolution_chunks.py
    └── test_transitive_closure.py
```

---

## Configuration

The `PhysicsConfig` class controls all thermodynamic parameters:

```python
from thermo_manifold import PhysicsConfig, PhysicsMedium

config = PhysicsConfig(
    # Core dynamics
    dt=0.02,                # Integration timestep
    tau=0.95,               # Homeostasis decay (higher = slower adaptation)
    eps=1e-8,               # Numerical stability
    
    # Idle pondering
    dream_sampling_temperature=1.0,  # Exploration randomness
    dream_energy_budget=100,         # Compute budget for pondering
    
    # Carrier dynamics (for BridgeManifold)
    carrier_tau=0.9,        # Per-carrier homeostasis
    
    # Physical medium properties
    medium=PhysicsMedium(
        thermal_resistance=0.1,
        viscosity=0.05,
        diffusion_rate=0.01,
    ),
)
```

---

## Physics Model

The core of Thermo Manifold is a rigorous thermodynamic simulation. This is not metaphor—it is accurate physics. All learning emerges from these fundamental rules.

### Core Entities

| Entity | Description |
|--------|-------------|
| **Particle** | A token/concept. Has position (embedding), energy, heat, and excitation. |
| **Carrier** | Intermediary node mediating transport. Has position, energy, and heat. |
| **Bond** | Connection between particle and carrier. Determines flow distribution. |

### The Thermodynamic Cascade

**Heat is the transport mechanism.** Heat carries energy as it flows. The cascade:

```
Heat arrives → Energy transfers → Temperature rises → Excitation rises → Heat generated → Heat flows out
     ↑                                                                                           ↓
     └───────────────────────────────────────────────────────────────────────────────────────────┘
```

This cycle applies to **every entity** (particles, carriers). Each step:

1. Heat arrives at entity
2. Energy transfers with heat
3. Temperature rises (kinetic energy = heat)
4. Excitation rises with temperature
5. Excitation generates heat
6. Heat flows out via bonds (split by attraction/gravity)

### Rule 1: Particles Bond to Carriers via Attraction

Bonds form based on **attraction** (cosine similarity = gravity):

```
attraction = (particle_position · carrier_position + 1) / 2
```

Attraction determines how heat/energy **splits** when flowing to multiple targets.

### Rule 2: Heat Flows via Bonds

Heat flows from hot to cold, carrying energy with it. Flow is split by attraction:

```
For particle p with heat H bonded to carriers c₁, c₂, c₃:

heat_to_c₁ = H × dt × (attraction(p, c₁) / total_attraction)
heat_to_c₂ = H × dt × (attraction(p, c₂) / total_attraction)
heat_to_c₃ = H × dt × (attraction(p, c₃) / total_attraction)
```

**Energy transfers with heat** in the same proportions.

### Rule 3: Temperature Drives Excitation

Temperature = heat (assuming unit mass). Excitation rises with temperature:

```
excitation += temperature × dt
```

### Rule 4: Excitation Generates Heat

Excitation converts to heat (excitation is consumed in the process):

```
heat_generated = excitation × dt
excitation -= heat_generated
heat += heat_generated
```

This is self-limiting: high excitation generates heat quickly, depleting itself.

### Rule 5: Carriers Follow the Same Physics

Carriers are not passive conduits. They have their own:
- Heat (which drives their excitation)
- Excitation (which generates more heat)
- Energy (which transfers with heat flow)

The cascade continues: particle → carrier → particle → carrier → ...

### Rule 6: Heat Decreases Viscosity

Hot entities move faster in embedding space:

```
effective_viscosity = base_viscosity / (1 + temperature)
```

### Rule 7: Bonds Snap When No Heat/Energy Flows

Bonds break when nothing flows through them:

```
if last_heat_flow == 0 and last_energy_flow == 0:
    bond.snap()
```

### Rule 8: Conservation Laws

**Energy is conserved**: Total energy (all particles + all carriers + all heat) only increases from external input.

**Heat only increases** (second law): Processes generate heat; heat cannot spontaneously convert back to usable energy.

### Rule 9: Homeostasis

Adaptive baselines prevent runaway dynamics:

```
homeostasis_ratio = current_energy / baseline_energy
baseline_energy ← τ × baseline_energy + (1-τ) × current_energy
```

### The Complete Cycle

Each timestep:

1. **Input** → External data injects energy/heat into active particles
2. **Bond Formation** → Active particles bond to carriers (attraction = gravity)
3. **Particle Thermodynamics** → Temperature → excitation → heat generation
4. **Heat Flows Out** → Heat flows via bonds to carriers (split by attraction)
5. **Energy Flows With Heat** → Energy transfers with heat
6. **Carrier Thermodynamics** → Carriers process heat → excitation → more heat
7. **Heat Flows to Particles** → Heat flows back to all bonded particles
8. **Bond Snapping** → Dead bonds (no flow) break
9. **Repeat** → Continuous cascade through the network

This creates a self-organizing system where structure emerges from thermodynamic flow patterns.

---

## Design Principles

1. **No Backpropagation**: All updates are local and Hebbian-style
2. **Sparse Structures**: Avoid dense V×V matrices; use directed graphs
3. **Emergent Behavior**: Structure forms from dynamics, not design
4. **Online Learning**: Continuous adaptation to streaming data
5. **Homeostatic Regulation**: Adaptive baselines prevent runaway activation
6. **Event-Horizon Locality**: Efficient neighbor search for scalability
7. **Scale-Free Operation**: Adaptive normalization, no hard thresholds

---

## Idle Pondering

A distinctive feature of Thermo Manifold is **idle pondering**—computation that occurs between observations to consolidate knowledge:

| Mechanism | Description | Benefit |
|-----------|-------------|---------|
| **Transitive Closure** | Infers shortcuts (A→C from A→B, B→C) | Accelerates future traversals |
| **Conflict Resolution** | Resolves ambiguous predictions | Improves consistency |
| **Dream Rollouts** | Explores hypothetical sequences | Discovers dead-ends, builds hunger signals |

This enables the system to improve without additional training data, similar to memory consolidation during sleep.

---

## Benchmarks

### Rule-Shift Adaptation

The `rule_shift_demo.py` evaluates how quickly the system adapts when underlying patterns change:

```
Step 1000: Pre-shift accuracy: 94.2%
Step 1001: Rule shift applied
Step 1010: Post-shift accuracy: 67.3%
Step 1050: Recovered accuracy: 89.1%
Step 1100: Recovered accuracy: 93.8%
```

Diagnostic metrics are logged to CSV/JSONL for analysis.

---

## Diagnostics

Enable comprehensive logging:

```python
from thermo_manifold.core.diagnostics import DiagnosticsLogger

logger = DiagnosticsLogger(
    output_dir="./logs",
    format="jsonl",  # or "csv"
)

manifold.attach_diagnostics(logger)
```

Captured metrics include:
- Bond graph statistics (edges, mean strength, pruning rate)
- Energy and heat distributions
- Chunk formation events
- Pondering outcomes (shortcuts found, dead-ends explored)

---

## Citation

If you use Thermo Manifold in your research, please cite:

```bibtex
@article{thermomanifold2026,
  title={Thermodynamic Primitives for Neural Computation Without Backpropagation},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2026},
  url={https://github.com/theapemachine/thermo_manifold}
}
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install with development dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linter
make lint

# Type checking
make typecheck

# Run all checks
make check
```

Run `make help` to see all available commands.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on insights from thermodynamic computing, Hebbian learning theory, and sparse representation research. We thank [acknowledgments] for valuable discussions and feedback.

---

<p align="center">
  <i>Learning as physics, not optimization.</i>
</p>
