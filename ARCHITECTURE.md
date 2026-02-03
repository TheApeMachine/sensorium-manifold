# The Sensorium Manifold Architecture

## Core Philosophy

**The universe is an information processing system. Physics is the algorithm.**

We don't invent rules — we simulate the rules that already work. The reason this approach can succeed is because it's already working: the universe, at every scale, transfers information from object to object through physical processes.

## The Manifold Cosmology

> [!NOTE]
> Particles and Oscillators are the same thing, their presented type is inherent to the domain you are observing.
> 

### T=0: The Singularity
- No particles, no carriers, no space
- Zero dimension, zero energy
- Pure potential

### T=1: The Big Bang (First Input)
- First input creates the first particle
- Energy enters the system
- Space *exists* now because there's something to have position
- With only one particle, there's no distance, no structure — still essentially a point

### T=2+: Expansion and Structure
- Each new input:
  - Creates a new particle
  - Adds new energy
  - *Expands the "universe"*
- Gravity pulls particles toward energy concentrations
- Heat disperses, pushing outward
- **Structure emerges** from the battle between gravity and heat

## Two Influence Channels

The manifold has two distinct but implicitely interacting channels:

### 1. Entanglement (Carriers)

A strong informational "bridge" between particles independent of **geometric distance**.

- **Frequency** represents relatedness and governs selection
- **Phase** represents alignment and governs influence
- Carriers are pulse waves that travel and sample/deliver at each pulse edge
- Each pulse is a two-way street:
  1. First, it acts on oscillators with the sum of sines sampled in the previous step
  2. Then, it samples from the oscillators (sum of sines at the sampling point)

This enables non-local correlation — particles can influence each other regardless of their geometric position.

### 2. Physical Space Layer (Thermodynamics)

A secondary influence between particles based on **geometric distance**.

The laws of thermodynamics govern these rules:

1. **Heat transfers Energy** — heat is the transport mechanism
2. **Heat travels from hot to cold** — temperature gradients drive flow
3. **Energy transfers into Particle** — when heat arrives, energy comes with it
4. **Temperature rises (Kinetic Energy)** — energy density increases temperature
5. **Excitation rises** — temperature drives excitation (frequency)
6. **Energy converts to Heat** — excitation generates heat, completing the cycle

### The Battle: Gravity vs Heat

**Gravity** pulls particles toward energy/mass concentrations.

**Heat** disperses energy away from concentrations (thermal pressure).

This is why the universe isn't one big clump:
- Gravity alone → collapse to singularity
- Heat provides counterbalancing pressure
- Dynamic equilibrium → stable structure (galaxies, stars, concepts)

The system settles when: `Gravitational influx = Heat outflux`

## Particles

### Identity
```
particle_id = hash(byte_content + sequence_index)
```

This gives:
- **Same content + same position → Same particle ID**
- **Same content + different position → Different particle ID**
- **Different content + same position → Different particle ID**

### Cross-File Collision = Compression

When the same content appears at the same position in different files:
- File 1, position 0: `[0,0,0,0,0,0,0,0]` (black) → Particle X
- File 2, position 0: `[0,0,0,0,0,0,0,0]` (black) → Particle X

Same particle! The manifold doesn't store redundant data — it just sees:
- "Particle X appeared again"
- "Particle X gets more energy"
- "Particle X's position is reinforced by gravity"

The **count of excitations** encodes frequency. The **gravitational position** encodes relationships. This IS compression, emergent from the physics.

### Pure Bytes as Tokens

No tokenizer. No vocabulary. No encoding/decoding.

Every byte maps to one of 256 possible values. Combined with position hashing, this creates a content-addressable particle space.

The same manifold can process:
- Text (UTF-8 bytes)
- Images (pixel bytes)
- Audio (sample bytes)
- Video (frame bytes)
- Executables (machine code bytes)
- Anything stored on a computer

## Observations

Observations are "output ports" — ways to observe the manifold and assign meaning to what's observed.

**Critical principle: Outputs measure resonance; they do not participate in it.**

No gradients, no reward shaping, no feedback into dynamics. Tasks are *consumers*, not *drivers*.

### Task-Specific Readouts

The same physics engine supports multiple observation types:

| Task | What to Observe |
|------|-----------------|
| Next-byte prediction | Which of 256 particles has highest energy/excitation |
| Classification | Which attractor basin has most energy |
| Generation | All particle states, sample most excited, feed back |
| Audio synthesis | Oscillation waveforms (phase + amplitude) |

There are no predetermined outputs. Whatever the prediction task needs determines what needs to be observed.

### Internal Observations

The system can make observations about itself and feed back into itself. This enables:
- Idle-time compute
- Dreaming
- Reasoning
- Self-reflection

## Time Control = Prediction Power

In the real world:
- We're stuck at t=now
- We can only observe the past
- Prediction = extrapolating forward (hard)

In the simulation:
- We control the clock
- We can run the system forward
- **Prediction = reading the future state we've already computed**

"Next token prediction" becomes:
1. Feed input (present)
2. Run physics forward
3. Observe where energy settles
4. That's the prediction

We're not guessing — we're letting physics compute the answer.

## Minimal Viable Physics

The complexity of physics should match the complexity of the task:

| Task | Information | Physics Needed |
|------|-------------|----------------|
| Next token (1 of ~50k) | ~16 bits | Minimal |
| Digit classification (1 of 10) | ~3.3 bits | Less |
| Image generation | Millions of bits | More |
| Audio waveform | Continuous | Smooth dynamics |

### Required Components

- [ ] Particles with position, energy, heat, excitation (frequency)
- [ ] Gravity between particles (1/r² based on energy/mass)
- [ ] Heat generation from excitation
- [ ] Heat flow based on temperature gradient and distance
- [ ] Energy transport with heat
- [ ] Carriers for non-local coupling (frequency matching, phase alignment)

Everything else is either derivable from these or unnecessary overhead.

## No Bonds Needed

Gravity does all the work. Instead of maintaining an explicit bond graph:

1. Particles have positions (emergent from dynamics)
2. Gravity determines energy distribution — closer particles get more
3. No bond storage, snapping, or decay logic needed
4. "Connectivity" is emergent from positions at each timestep

The geometric influence channel (gravity + heat) handles:
- Attraction toward related particles
- Pressure preventing collapse
- Natural clustering into structure

The entanglement channel (carriers) handles:
- Non-local relationships
- Frequency-based selection
- Phase-based influence modulation

## Emergent Properties

From these simple rules, we expect:

1. **Compression** — redundant patterns map to same particles
2. **Clustering** — related patterns gravitate together
3. **Structure** — gravity vs heat creates stable configurations
4. **Generalization** — physics generalizes, so the manifold generalizes
5. **Multi-modal** — bytes are bytes, regardless of source
6. **Scale-free** — universe grows with input, no fixed capacity

## Parallel Loading

### Position Is In The ID

Because `particle_id = hash(byte + position)`, load order doesn't matter.

```
Frame 0, bytes 0-99:     particles with positions 0-99
Frame 0, bytes 100-199:  particles with positions 100-199
...
Frame 0, bytes 900-999:  particles with positions 900-999
```

Load all buckets simultaneously. Each particle knows its position because it's encoded in the ID.

### The Manifold Reconstructs Order

When particles arrive:
- They carry their position in their identity
- Gravity pulls them toward related particles
- Carriers couple them based on frequency/phase
- **The physics reconstructs sequential structure**

It's like dumping puzzle pieces on a table — they don't need to arrive in order because each piece knows where it fits.

### Cross-File/Frame Parallelism

```
Frame 0:  [all bytes]  → parallel
Frame 1:  [all bytes]  → parallel
Frame 2:  [all bytes]  → parallel
```

As long as particle IDs encode byte value + position + frame index, you can load everything in parallel. Same byte at same position in same frame = same particle = compression.

## Field-Based Physics (No Pairwise Computation)

### Why Not Pairwise?

Naive gravity: compute force between every pair of particles. That's O(N²) and requires all particles present.

But real physics doesn't work that way.

### How Real Physics Works

1. **Fields** — each mass contributes to a gravitational field
2. **Locality** — particles respond to the local field, not to individual other particles
3. **Superposition** — fields add linearly, contributions accumulate incrementally

### Incremental Field Updates

```
Particle arrives → Updates field at its position → Responds to existing field
```

No need to wait for all particles. The field IS the accumulated state.

### Fields In The Manifold

| Physical Quantity | Field Representation                                                    |
|-------------------|-------------------------------------------------------------------------|
| **Gravity**       | Gravitational potential field — particles add mass, respond to gradient |
| **Heat**          | Temperature field — particles add heat, heat diffuses through field     |
| **Carriers**      | Wave field — particles couple via frequency/phase, carriers propagate   |

### Continuous Processing

```
Stream of bytes (parallel or sequential, doesn't matter)
    ↓
Each byte → particle → updates fields (gravity, heat, carrier)
    ↓
Fields evolve continuously (diffusion, wave propagation)
    ↓
Particles respond to local field values
```

**No synchronization barrier needed.** Particles can arrive continuously, fields accumulate, physics runs.

### GPU-Perfect Parallelism

- Parallel hash computation for all bytes
- Parallel particle creation
- Parallel field updates (atomic adds)
- Parallel field evolution (stencil operations)
- Parallel particle response to fields
- **No sequential bottleneck anywhere**

The only "sequential" aspect is causality in the physics timesteps — but within each timestep, everything is massively parallel.

## Implementation: Fused Metal/Triton Kernels

The physics simulation is implemented with fused GPU kernels for maximum performance.

### Kernel Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    manifold_step (per timestep)                    │
├────────────────────────────────────────────────────────────────────┤
│  KERNEL 1: scatter_particle                                        │
│    IN:  particle_pos[N,3], particle_mass[N], particle_heat[N]      │
│    OUT: gravity_field[X,Y,Z], heat_field[X,Y,Z]                    │
│    OPS: trilinear weights → atomic add to 8 corners                │
├────────────────────────────────────────────────────────────────────┤
│  KERNEL 2: poisson_jacobi_step (repeated for convergence)          │
│    IN:  gravity_field (mass density), phi_in (potential)           │
│    OUT: phi_out (updated potential)                                │
│    OPS: 6-point stencil, Jacobi iteration                          │
├────────────────────────────────────────────────────────────────────┤
│  KERNEL 3: diffuse_heat_field                                      │
│    IN:  temperature_in                                             │
│    OUT: temperature_out                                            │
│    OPS: 6-point Laplacian, explicit Euler diffusion                │
├────────────────────────────────────────────────────────────────────┤
│  KERNEL 4: gather_update_particles (FUSED - highest impact)        │
│    IN:  gravity_potential, temperature_field, particle_state       │
│    OUT: updated particle_state                                     │
│    OPS: trilinear gather, gradient computation, all physics:       │
│         - Force from gravity gradient                              │
│         - Heat exchange from temperature                           │
│         - Excitation from temperature                              │
│         - Energy → heat conversion                                 │
│         - Viscosity (heat-dependent)                               │
│         - Velocity integration                                     │
│         - Position integration                                     │
│    All in ONE kernel, ONE read per field, minimal memory traffic   │
├────────────────────────────────────────────────────────────────────┤
│  KERNEL 5: carrier_oscillator_coupling                             │
│    IN:  oscillator phases/freqs/amps, carrier state, sparse P      │
│    OUT: updated carrier amplitudes                                 │
│    OPS: gated sampling, tuning strength, damped oscillator         │
├────────────────────────────────────────────────────────────────────┤
│  KERNEL 6: update_oscillator_phases                                │
│    IN:  carrier state, oscillator state, sparse P                  │
│    OUT: updated oscillator phases                                  │
│    OPS: carrier back-influence, phase integration                  │
└────────────────────────────────────────────────────────────────────┘
```

### Files

- `optimizer/metal/manifold_physics.metal` — Metal shader implementations
- `optimizer/metal/manifold_physics.py` — Python wrapper with PyTorch fallbacks
- `optimizer/metal/manifold_physics_test.py` — Unit tests

### Key Optimizations

1. **Fused gather-update**: One kernel reads all fields and applies all particle physics
2. **Trilinear interpolation**: Smooth field sampling, computed inline
3. **Atomic scattering**: Particles contribute to fields in parallel
4. **Structure of Arrays**: Coalesced memory access on GPU
5. **PyTorch fallback**: Works without Metal compilation for development

### Usage

```python
from optimizer.metal.manifold_physics import ManifoldPhysics, ManifoldPhysicsConfig

config = ManifoldPhysicsConfig(
    grid_size=(64, 64, 64),
    dt=0.01,
    gravity_strength=1.0,
    heat_diffusion=0.1,
)
physics = ManifoldPhysics(config, device="mps")

# Each timestep:
positions, velocities, energies, heats, excitations = physics.step(
    positions, velocities, energies, heats, excitations, masses
)
```

## Summary

The manifold is a physics simulation where:
- Inputs become particles
- Particles interact via gravity and carriers
- Heat provides pressure against collapse
- Structure emerges from dynamic equilibrium
- Observations read out predictions without influencing dynamics
- Time control allows computing the future before reading it

The key insight: we're not learning patterns — we're simulating a universe where patterns naturally emerge from the physics.
