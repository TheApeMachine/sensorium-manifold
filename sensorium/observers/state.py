import torch
from sensorium.kernels.metal.manifold_physics import ThermodynamicsDomain
from sensorium.observers.base import ObserverProtocol


class StateObserver(ObserverProtocol):
    def __init__(self, geometric_state: ThermodynamicsDomain = None):
        self.geometric_state = geometric_state
        self.state_dict = {}

    def observe(self, observation=None, **kwargs):
        # If observation is provided, merge it with our state.
        #
        # IMPORTANT: This observer is used inside `InferenceObserver`, which expects
        # each stage to return a dict-like state. Never return scalars here.
        if isinstance(observation, dict):
            self.state_dict.update(observation)

            # Optional: if mode fields are present, derive a simple
            # confidence proxy (used by some dashboards/experiments).
            amp = observation.get("amplitudes")
            cs = observation.get("mode_state")
            conf = observation.get("conflict")
            if amp is not None and cs is not None:
                amp_slice = amp[:10].to(torch.float32)
                cs_slice = cs[:10]
                eps = 1e-8
                total_amp = amp_slice.sum()
                crystallized_amp = (amp_slice * (cs_slice == 2).to(torch.float32)).sum()
                amp_ratio = crystallized_amp / (total_amp + eps)
                mean_conflict = (
                    conf[:10].to(torch.float32).mean()
                    if conf is not None
                    else torch.tensor(0.0, device=amp_slice.device)
                )
                score = amp_ratio * (1.0 - torch.clamp(mean_conflict, 0.0, 1.0))
                try:
                    self.state_dict["confidence"] = float(score.detach().item())
                except Exception:
                    # Best-effort only; keep previous confidence on failure.
                    pass
        elif isinstance(observation, tuple):
            # Handle step() return value (positions, velocities, energies, heats, excitations)
            # Convert to dict for easier handling (if we need it later).
            pass

        
        return {
            "readiness": self.readiness,
            "stability": self.stability,
            "coherence": self.coherence,
            "complexity": self.complexity,
            "entropy": self.entropy,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "done_thinking": False,  # Default: continue thinking
            "confidence": self.confidence
        }

    @property
    def readiness(self):
        return self.state_dict.get("readiness", 0.0)

    @property
    def stability(self):
        return self.state_dict.get("stability", 0.0)

    @property
    def coherence(self):
        return self.state_dict.get("coherence", 0.0)

    @property
    def complexity(self):
        return self.state_dict.get("complexity", 0.0)

    @property
    def entropy(self):
        return self.state_dict.get("entropy", 0.0)

    @property
    def temperature(self):
        return self.state_dict.get("temperature", 0.0)

    @property
    def pressure(self):
        return self.state_dict.get("pressure", 0.0)
    
    @property
    def confidence(self):
        return self.state_dict.get("confidence", 0.0)