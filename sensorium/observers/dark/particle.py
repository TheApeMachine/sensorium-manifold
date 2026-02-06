from sensorium.tokenizer.universal import UniversalTokenizer
from sensorium.observers.types import ObserverProtocol


class DarkParticleObserver(ObserverProtocol):
    def __init__(self, simulation, tokenizer: UniversalTokenizer):
        self.simulation = simulation
        self.tokenizer = tokenizer
        self.state = TensorDict()

    def observe(self, state: dict) -> dict:
        for token in self.tokenizer.stream():
            self.state = self.simulation.step(token)
        
        return state