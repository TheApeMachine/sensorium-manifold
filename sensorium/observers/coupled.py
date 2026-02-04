"""Coupled observer is specialized in observing across domains

Use it to make observations between the geometric and spectral domains,
based on either a particle or a oscillator.
"""

from sensorium.observers.base import ObserverProtocol

class CoupledObserver(ObserverProtocol):
    def __init__(self, observer: ObserverProtocol):
        self.observer = observer

    def observe(self, observation=None, **kwargs):
        """Observe the state of the system through the coupled observer."""