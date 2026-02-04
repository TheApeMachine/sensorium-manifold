"""Inference observer for the manifold

This is a more general, more capable observer which is used to observe
the manifold in a way that is equivalent to inference.
The difference is that inference means flexibility, and choice of what to
observe, how to observe it, and deciding what the observation means to you.
To make it easier to reason about, start by thinking that observations are
determined by the task you are trying to solve.
"""

from sensorium.observers.base import ObserverProtocol


class InferenceObserver(ObserverProtocol):
    """Inference observer is essentially a pipeline of observers

    Each observer in the pipeline sends its observation to the next one.
    Given inference is essentially a non-predetermined selection of possible
    operations on the manifold, this makes it easier to design your own
    inference mechanism on-the-fly.

    One quick and dirty way to define your own custom observers for the 
    pipeline is to simply pass a first-order object with an `observe` method.
    """
    def __init__(self, observers: list[ObserverProtocol]):
        self.observers = observers
        self.state = None

    def observe(self, observation=None, **kwargs):
        # Use provided observation or current state
        current_state = observation if observation is not None else self.state
        
        for observer in self.observers:
            # Pass the current state to each observer in the pipeline
            current_state = observer.observe(current_state, **kwargs)

        self.state = current_state
        return self.state