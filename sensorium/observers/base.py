from typing import Protocol, Any

class ObserverProtocol(Protocol):
    def observe(self, observation: Any = None, **kwargs) -> dict:
        """Observe the state of the system and return a dictionary of observations.
        
        Args:
            observation: Optional observation data to process
            **kwargs: Additional keyword arguments
        """
        raise NotImplementedError("Subclasses must implement this method")