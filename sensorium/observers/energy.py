import numpy as np

from sensorium.observers.base import ObserverProtocol


class EnergyObserver(ObserverProtocol):

    def __init__(
        self,
        *,
        prime: int,
        vocab: int,
        MNIST_IMAGE_SIZE: int,
    ):
        self.prime = prime
        self.vocab = vocab
        self.MNIST_IMAGE_SIZE = MNIST_IMAGE_SIZE
        self.prompt_flat = None
        self.prompt_len = 0
        self.energy_by_tid = None

    def observe(self, observation=None, **kwargs):
        """Decode missing pixels by argmax over bytes using learned token energies.

        NOTE: This observer returns an image array (not a dict). It's a specialized
        helper for experiments; callers should not assume pipeline dict semantics.
        """
        if self.prompt_flat is None:
            raise ValueError("prompt_flat must be set before calling observe()")
        if self.energy_by_tid is None:
            raise ValueError("energy_by_tid must be set before calling observe()")
        
        mask = int(self.vocab) - 1
        # Always create recon from prompt_flat (use observation if provided, otherwise use prompt_flat)
        if observation is not None:
            recon = np.array(observation, dtype=np.uint8) if not isinstance(observation, np.ndarray) else observation.astype(np.uint8)
        else:
            recon = self.prompt_flat.copy().astype(np.uint8)
        
        # Ensure recon is the right size
        if len(recon) < self.MNIST_IMAGE_SIZE:
            # Pad with zeros if needed
            recon = np.pad(recon, (0, self.MNIST_IMAGE_SIZE - len(recon)), mode='constant', constant_values=0)
        elif len(recon) > self.MNIST_IMAGE_SIZE:
            # Truncate if needed
            recon = recon[:self.MNIST_IMAGE_SIZE]
        
        # For each missing position, choose byte b maximizing energy[tid(b, pos)].
        for pos in range(self.prompt_len, self.MNIST_IMAGE_SIZE):
            best_b = 0
            best_e = -1.0
            for b in range(256):
                tid = ((b * int(self.prime)) + int(pos)) & mask
                e = float(self.energy_by_tid[tid])
                if e > best_e:
                    best_e = e
                    best_b = b
            recon[pos] = np.uint8(best_b)

        return recon.reshape(28, 28)
