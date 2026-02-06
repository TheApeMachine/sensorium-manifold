"""Reconstruction projector - output reconstructed data.

For generation tasks: images, audio, text, raw bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensorium.projectors.base import BaseProjector


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction output.
    
    name: Output filename (without extension)
    output_type: Type of output ("image", "audio", "text", "bytes")
    field: Field name containing the reconstructed data
    sample_rate: For audio output (default: 22050)
    image_size: For image output (width, height) or None for auto
    colormap: For image output (None for RGB/grayscale)
    """
    name: str = "reconstruction"
    output_type: str = "bytes"
    field: str = "reconstruction"
    sample_rate: int = 22050
    image_size: Optional[tuple] = None
    colormap: Optional[str] = None


class ReconstructionProjector(BaseProjector):
    """Output reconstructed data from observation.
    
    Example:
        projector = ReconstructionProjector(ReconstructionConfig(
            name="generated_audio",
            output_type="audio",
            field="audio_samples",
            sample_rate=22050,
        ))
        projector.project(observation)
    """
    
    def __init__(self, config: ReconstructionConfig, output_dir: Path | None = None):
        super().__init__(output_dir or Path("artifacts"))
        self.config = config
    
    def project(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Output reconstructed data."""
        data = observation.get(self.config.field)
        
        if data is None:
            return {"status": "skipped", "reason": f"field '{self.config.field}' not found"}
        
        self.ensure_output_dir()
        
        if self.config.output_type == "image":
            return self._save_image(data)
        elif self.config.output_type == "audio":
            return self._save_audio(data)
        elif self.config.output_type == "text":
            return self._save_text(data)
        else:
            return self._save_bytes(data)
    
    def _save_image(self, data) -> Dict[str, Any]:
        """Save as image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        data = np.array(data)
        
        # Reshape if needed
        if self.config.image_size and data.ndim == 1:
            data = data.reshape(self.config.image_size)
        
        output_path = self.output_dir / f"{self.config.name}.png"
        
        if self.config.colormap:
            plt.imsave(output_path, data, cmap=self.config.colormap)
        else:
            plt.imsave(output_path, data)
        
        return {"status": "success", "path": str(output_path)}
    
    def _save_audio(self, data) -> Dict[str, Any]:
        """Save as WAV audio."""
        import numpy as np
        import wave
        
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        data = np.array(data)
        
        # Normalize to int16 range
        if data.dtype != np.int16:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = (data / max_val * 32767).astype(np.int16)
            else:
                data = data.astype(np.int16)
        
        output_path = self.output_dir / f"{self.config.name}.wav"
        
        with wave.open(str(output_path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(data.tobytes())
        
        return {"status": "success", "path": str(output_path)}
    
    def _save_text(self, data) -> Dict[str, Any]:
        """Save as text file."""
        output_path = self.output_dir / f"{self.config.name}.txt"
        
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        elif isinstance(data, list):
            # List of byte values or chars
            if data and isinstance(data[0], int):
                text = bytes(data).decode("utf-8", errors="replace")
            else:
                text = "".join(str(c) for c in data)
        else:
            text = str(data)
        
        output_path.write_text(text)
        
        return {"status": "success", "path": str(output_path)}
    
    def _save_bytes(self, data) -> Dict[str, Any]:
        """Save as raw bytes."""
        output_path = self.output_dir / f"{self.config.name}.bin"
        
        if isinstance(data, bytes):
            output_path.write_bytes(data)
        elif isinstance(data, list):
            output_path.write_bytes(bytes(data))
        else:
            import numpy as np
            if hasattr(data, 'cpu'):
                data = data.cpu().numpy()
            output_path.write_bytes(np.array(data).tobytes())
        
        return {"status": "success", "path": str(output_path)}
