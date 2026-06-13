from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OscilloscopeData:
    metadata: dict[str, str]
    time: np.ndarray
    channels: dict[str, np.ndarray]
    source_files: tuple[Path, ...] = field(default_factory=tuple)
