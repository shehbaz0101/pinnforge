"""CPU training defaults for the Burgers FNO.

The architecture, optimizer, and window settings are the Stage 3 run.
``PREREGISTERED_HYBRID_WEIGHTS`` is the Stage 4 validation sweep. The
reported hybrid run is the member with the lowest validation mean relative
L2. Test scores do not choose the weight. This module does not import torch.
"""

from __future__ import annotations

DEFAULT_WIDTH = 32
DEFAULT_MODES = 16
DEFAULT_LAYERS = 4
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_SEED = 0
# Fixed before the Stage 4 test evaluation. Selection uses validation only.
PREREGISTERED_HYBRID_WEIGHTS = (1e-6, 1e-4, 1e-2)
