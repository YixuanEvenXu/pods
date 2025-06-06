"""
This file is adapted from trl.trainers.__init__.py (trl version 0.14.0)

Adapted by: Yixuan Even Xu in 2025
"""

from typing import TYPE_CHECKING

from trl.import_utils import OptionalDependencyNotAvailable, _LazyModule


_import_structure = {
    "maxvargrpo_trainer": ["MaxVarGRPOTrainer"],
    "maxvargrpo_config": ["MaxVarGRPOConfig"],
    "maxrewardgrpo_trainer": ["MaxRewardGRPOTrainer"],
    "maxrewardgrpo_config": ["MaxRewardGRPOConfig"],
    "randomgrpo_trainer": ["RandomGRPOTrainer"],
    "randomgrpo_config": ["RandomGRPOConfig"],
}

if TYPE_CHECKING:
    from .maxvargrpo_trainer import MaxVarGRPOTrainer
    from .maxvargrpo_config import MaxVarGRPOConfig
    from .maxrewardgrpo_trainer import MaxRewardGRPOTrainer
    from .maxrewardgrpo_config import MaxRewardGRPOConfig
    from .randomgrpo_trainer import RandomGRPOTrainer
    from .randomgrpo_config import RandomGRPOConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
