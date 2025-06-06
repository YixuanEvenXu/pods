"""
To evaluate the base model performance.

Developed by: Yixuan Even Xu in 2025
"""

import json
import hydra
from omegaconf import DictConfig
from utils.eval import test_model

@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig) -> None:
    result = test_model(cfg, "", "", None)
    with open(f"results/base.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()