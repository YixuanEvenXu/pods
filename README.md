# Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning

## Table of Contents

- [General Information](#general-information)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## General Information

This repository contains the source code for the experiments in paper ["Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning"](https://arxiv.org/abs/2504.13818) by Yixuan Even Xu*, Yash Savani*, Fei Fang, Zico Kolter. We implemented GRPO-PODS (Policy Optimization with Down-Sampling) and compared its performance with vanilla GRPO. Our implementation is based on [Unsloth](https://github.com/unslothai/unsloth) and [OpenR1](https://github.com/huggingface/open-r1).

## Reproducing the Experiments

- To install relevant dependencies, install `uv` and enter

  ``` bash
  uv sync
  ```

- To re-run the single-GPU experiments, edit `config/train.yaml`, and enter

  ``` bash
  mkdir -p checkpoints
  uv run python3 train.py
  ```

- To evaluate the saved checkpoints of a single-GPU experiment run, edit `config/test.yaml`, and enter

  ``` bash
  uv run python3 evaluate-run.py
  ```

- To run the multi-GPU experiments, `cd` into the `open-r1` directory, follow the install instructions in the `README.md` file within the directory, and then run the following script.

  ``` bash
  bash exp.sh
  ```

  The data can be collected and downloaded from the corresponding wandb runs and plotted using the plotting scripts.

- To generate the plots in the paper, enter

  ``` bash
  uv run python3 scripts/plot.py
  uv run python3 scripts/plot-h100s.py
  uv run python3 scripts/plot-a100s.py
  ```
  
## License

This repository's source code is available under the [Apache-2.0 License](LICENSE).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{xu2025not,
  title={Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning},
  author={Xu, Yixuan Even and Savani, Yash and Fang, Fei and Kolter, Zico},
  journal={arXiv preprint arXiv:2504.13818},
  year={2025}
}
```

## Contact

For any questions or issues, please contact us via email:
- Yixuan Even Xu: yixuanx@cs.cmu.edu
- Yash Savani: ysavani@cs.cmu.edu