Our paper, **TiME: Test-Time Mixture-of-Experts Routing via Asymmetric CO-Optimal Transport for Continual Test-Time Adaptation**, has been accepted by **ICML 2026**!

## Setup
1. Install basic packages. `pip install -r requirements.txt`
2. Install `lm-eval`. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)


## Dataset Preparation
Please download the C4 training data c4-train.00000-of-01024.json from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz).

Then save it to the path `data/c4-train.00000-of-01024.json`.


## Experiments
We provide the code script in `scripts/run_moe.sh`. Change the settings in those files. Run the script file as follows.

```
bash ./scripts/run_moe.sh
```


## Citation
```
@misc{liu2026time,
title={Ti{ME}: Test-Time Mixture-of-Experts Routing via Asymmetric {CO}-Optimal Transport for Continual Test-Time Adaptation},
author={Tianlun Liu and Zhiliang Tian and Zhen Huang and Tianle Liu and Xingzhi Zhou and Feng Liu and Dongsheng Li},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```
