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
@InProceedings{pmlr-v306-liu26ec,
  title = 	 {{T}i{ME}: Test-Time Mixture-of-Experts Routing via Asymmetric {CO}-Optimal Transport for Continual Test-Time Adaptation},
  author =       {Liu, Tianlun and Tian, Zhiliang and Huang, Zhen and Liu, Tianle and Zhou, Xingzhi and Liu, Feng and Li, Dongsheng},
  booktitle = 	 {Proceedings of the 43rd International Conference on Machine Learning},
  pages = 	 {78597--78632},
  year = 	 {2026},
  editor = 	 {Zhang, Tong and Dudik, Miroslav and Jaggi, Martin and Agarwal, Alekh and Li, Sharon and Schuurmans, Dale and Zhu, Jerry and Berkenkamp, Felix and Dong, Hanze and Bietti, Alberto},
  volume = 	 {306},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--11 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v306/main/assets/liu26ec/liu26ec.pdf},
  url = 	 {https://proceedings.mlr.press/v306/liu26ec.html},
  abstract = 	 {Large language models usually face continuous domain shifts during testing, which degrade performance on unseen shifting domains. So, researchers propose continual test-time adaptation (CTTA) to adapt to evolving testing domains while preserving knowledge of previous domains, making adaptability-stability (A-S) balance. Existing CTTA methods are constrained by dense base models that encode knowledge from all domains into a global model, hardly achieving the A-S balance. We observe that the model sparsity of mixture-of-experts (MoE) models is better for achieving A–S balance than dense models. In CTTA, however, MoE faces difficulty in (1) correctly routing samples from unseen shifting domains and (2) capturing domain-level shifts. In this paper, we propose test-time mixture-of-experts routing (TiME) via asymmetric co-optimal transport (As-COOT): we model MoE routing in CTTA as a test-time allocation problem via COOT. To ensure reliable routing, we propose a semantic space alignment to align sample-expert distributions via bidirectional contrastive learning. To address COOT’s limitations in CTTA, we propose As-COOT, relaxing sample-side constraints while enforcing expert-side constraints to ensure noise robustness and balance expert load. Experiments show TiME outperforms baselines.}
}

```
