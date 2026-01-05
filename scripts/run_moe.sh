#!/bin/bash
#SBATCH -p ai
#SBATCH --job-name=moe
#SBATCH --gres=gpu:1
module load CUDA/12.1
export  PYTHONUNBUFFERED=1

source activate moe

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="./datasets"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DOWNLOAD_TIMEOUT=86400
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="./data"
export HF_METRICS_CACHE="./datasets/metrics"
export TRANSFORMERS_OFFLINE=1

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 0 hcsmoe/merging-qwen.py \
  --model_name="./Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,openbookqa,rte,xquad_zh,xquad_es" \
  --task="xquad_es,xquad_zh,rte,openbookqa,boolq,arc_easy,arc_challenge,winogrande" \ #reverse
  --task="meddialog_raw_dialogues,medtext,mimic_repsum,mc_taco,med_concepts_qa_icd9cm_easy,medqa_4options,medmcqa,meqsum,olaph " \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --epsilon=0.01 \
  --PI_OUTLIER_RATIO=0.01 \
  --reg_lambda=0.1 \
  --ot_strength=5.0 \
  --lr=1e-5 \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/merge-45e/test" |& tee results/log_45e_test