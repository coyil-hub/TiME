from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import math
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# -----------------------------
# Helper: extract LoRA-adapter parameters for each expert
# -----------------------------

def extract_adapter_params_for_experts(model, layers_to_use: List[int]) -> Dict[Tuple[int,int], torch.Tensor]:
    """
    返回每个(层_idx, expert_idx)对应的 adapter 参数向量（扁平化）
    约定：adapter 的命名为 expert.lora，且内部包含 down/up 两个线性层（inject_lora_into_moe 中的 ExpertLoRA 格式）

    返回格式: {(layer_idx, expert_idx): tensor(flattened_params)}
    """
    out = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in layers_to_use:
            continue
        for expert_idx, expert in enumerate(layer.mlp.experts):
            if not hasattr(expert, 'lora'):
                raise RuntimeError(f"expert at layer {layer_idx} idx {expert_idx} has no lora attribute")
            lora = expert.lora
            # 将 adapter 参数扁平化并拼接：down.weight, up.weight
            parts = []
            for name, p in lora.named_parameters():
                parts.append(p.detach().cpu().reshape(-1))
            flat = torch.cat(parts, dim=0).to(torch.float32)
            out[(layer_idx, expert_idx)] = flat
    return out

# -----------------------------
# Expert Encoder: 将 adapter 参数向量投影为 embedding
# 设计依据：不能直接使用原始高维参数作为 embedding，大多数工作采用降维+线性映射或平均池化等方式。
# 我们使用一个小型的可学习线性投影（nn.Linear）对扁平参数做降维。这样做的优点是：
#   - 与直接PCA相比可端到端训练
#   - 参数量小、易于保存与部署
# -----------------------------

class ExpertEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embedding_dim: int = 256,
                 hidden: int = 512,
                 device: torch.device | str | None = None):
        super().__init__()
        self.device = device if device is not None else "cpu"

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embedding_dim),
        )
        self.to(self.device)

    def forward(self, param_vec):
        param_vec = param_vec.to(self.device)
        x = self.norm(param_vec)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class TrainableSampleEncoder(torch.nn.Module):
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2",
                 embedding_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.encoder = AutoModel.from_pretrained(encoder_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        hidden_size = self.encoder.config.hidden_size
        self.proj = torch.nn.Linear(hidden_size, embedding_dim).to(device)

    def forward(self, texts):
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.encoder(**tok)
        cls = out.last_hidden_state[:, 0]          # (B, hidden) 已经在 device
        emb = self.proj(cls)                       # (B, D) 同样在 device
        return emb


# -----------------------------
# Sample Encoder: 文本编码器（使用 sentence-transformers 的轻量模型或 HuggingFace encoder + pooling）
# 建议使用小型句向量模型：'sentence-transformers/all-MiniLM-L6-v2'（用户需本地下载）
# -----------------------------
class SampleEncoder(nn.Module):
    def __init__(self, encoder_name: str = './all-MiniLM-L6-v2', embedding_dim: int = 256, device="cpu"):
        super().__init__()
        self.device = device
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers 未安装，请安装 sentence-transformers 或替换为 HuggingFace encoder')
        self.st = SentenceTransformer(encoder_name)
        # 如果 sentence-transformers 输出维度 != embedding_dim，添加线性投影
        out_dim = self.st.get_sentence_embedding_dimension()
        if out_dim != embedding_dim:
            self.proj = nn.Linear(out_dim, embedding_dim)
        else:
            self.proj = None

    def forward(self, texts: List[str]) -> torch.Tensor:
        # 返回归一化的向量 (batch, D)
        with torch.no_grad():
            emb = self.st.encode(texts, convert_to_numpy=False, show_progress_bar=False)
            # emb is a numpy array or tensor; ensure it's torch.Tensor
        if isinstance(emb, list):
            # list of numpy arrays → stack
            emb = torch.stack(emb, dim=0)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        emb = emb.to(self.device)
        if self.proj is not None:
            emb = self.proj(emb)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


# -----------------------------
# Dataset for Sersa: 使用已经构建好的正负样本集合文件 (posneg.pt)
# posneg structure: list of dict { 'text': str, 'positives': List[(layer,expert)], 'negatives': List[(layer,expert)] }
# -----------------------------
class SersaDataset(Dataset):
    def __init__(self, posneg_path: str):
        assert os.path.exists(posneg_path), f"{posneg_path} not found"
        with open(posneg_path, 'rb') as f:
            data = pickle.load(f)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # item: { 'text': str, 'positives': [(layer,expert), ...], 'negatives': [...] }
        return item


# -----------------------------
# 构建正负样本集合（离线）
# 说明：
#  - 利用 register_layer_router_hook 写入的 routed_experts 字典。该字典在 run 时由 hook 写入：{ layer_idx: tensor(batch_size, topk) }
#  - 我们需要把 dataloader_for_Sersa 中的所有样本遍历一次，记录每个样本在指定层上被路由到的expert id。
#  - 输出为 pickle 文件，内容 list of dict: { 'text': raw_text, 'positives': [(layer,expert_id),...], 'negatives': [...] }
# 注意：dataloader_for_Sersa 的返回格式可能因你实现而异，本函数对常见格式做了容错处理（期望包含 'input_ids' 或 'text'）。
# -----------------------------

def build_pos_neg_sets(model, dataloader_for_Sersa, routed_experts_dict: Dict[int, torch.Tensor],
                       layers_to_use: List[int] = [0], topk: int = 2, save_path: str = './Sersa_outputs/posneg.pt', device: Optional[str] = None,
                       max_samples: Optional[int] = None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_entries = []
    sample_idx = 0
    skipped_no_pos = 0

    pbar = tqdm(dataloader_for_Sersa, desc='build_posneg')
    with torch.no_grad():
        for batch in pbar:
            routed_experts_dict.clear()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            model.current_attention_mask = attention_mask
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            texts = batch.get('text', None)
            if texts is None:
                tokenizer = getattr(dataloader_for_Sersa, 'tokenizer', None)
                if tokenizer is not None:
                    texts = [tokenizer.decode(x, skip_special_tokens=True) for x in input_ids]
                else:
                    texts = [f'<sample_{sample_idx+i}>' for i in range(input_ids.size(0))]
            batch_size = len(texts)
            for i in range(batch_size):
                selected = set()
                for layer_idx in layers_to_use:
                    arr = routed_experts_dict.get(layer_idx, None)
                    if arr is None:
                        continue
                    total_tokens = arr.size(0)
                    num_experts = arr.size(1)
                    if batch_size == 1:
                        token_start = 0
                        token_end = total_tokens
                    else:
                        seq_len = total_tokens // batch_size
                        token_start = i * seq_len
                        token_end = (i + 1) * seq_len
                    token_ids = arr[token_start:token_end]
                    all_token_experts = token_ids.reshape(-1).tolist()
                    for eid in set(all_token_experts):
                        selected.add((layer_idx, int(eid)))

                if len(selected) == 0:
                    skipped_no_pos += 1
                    print(f"[SKIP] sample {sample_idx} has NO positive experts, skipped.")
                    sample_idx += 1
                    continue

                negatives = []
                for layer_idx in layers_to_use:
                    n_experts = len(model.model.layers[layer_idx].mlp.experts)
                    for eid in range(n_experts):
                        if (layer_idx, eid) not in selected:
                            negatives.append((layer_idx, eid))
                item = {
                    'text': texts[i],
                    'positives': list(selected),
                    'negatives': negatives,
                }
                all_entries.append(item)
                sample_idx += 1
                if (max_samples is not None) and (sample_idx >= max_samples):
                    break
            if (max_samples is not None) and (sample_idx >= max_samples):
                break
    with open(save_path, 'wb') as f:
        pickle.dump(all_entries, f)

    print(f"\nSaved pos/neg sets to {save_path}, total {len(all_entries)} samples")
    print(f"Skipped {skipped_no_pos} samples with NO positive experts.")

    return save_path

# -----------------------------
# 损失函数：双向 InfoNCE (样本->专家，专家->样本) 以及 簇内 KL
# -----------------------------

def info_nce_loss_matrix(sim_matrix, positives_mask, temperature=0.07):
    logits = sim_matrix / temperature

    logsumexp_all = torch.logsumexp(logits, dim=1)
    masked = logits.masked_fill(positives_mask == 0, float('-inf'))
    logsumexp_pos = torch.logsumexp(masked, dim=1)
    loss = -(logsumexp_pos - logsumexp_all)
    return loss.mean()


def compute_cluster_distributions(sample_embeddings: torch.Tensor,
                                  expert_embeddings: torch.Tensor,
                                  G: int = 100,
                                  tau: float = 0.1):
    """
    可微分版本的 '对一批样本做聚类并计算簇的专家偏好分布'
    返回:
        P_x: (B, E)   每个样本的专家偏好分布
        P_cx: (B, E)   每个样本对应的“簇专家分布”（可微）
    """
    sample_embeddings = sample_embeddings.detach()
    expert_embeddings = expert_embeddings.detach()
    sim_xe = sample_embeddings @ expert_embeddings.t()   # (B, E)
    P_x = F.softmax(sim_xe, dim=1)                       # (B, E)
    B, D = sample_embeddings.size()
    k = min(G, max(1, B // 5))
    indices = torch.randperm(B)[:1]
    centers = sample_embeddings[indices]   # (1, D)
    if k > 1:
        rand_indices = torch.randperm(B)[:(k - 1)]
        others = sample_embeddings[rand_indices]  # (k-1, D)
        centers = torch.cat([centers, others], dim=0)   # (k, D)
    dists = torch.cdist(sample_embeddings, centers, p=2)   # (B, k)
    r = F.softmax(-dists / tau, dim=1)  # (B, k), soft cluster assignment
    P_ce = r.t() @ P_x                                # (k, E)
    cluster_mass = r.sum(dim=0, keepdim=True).t()     # (k, 1)
    P_ce = P_ce / (cluster_mass + 1e-12)              # (k, E)
    P_cx = r @ P_ce

    return P_x, P_cx


# -----------------------------
# Sersa Trainer
# -----------------------------
class SersaTrainer:
    def __init__(self,
                 posneg_path: str,
                 model,  # the MoE model (for extracting adapter params)
                 dataloader_for_Sersa,  # used for texts if needed
                 layers_to_use: List[int] = [0],
                 embedding_dim: int = 256,
                 batch_size: int = 64,
                 lr: float = 1e-4,
                 device: Optional[str] = None,
                 encoder_name: str = './all-MiniLM-L6-v2',
                 save_dir: str = './Sersa_outputs/sersa_ckpt'):
        self.device = device or (next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu'))
        self.model = model
        self.posneg_path = posneg_path
        self.dataset = SersaDataset(posneg_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate)
        self.layers_to_use = layers_to_use
        # extract one adapter param to know input_dim
        adapter_map = extract_adapter_params_for_experts(model, layers_to_use)
        # take first
        _, first_vec = next(iter(adapter_map.items()))
        input_dim = first_vec.shape[0]
        self.expert_encoder = ExpertEncoder(input_dim=input_dim, embedding_dim=embedding_dim, device=self.device)
        self.sample_encoder = TrainableSampleEncoder(encoder_name=encoder_name, embedding_dim=embedding_dim, device=self.device)
        self.encoder_on_gpu = True
        self.optimizer = torch.optim.Adam(
            list(self.expert_encoder.parameters()) +
            list(self.sample_encoder.parameters()),
            lr=lr
        )
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def collate(self, batch_items):
        # batch_items: list of dicts
        texts = [it['text'] for it in batch_items]
        positives = [it['positives'] for it in batch_items]
        negatives = [it['negatives'] for it in batch_items]
        return {'texts': texts, 'positives': positives, 'negatives': negatives}

    def build_expert_embeddings(self):
        adapter_map = extract_adapter_params_for_experts(self.model, self.layers_to_use)
        keys = sorted(adapter_map.keys())
        mats = [adapter_map[k] for k in keys]
        mats = torch.stack(mats, dim=0)
        mats_gpu = mats.to(self.device)
        with torch.no_grad():
            emb = self.expert_encoder(mats_gpu)
        return keys, emb

    def train_epoch(self, temperature: float = 0.07, cluster_G: int = 50, lambda_cluster: float = 1.0):
        self.expert_encoder.train()
        total_loss = 0.0
        keys, expert_embeddings = self.build_expert_embeddings()
        expert_embeddings = expert_embeddings.detach()
        # build index map from (layer,expert) to col idx
        key2idx = {k: i for i, k in enumerate(keys)}
        for batch in tqdm(self.dataloader, desc='Sersa train'):
            texts = batch['texts']
            positives = batch['positives']
            negatives = batch['negatives']
            B = len(texts)
            sample_emb = self.sample_encoder(texts)  # (B, D) tensor on cpu
            if isinstance(sample_emb, torch.Tensor):
                sample_emb = sample_emb.to(self.device)
            E = expert_embeddings.size(0)
            pos_mask = torch.zeros((B, E), dtype=torch.long, device=self.device)
            for i, pos_list in enumerate(positives):
                for k in pos_list:
                    if k in key2idx:
                        pos_mask[i, key2idx[k]] = 1

            sim_se = sample_emb @ expert_embeddings.t()  # (B, E)
            loss_se = info_nce_loss_matrix(sim_se, pos_mask.float(), temperature=temperature)
            exp_pos_mask = torch.zeros((E, B), dtype=torch.long, device=self.device)
            for i, pos_list in enumerate(positives):
                for k in pos_list:
                    if k in key2idx:
                        exp_pos_mask[key2idx[k], i] = 1
            sim_es = expert_embeddings @ sample_emb.t()  # (E, B)
            loss_es = info_nce_loss_matrix(sim_es, exp_pos_mask.float(), temperature=temperature)
            # # cluster loss
            # if B < 20:
            #     loss_cluster = 0.0
            # else:
            #     P_x, P_c = compute_cluster_distributions(sample_emb, expert_embeddings, G=cluster_G)
            # # compute KL per sample
            # P_x_log = torch.log(P_x + 1e-12)
            # P_c = (P_c + 1e-12)
            # # P_c = P_c / P_c.sum(dim=1, keepdim=True)
            # loss_cluster = F.kl_div(P_x_log, P_c, reduction='batchmean')
            # loss = loss_se + loss_es + lambda_cluster * loss_cluster
            if B < 50:
                break
            loss = (loss_se + loss_es) / 2
            print(f"loss:{loss}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def train(self, epochs: int = 5, temperature: float = 0.07, cluster_G: int = 50, lambda_cluster: float = 1.0):
        for ep in range(epochs):
            avg_loss = self.train_epoch(temperature=temperature, cluster_G=cluster_G, lambda_cluster=lambda_cluster)
            print(f"Epoch {ep+1}/{epochs}, loss={avg_loss:.6f}")
            # 保存中间模型
            self.save_checkpoint(f"epoch_{ep+1}.pt")
        # 最终保存
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, name: str = 'final.pt'):
        path = os.path.join(self.save_dir, name)
        payload = {
            'expert_encoder_state': self.expert_encoder.state_dict(),
            'sample_encoder_state': self.sample_encoder.state_dict()
        }
        torch.save(payload, path)
        print(f"Saved Sersa checkpoint to {path}")