import torch
from hcsmoe.module.Sersa_module import extract_adapter_params_for_experts

def _get_adapter_map_for_layer(model, layer_idx):
    """
    返回该层 adapter_map: {expert_key: tensor_cpu_vector}
    这里沿用你 extract_adapter_params_for_experts 的接口（如果接口返回 dict of tensors）
    确保返回的 tensor 在 CPU 上且为 float32
    """
    adapter_map = extract_adapter_params_for_experts(model, layers_to_use=[layer_idx])
    # adapter_map 里的 tensor 可能在 GPU 或 CPU；统一搬到 cpu，detach
    out = {}
    for k, v in adapter_map.items():
        t = v.detach().cpu().float().clone()
        out[k] = t
    return out

def _compute_checksum(tensor):
    """轻量 checksum：返回 tensor 的 float(sum) 的 Python float 形式（在 cpu 上运行）"""
    # tensor 可能在 cpu，也可能在 gpu — 确保在 cpu
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    # 使用 float sum，代价小
    return float(torch.sum(tensor))

class ExpertEncodingCache:
    """
    管理 per-layer, per-expert encoding 缓存（存在 CPU 上）。
    结构:
      cache[layer_idx] = {
          'keys': [key1,key2,...],   # sorted keys
          'embeddings_cpu': tensor (E, D),  # float32 on cpu
          'checksums': {key: checksum_float},  # last seen checksums
      }
    提供方法：ensure_layer_encoded(model, layer_idx, expert_encoder)
      - 会提取当前 adapter 参数并只对 changed experts 使用 expert_encoder 编码（在 CPU 上运行）
      - 返回 (keys, embeddings_cpu)； embeddings_cpu 在 CPU
    """
    def __init__(self, expert_encoder):
        self.cache = {}
        self.expert_encoder = expert_encoder
        # 强制 encoder 在 CPU
        try:
            self.expert_encoder.to('cpu')
        except Exception:
            pass
        self.expert_encoder.eval()

    def ensure_layer_encoded(self, model, layer_idx, force_reencode_all=False):
        """
        确保 layer_idx 在 cache 中，并更新有改动的 experts。
        返回 (keys_sorted, embeddings_cpu_tensor)
        """
        # 1) 取当前 adapter_map（所有 experts 的原始参数向量） -> 在 CPU
        adapter_map = _get_adapter_map_for_layer(model, layer_idx)  # {key: tensor_cpu}
        keys = sorted(adapter_map.keys())
        # prepare cache entry
        if layer_idx not in self.cache or force_reencode_all:
            # 全量编码
            mats = torch.stack([adapter_map[k] for k in keys], dim=0)  # (E, input_dim) on CPU
            with torch.no_grad():
                # expert_encoder 在 CPU，直接计算得到 embeddings (E, D) 在 CPU（确保）
                Y = self.expert_encoder(mats)  # expect tensor on CPU
                if isinstance(Y, torch.Tensor):
                    Y_cpu = Y.detach().cpu().float().clone()
                else:
                    # 保险处理：如果 encoder 返回 numpy 等
                    Y_cpu = torch.tensor(Y, dtype=torch.float32)
            checksums = {k: _compute_checksum(adapter_map[k]) for k in keys}
            self.cache[layer_idx] = {
                'keys': keys,
                'embeddings_cpu': Y_cpu,
                'checksums': checksums
            }
            return keys, Y_cpu

        # 已有缓存 -> 只更新发生变化的 experts
        entry = self.cache[layer_idx]
        cached_keys = entry['keys']
        cached_checksums = entry['checksums']
        cached_emb = entry['embeddings_cpu']  # shape (E_cached, D)

        # detect if expert set changed (新增或删除)
        if cached_keys != keys:
            # 结构发生变化，做一次全量重编码
            mats = torch.stack([adapter_map[k] for k in keys], dim=0)
            with torch.no_grad():
                Y = self.expert_encoder(mats)
                Y_cpu = Y.detach().cpu().float().clone()
            checksums = {k: _compute_checksum(adapter_map[k]) for k in keys}
            self.cache[layer_idx] = {
                'keys': keys,
                'embeddings_cpu': Y_cpu,
                'checksums': checksums
            }
            return keys, Y_cpu

        # keys 一样 -> 检查哪些 checksum 变了
        changed_indices = []
        for idx, k in enumerate(keys):
            cs = _compute_checksum(adapter_map[k])
            if abs(cs - cached_checksums.get(k, float('nan'))) > 1e-6:
                changed_indices.append(idx)
                cached_checksums[k] = cs  # 更新 checksum
        if len(changed_indices) == 0:
            return cached_keys, entry['embeddings_cpu']  # 无变更，直接返回缓存（cpu）
        else:
            # 只重编码变更的 rows -> 合并回缓存
            # 取出对应 mats 行并运行 encoder（在 CPU）
            mats = torch.stack([adapter_map[k] for k in keys], dim=0)
            with torch.no_grad():
                Y = self.expert_encoder(mats)
                Y_cpu = Y.detach().cpu().float().clone()
            # 用全量结果替换（简单可靠）
            self.cache[layer_idx]['embeddings_cpu'] = Y_cpu
            self.cache[layer_idx]['checksums'] = cached_checksums
            return keys, Y_cpu