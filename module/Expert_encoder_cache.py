import torch
from module.SSA_module import extract_adapter_params_for_experts

def _get_adapter_map_for_layer(model, layer_idx):
    adapter_map = extract_adapter_params_for_experts(model, layers_to_use=[layer_idx])
    out = {}
    for k, v in adapter_map.items():
        t = v.detach().cpu().float().clone()
        out[k] = t
    return out

def _compute_checksum(tensor):

    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    return float(torch.sum(tensor))

class ExpertEncodingCache:

    def __init__(self, expert_encoder):
        self.cache = {}
        self.expert_encoder = expert_encoder
        try:
            self.expert_encoder.to('cpu')
        except Exception:
            pass
        self.expert_encoder.eval()

    def ensure_layer_encoded(self, model, layer_idx, force_reencode_all=False):

        adapter_map = _get_adapter_map_for_layer(model, layer_idx)
        keys = sorted(adapter_map.keys())
        if layer_idx not in self.cache or force_reencode_all:
            mats = torch.stack([adapter_map[k] for k in keys], dim=0)
            with torch.no_grad():
                Y = self.expert_encoder(mats)
                if isinstance(Y, torch.Tensor):
                    Y_cpu = Y.detach().cpu().float().clone()
                else:
                    Y_cpu = torch.tensor(Y, dtype=torch.float32)
            checksums = {k: _compute_checksum(adapter_map[k]) for k in keys}
            self.cache[layer_idx] = {
                'keys': keys,
                'embeddings_cpu': Y_cpu,
                'checksums': checksums
            }
            return keys, Y_cpu

        entry = self.cache[layer_idx]
        cached_keys = entry['keys']
        cached_checksums = entry['checksums']
        cached_emb = entry['embeddings_cpu']

        if cached_keys != keys:
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

        changed_indices = []
        for idx, k in enumerate(keys):
            cs = _compute_checksum(adapter_map[k])
            if abs(cs - cached_checksums.get(k, float('nan'))) > 1e-6:
                changed_indices.append(idx)
                cached_checksums[k] = cs
        if len(changed_indices) == 0:
            return cached_keys, entry['embeddings_cpu']
        else:

            mats = torch.stack([adapter_map[k] for k in keys], dim=0)
            with torch.no_grad():
                Y = self.expert_encoder(mats)
                Y_cpu = Y.detach().cpu().float().clone()

            self.cache[layer_idx]['embeddings_cpu'] = Y_cpu
            self.cache[layer_idx]['checksums'] = cached_checksums
            return keys, Y_cpu