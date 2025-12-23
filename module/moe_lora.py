import types
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
###############################################################
#  LoRA Adapter
###############################################################

class ExpertLoRA(nn.Module):
    def __init__(self, hidden_size, r=16, alpha=16, dtype=torch.float32):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.down = nn.Linear(hidden_size, r, bias=False, dtype=dtype)
        self.up = nn.Linear(r, hidden_size, bias=False, dtype=dtype)

    def forward(self, x):
        return self.up(self.down(x)) * self.scaling


###############################################################
#  Inject LoRA
###############################################################
def get_expert_hidden_size(expert):
    for m in expert.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise RuntimeError("Expert does not contain Linear layers.")

def inject_lora_into_moe(model, r=16, alpha=16):

    for layer in model.model.layers:
        for expert in layer.mlp.experts:
            if hasattr(expert, "lora"):
                continue
            hidden_size = get_expert_hidden_size(expert)
            dtype = next(expert.parameters()).dtype
            device = next(expert.parameters()).device
            expert.lora = ExpertLoRA(hidden_size, r=r, alpha=alpha, dtype=dtype).to(device)
            expert._orig_forward = expert.forward
            def patched_forward(self, x):
                out = self._orig_forward(x)
                out = out + self.lora(x)
                return out
            expert.forward = types.MethodType(patched_forward, expert)
    print("LoRA Adapter inject into all MoE experts")
    return model


###############################################################
#  Router Hook (Recording which routing experts belong to this batch)
###############################################################

def register_layer_router_hook(model, routed_experts_dict, layers_to_track=None, topk=2):

    if layers_to_track is None:
        layers_to_track = list(range(len(model.model.layers)))
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in layers_to_track:
            continue
        def make_hook(idx):
            def router_hook(module, inputs, outputs):
                topk_values, topk_ids = torch.topk(outputs, k=topk, dim=-1)
                threshold = torch.quantile(topk_values[:, 0].float(), 0.95)
                gating_mask = topk_values[:, 0] >= threshold
                filtered_ids = topk_ids[gating_mask]
                routed_experts_dict[idx] = filtered_ids.detach().cpu()
            return router_hook
        layer.mlp.gate.register_forward_hook(make_hook(layer_idx))


def register_layer_ucoot_hook(model, layer_idx, pi_s, sel_input_ids, sel_attention_mask):

    def make_hook(pi_s_local, sel_input_ids_local, sel_attention_mask_local):
        def hook(module, inputs, outputs):
            B = sel_input_ids_local.size(0)
            seq_len = sel_attention_mask_local.size(1)
            E = outputs.size(-1)
            logits = outputs.view(B, seq_len, E)
            p_orig = torch.softmax(logits, dim=-1)
            p_comb = p_orig * pi_s_local.unsqueeze(1)
            p_new = p_comb / (p_comb.sum(dim=-1, keepdim=True) + 1e-12)
            logits_new = torch.log(p_new + 1e-12)
            logits_new = logits_new * sel_attention_mask_local.unsqueeze(-1)
            return logits_new.view(-1, E)
        return hook
    h = model.model.layers[layer_idx].mlp.gate.register_forward_hook(make_hook(pi_s, sel_input_ids, sel_attention_mask))
    return h

###############################################################
#  Freeze all non-lora parameters
###############################################################

def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("All parameters except LoRA have been frozen")

def print_mem(prefix=""):
    if not torch.cuda.is_available():
        print(f"[{prefix}] no cuda")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3

    print(f"[{prefix}] allocated={allocated:.2f}GB | "
          f"reserved={reserved:.2f}GB | "

          f"max_alloc={max_alloc:.2f}GB")
