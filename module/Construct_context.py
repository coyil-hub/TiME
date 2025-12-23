import torch
import torch.nn.functional as F

@torch.no_grad()
def construct_context(
        requests_list,
        tokenizer,
        model,
        task_family, 
        device="cuda",
):
    if task_family == "discriminative":
        final_sentences = []
        for requests in requests_list:
            sentences = []
            candidate_spans = []

            for req in requests:
                prefix, suffix = req.arguments
                sentences.append(prefix + suffix)

            enc = tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            spans = []
            for req in requests:
                prefix, suffix = req.arguments
                prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
                suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                start = len(prefix_ids)
                end = start + len(suffix_ids)
                spans.append((start, end))

            with torch.no_grad():
                logits = model(input_ids, attention_mask=attn).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            ppl_scores = []
            for i, (s, e) in enumerate(spans):
                loss = F.cross_entropy(
                    shift_logits[i, s:e].reshape(-1, shift_logits.size(-1)),
                    shift_labels[i, s:e].reshape(-1),
                    reduction="mean"
                )
                ppl_scores.append(loss.item())

            best_idx = int(torch.argmin(torch.tensor(ppl_scores)))
            final_sentences.append(sentences[best_idx])

        return final_sentences

    elif task_family == "generative":
        final_sentences = []
        for requests in requests_list:
            context, _ = requests.arguments
            final_sentences.append(context)

        return final_sentences
