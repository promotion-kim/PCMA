from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


def pad_to_length(tensor, length, pad_value, dim=-1):
    """
    Pad tensor on `dim` until it reaches `length`.
    Compatible with the helper used by older TRL DPOTrainer.
    """
    if tensor.size(dim) >= length:
        return tensor

    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.size(dim)

    padding = torch.full(
        pad_size,
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    return torch.cat([tensor, padding], dim=dim)


def disable_dropout_in_model(model):
    """
    Disable dropout modules in a model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def compute_accuracy(eval_pred):
    """
    Compute token accuracy while ignoring label positions set to -100.
    Mirrors the helper expected by older TRL trainers.
    """
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = torch.as_tensor(predictions)
    labels = torch.as_tensor(labels)

    preds = predictions.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return {"accuracy": 0.0}

    return {"accuracy": float((preds[mask] == labels[mask]).float().mean())}


def peft_module_casting_to_bf16(model):
    """
    Compatibility no-op for older TRL versions.
    """
    return model


@dataclass
class DPODataCollatorWithPadding:
    """
    Minimal DPO padding collator compatible with TRL-style tokenized examples.

    It pads:
      - *_input_ids with pad_token_id
      - *_attention_mask with 0
      - *_labels with label_pad_token_id
    and stacks scalar fields when possible.
    """
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: bool = False

    def _pad_sequence(self, tensors, pad_value):
        tensors = [
            t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long)
            for t in tensors
        ]
        max_len = max(t.size(-1) for t in tensors)
        padded = [pad_to_length(t, max_len, pad_value, dim=-1) for t in tensors]
        return torch.stack(padded, dim=0)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        if len(features) == 0:
            return batch

        keys = features[0].keys()

        for key in keys:
            values = [f[key] for f in features]

            if key.endswith("_input_ids"):
                batch[key] = self._pad_sequence(values, self.pad_token_id)

            elif key.endswith("_attention_mask"):
                batch[key] = self._pad_sequence(values, 0)

            elif key.endswith("_labels"):
                batch[key] = self._pad_sequence(values, self.label_pad_token_id)

            else:
                # Preserve strings/lists as-is, stack numeric scalars if possible.
                if isinstance(values[0], str):
                    batch[key] = values
                elif isinstance(values[0], torch.Tensor):
                    try:
                        batch[key] = torch.stack(values)
                    except Exception:
                        batch[key] = values
                elif isinstance(values[0], (int, float, bool)):
                    batch[key] = torch.tensor(values)
                else:
                    batch[key] = values

        return batch
