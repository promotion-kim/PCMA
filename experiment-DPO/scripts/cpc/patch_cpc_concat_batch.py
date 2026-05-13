#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustly patch CPCMODPOTrainer._concat_batch so chosen/rejected tensors are padded
to the same length right before concatenation.

Run from experiment-DPO root:
  python scripts/cpc/patch_cpc_concat_batch.py
"""

from __future__ import annotations

from pathlib import Path
import re

path = Path("src/trainer/cpcmodpo_trainer.py")
text = path.read_text(encoding="utf-8")
original = text

backup = path.with_suffix(path.suffix + ".bak_concat_padfix")
if not backup.exists():
    backup.write_text(original, encoding="utf-8")
    print(f"[backup] {backup}")

new_method = '    @staticmethod\n    def _pad_pair_to_same_length(\n        left: torch.Tensor,\n        right: torch.Tensor,\n        pad_value: int,\n    ) -> Tuple[torch.Tensor, torch.Tensor]:\n        """Pad two [B, L] tensors to the same sequence length.\n\n        This is a defensive fix because chosen and rejected may be padded\n        separately by the collator. torch.cat along batch dimension requires\n        the sequence dimension to match.\n        """\n        if left.shape[1] == right.shape[1]:\n            return left, right\n\n        max_len = max(left.shape[1], right.shape[1])\n\n        def _pad(x: torch.Tensor) -> torch.Tensor:\n            if x.shape[1] == max_len:\n                return x\n            pad_width = max_len - x.shape[1]\n            pad = torch.full(\n                (x.shape[0], pad_width),\n                pad_value,\n                dtype=x.dtype,\n                device=x.device,\n            )\n            return torch.cat([x, pad], dim=1)\n\n        return _pad(left), _pad(right)\n\n    def _concat_batch(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:\n        chosen_input_ids, rejected_input_ids = self._pad_pair_to_same_length(\n            inputs["chosen_input_ids"],\n            inputs["rejected_input_ids"],\n            pad_value=self.tokenizer.pad_token_id\n            if getattr(self, "tokenizer", None) is not None and self.tokenizer.pad_token_id is not None\n            else 0,\n        )\n        chosen_attention_mask, rejected_attention_mask = self._pad_pair_to_same_length(\n            inputs["chosen_attention_mask"],\n            inputs["rejected_attention_mask"],\n            pad_value=0,\n        )\n        chosen_labels, rejected_labels = self._pad_pair_to_same_length(\n            inputs["chosen_labels"],\n            inputs["rejected_labels"],\n            pad_value=-100,\n        )\n\n        return {\n            "input_ids": torch.cat([chosen_input_ids, rejected_input_ids], dim=0),\n            "attention_mask": torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0),\n            "labels": torch.cat([chosen_labels, rejected_labels], dim=0),\n        }\n'

pattern = r'    def _concat_batch\(self, inputs: Dict\[str, Any\]\) -> Dict\[str, torch\.Tensor\]:.*?(?=\n    def _forward_logps)'
text_new, n = re.subn(pattern, new_method.rstrip(), text, count=1, flags=re.DOTALL)

if n != 1:
    print("[debug] Could not match typed _concat_batch; trying looser pattern...")
    pattern2 = r'    def _concat_batch\(self, inputs:.*?(?=\n    def _forward_logps)'
    text_new, n = re.subn(pattern2, new_method.rstrip(), text, count=1, flags=re.DOTALL)

if n != 1:
    raise RuntimeError("Could not find exactly one _concat_batch block. Please inspect src/trainer/cpcmodpo_trainer.py.")

path.write_text(text_new, encoding="utf-8")
print("[patch] Patched _concat_batch to pad chosen/rejected to same length before torch.cat.")

updated = path.read_text(encoding="utf-8")
if 'torch.cat([inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0)' in updated:
    print("[warning] Old brittle cat expression still exists somewhere.")
else:
    print("[verify] Old brittle cat expression removed.")
