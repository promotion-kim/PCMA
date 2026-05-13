#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRL/DeepSpeed-free CPC-MODPO trainer, aligned with MODPO logging.

This file intentionally does NOT import:
  - src.trainer.dpo_trainer
  - src.trainer.modpo_trainer
  - trl
  - deepspeed

It implements the MODPO loss directly on top of transformers.Trainer.

Fair-comparison principle:
  MODPO uses
      reward = (1 / w_anchor) [ beta log(pi/ref) - sum_margin w_i r_i ].
  CPC-MODPO uses the exact same algebra, but replaces
      w_i -> c_i = w_i * exp(gamma * a_i),
  where a_i is the learned objective-level log precision.

For gamma=0, c_i = w_i, so this reduces to the standard MODPO coefficient
structure, up to this lite trainer implementation.
"""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput


# ---------------------------------------------------------------------
# Compatibility patch:
# Some environments have a new transformers Trainer with an old accelerate.
# New Trainer may pass unsupported kwargs such as use_seedable_sampler.
# ---------------------------------------------------------------------
def _patch_accelerator_init_for_transformers_compat() -> None:
    try:
        import accelerate
        from accelerate import Accelerator
    except Exception:
        return

    if getattr(Accelerator.__init__, "_cpc_patched", False):
        return

    orig_init = Accelerator.__init__
    sig = inspect.signature(orig_init)
    allowed = set(sig.parameters.keys())
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def patched_init(self, *args, **kwargs):
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        else:
            # Still drop the known problematic key for older accelerate variants.
            kwargs.pop("use_seedable_sampler", None)
        return orig_init(self, *args, **kwargs)

    patched_init._cpc_patched = True
    Accelerator.__init__ = patched_init


_patch_accelerator_init_for_transformers_compat()


@dataclass
class CPCPreferenceDataCollator:
    """Pad chosen/rejected together and keep raw strings.

    MODPO's collator keeps raw_prompt and response strings for margin rewards.
    Here margin rewards are computed by adapter log-probabilities, but we keep
    raw strings for debugging and W&B generation consistency.
    """

    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined = []
        combined_labels = []

        # First B are chosen.
        for f in features:
            combined.append(
                {
                    "input_ids": f["chosen_input_ids"],
                    "attention_mask": f["chosen_attention_mask"],
                }
            )
            combined_labels.append(list(f["chosen_labels"]))

        # Next B are rejected.
        for f in features:
            combined.append(
                {
                    "input_ids": f["rejected_input_ids"],
                    "attention_mask": f["rejected_attention_mask"],
                }
            )
            combined_labels.append(list(f["rejected_labels"]))

        padded = self.tokenizer.pad(combined, padding=True, return_tensors="pt")
        max_len = padded["input_ids"].shape[1]

        labels = []
        for lab in combined_labels:
            labels.append(lab + [-100] * (max_len - len(lab)))
        labels = torch.tensor(labels, dtype=torch.long)

        B = len(features)
        return {
            "chosen_input_ids": padded["input_ids"][:B],
            "chosen_attention_mask": padded["attention_mask"][:B],
            "chosen_labels": labels[:B],
            "rejected_input_ids": padded["input_ids"][B:],
            "rejected_attention_mask": padded["attention_mask"][B:],
            "rejected_labels": labels[B:],
            "raw_prompt": [f.get("raw_prompt", f.get("prompt", "")) for f in features],
            "prompt": [f.get("prompt", "") for f in features],
            "chosen_text": [f.get("chosen", "") for f in features],
            "rejected_text": [f.get("rejected", "") for f in features],
        }


class CPCMODPOTrainer(Trainer):
    """Minimal CPC-MODPO trainer without TRL/DeepSpeed imports."""

    def __init__(
        self,
        *args,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        cpc_log_precisions: Sequence[float] = (0.0, 0.0),
        base_w: Sequence[float] = (0.5, 0.5),
        gamma: float = 1.0,
        anchor_objective_idx: int = 0,
        margin_adapter_names: Optional[Sequence[str]] = None,
        policy_adapter_name: str = "default",
        coefficient_floor: float = 1e-6,
        debug_print_every: int = 20,
        prompt_template: Optional[str] = None,
        generate_during_eval: bool = True,
        wandb_eval_generation_n: int = 5,
        wandb_eval_generation_max_new_tokens: int = 512,
        wandb_eval_generation_do_sample: bool = False,
        wandb_eval_generation_temperature: float = 0.7,
        wandb_eval_generation_top_p: float = 0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.beta = float(beta)
        self.loss_type = loss_type
        self.cpc_log_precisions = torch.tensor(list(cpc_log_precisions), dtype=torch.float32)
        self.base_w = torch.tensor(list(base_w), dtype=torch.float32)
        self.gamma = float(gamma)
        self.anchor_objective_idx = int(anchor_objective_idx)
        self.margin_adapter_names = list(margin_adapter_names or [])
        self.policy_adapter_name = str(policy_adapter_name)
        self.coefficient_floor = float(coefficient_floor)
        self.debug_print_every = int(debug_print_every)
        self.prompt_template = prompt_template

        self.generate_during_eval = bool(generate_during_eval)
        self.wandb_eval_generation_n = int(wandb_eval_generation_n)
        self.wandb_eval_generation_max_new_tokens = int(wandb_eval_generation_max_new_tokens)
        self.wandb_eval_generation_do_sample = bool(wandb_eval_generation_do_sample)
        self.wandb_eval_generation_temperature = float(wandb_eval_generation_temperature)
        self.wandb_eval_generation_top_p = float(wandb_eval_generation_top_p)
        self._last_wandb_generation_step = None

        if self.cpc_log_precisions.numel() != self.base_w.numel():
            raise ValueError(
                f"cpc_log_precisions length={self.cpc_log_precisions.numel()} "
                f"but base_w length={self.base_w.numel()}"
            )
        if self.anchor_objective_idx < 0 or self.anchor_objective_idx >= self.base_w.numel():
            raise ValueError(f"Invalid anchor_objective_idx={self.anchor_objective_idx}")
        if len(self.margin_adapter_names) != self.base_w.numel() - 1:
            raise ValueError(
                f"Expected {self.base_w.numel() - 1} margin adapters, "
                f"got {len(self.margin_adapter_names)}"
            )

        c = self.base_w * torch.exp(self.gamma * self.cpc_log_precisions)
        if c[self.anchor_objective_idx].item() <= 0.0:
            raise ValueError("Anchor CPC coefficient is zero. Use w_anchor > 0.")
        self.cpc_coefficients = c

    # Keep string columns from being moved to device as tensors.
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[override]
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.args.device)
            else:
                prepared[k] = v
        return prepared

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    def _set_adapter_if_available(self, model: nn.Module, adapter_name: str) -> None:
        unwrapped = self._unwrap(model)
        if hasattr(unwrapped, "set_adapter"):
            unwrapped.set_adapter(adapter_name)

    def _disable_adapter_context(self, model: nn.Module):
        unwrapped = self._unwrap(model)
        if hasattr(unwrapped, "disable_adapter"):
            return unwrapped.disable_adapter()

        class _Null:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False

        return _Null()

    @staticmethod
    def _sequence_logps_from_outputs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:].clone()
        mask = labels.ne(-100)
        labels = labels.masked_fill(~mask, 0)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (token_logps * mask).sum(dim=-1)

    @staticmethod
    def _pad_pair_to_same_length(
        left: torch.Tensor,
        right: torch.Tensor,
        pad_value: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if left.shape[1] == right.shape[1]:
            return left, right
        max_len = max(left.shape[1], right.shape[1])

        def _pad(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == max_len:
                return x
            pad_width = max_len - x.shape[1]
            pad = torch.full((x.shape[0], pad_width), pad_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=1)

        return _pad(left), _pad(right)

    def _concat_batch(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pad_id = 0
        if getattr(self, "tokenizer", None) is not None and self.tokenizer.pad_token_id is not None:
            pad_id = int(self.tokenizer.pad_token_id)

        chosen_input_ids, rejected_input_ids = self._pad_pair_to_same_length(
            inputs["chosen_input_ids"], inputs["rejected_input_ids"], pad_value=pad_id
        )
        chosen_attention_mask, rejected_attention_mask = self._pad_pair_to_same_length(
            inputs["chosen_attention_mask"], inputs["rejected_attention_mask"], pad_value=0
        )
        chosen_labels, rejected_labels = self._pad_pair_to_same_length(
            inputs["chosen_labels"], inputs["rejected_labels"], pad_value=-100
        )

        return {
            "input_ids": torch.cat([chosen_input_ids, rejected_input_ids], dim=0),
            "attention_mask": torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0),
            "labels": torch.cat([chosen_labels, rejected_labels], dim=0),
        }

    def _forward_logps(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self._sequence_logps_from_outputs(out.logits, batch["labels"])

    def _policy_ref_and_margin_logps(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Policy logps with trainable policy adapter.
        self._set_adapter_if_available(model, self.policy_adapter_name)
        policy_logps = self._forward_logps(model, batch)

        # Reference logps from adapter-disabled base model.
        with torch.no_grad():
            with self._disable_adapter_context(model):
                ref_logps = self._forward_logps(model, batch)

        # Margin objective implicit rewards from frozen margin adapters.
        margin_rewards: List[torch.Tensor] = []
        for adapter_name in self.margin_adapter_names:
            with torch.no_grad():
                self._set_adapter_if_available(model, adapter_name)
                margin_logps = self._forward_logps(model, batch)
                margin_rewards.append(self.beta * (margin_logps - ref_logps))

        # Restore policy adapter for backward.
        self._set_adapter_if_available(model, self.policy_adapter_name)
        return policy_logps, ref_logps, margin_rewards

    def _compute_cpc_loss_and_metrics(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        train_eval: Literal["train", "eval"],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = self._concat_batch(inputs)
        B = inputs["chosen_input_ids"].shape[0]

        policy_logps, ref_logps, margin_rewards_all = self._policy_ref_and_margin_logps(model, batch)

        policy_chosen, policy_rejected = policy_logps[:B], policy_logps[B:]
        ref_chosen, ref_rejected = ref_logps[:B], ref_logps[B:]

        c = self.cpc_coefficients.to(device=policy_logps.device, dtype=policy_logps.dtype)
        c_anchor = c[self.anchor_objective_idx].clamp_min(self.coefficient_floor)
        margin_indices = [i for i in range(c.numel()) if i != self.anchor_objective_idx]
        c_margin = c[margin_indices]

        if len(margin_rewards_all) != len(margin_indices):
            raise RuntimeError(
                f"Expected {len(margin_indices)} margin reward tensors, got {len(margin_rewards_all)}"
            )

        if len(margin_rewards_all) > 0:
            margin_mat = torch.stack(margin_rewards_all, dim=-1)  # [2B, M-1]
            chosen_margin, rejected_margin = margin_mat[:B], margin_mat[B:]
            chosen_margin_scalar = (chosen_margin * c_margin.view(1, -1)).sum(dim=-1)
            rejected_margin_scalar = (rejected_margin * c_margin.view(1, -1)).sum(dim=-1)
            margin_delta = chosen_margin - rejected_margin
        else:
            chosen_margin_scalar = torch.zeros_like(policy_chosen)
            rejected_margin_scalar = torch.zeros_like(policy_rejected)
            margin_delta = None

        chosen_rewards = (self.beta * (policy_chosen - ref_chosen) - chosen_margin_scalar) / c_anchor
        rejected_rewards = (self.beta * (policy_rejected - ref_rejected) - rejected_margin_scalar) / c_anchor
        logits = chosen_rewards - rejected_rewards

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1.0 - logits)
        else:
            raise ValueError(f"Unknown loss_type={self.loss_type}")

        loss = losses.mean()
        acc = (chosen_rewards > rejected_rewards).float()

        # Match MODPO metric names where possible.
        prefix = "eval_" if train_eval == "eval" else ""
        metrics: Dict[str, float] = {
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).detach().float().mean().item(),
            f"{prefix}rewards/chosen": chosen_rewards.detach().float().mean().item(),
            f"{prefix}rewards/rejected": rejected_rewards.detach().float().mean().item(),
            f"{prefix}logps/margins": (policy_chosen - policy_rejected).detach().float().mean().item(),
            f"{prefix}logps/chosen": policy_chosen.detach().float().mean().item(),
            f"{prefix}logps/rejected": policy_rejected.detach().float().mean().item(),
            f"{prefix}cpc/logit_mean": logits.detach().float().mean().item(),
            f"{prefix}cpc/logit_abs_mean": logits.detach().float().abs().mean().item(),
            f"{prefix}cpc/c_anchor": c_anchor.detach().float().item(),
        }
        if train_eval == "train":
            metrics["accuracy"] = acc.detach().float().mean().item()
        else:
            metrics["eval_accuracy"] = acc.detach().float().mean().item()

        for i, ci in enumerate(c):
            metrics[f"{prefix}cpc/c_{i}"] = ci.detach().float().item()
        for i, ai in enumerate(self.cpc_log_precisions):
            metrics[f"{prefix}cpc/a_{i}"] = float(ai.detach().cpu().item())
        metrics[f"{prefix}cpc/gamma"] = float(self.gamma)

        if margin_delta is not None:
            for j in range(margin_delta.shape[-1]):
                metrics[f"{prefix}cpc/margin_delta_{j}_mean"] = margin_delta[:, j].detach().float().mean().item()
                metrics[f"{prefix}cpc/margin_delta_{j}_abs_mean"] = margin_delta[:, j].detach().float().abs().mean().item()

        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        train_eval: Literal["train", "eval"] = "train" if model.training else "eval"
        loss, metrics = self._compute_cpc_loss_and_metrics(model, inputs, train_eval=train_eval)

        # Log train metrics at a MODPO-like cadence. Eval metrics are logged during eval calls.
        step = int(getattr(self.state, "global_step", 0))
        interval = max(int(self.debug_print_every), 1)
        if step % interval == 0:
            self.log(metrics)

        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss

    # ------------------------------------------------------------------
    # Optional W&B generation logging, similar in spirit to PCMODPOTrainer.
    # ------------------------------------------------------------------
    def _format_prompt_for_generation(self, raw_prompt: str) -> str:
        text = str(raw_prompt)
        if "BEGINNING OF CONVERSATION:" in text and "ASSISTANT:" in text:
            return text
        if self.prompt_template is not None and "{raw_prompt}" in self.prompt_template:
            return self.prompt_template.format(raw_prompt=text)
        return text

    def _extract_raw_prompt_from_eval_example(self, example: dict) -> str:
        if "raw_prompt" in example and example["raw_prompt"] is not None:
            return str(example["raw_prompt"])
        if "prompt" in example and example["prompt"] is not None:
            return str(example["prompt"])
        return ""

    @torch.no_grad()
    def _log_eval_generations_to_wandb(self, dataloader) -> None:
        if not self.generate_during_eval:
            return
        if self.wandb_eval_generation_n <= 0:
            return
        if not self.is_world_process_zero():
            return

        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return

        step = int(getattr(self.state, "global_step", 0))
        if self._last_wandb_generation_step == step:
            return
        self._last_wandb_generation_step = step

        dataset = dataloader.dataset
        if len(dataset) == 0:
            return

        n = min(self.wandb_eval_generation_n, len(dataset))
        rng = random.Random(1234 + step)
        indices = rng.sample(range(len(dataset)), k=n)

        raw_prompts = [self._extract_raw_prompt_from_eval_example(dataset[int(idx)]) for idx in indices]
        model_inputs_text = [self._format_prompt_for_generation(p) for p in raw_prompts]

        model = self._unwrap(self.model)
        was_training = model.training
        model.eval()
        self._set_adapter_if_available(model, self.policy_adapter_name)

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            inputs = self.tokenizer(
                model_inputs_text,
                padding=True,
                truncation=True,
                max_length=min(getattr(self, "max_length", 1024), 1024),
                return_tensors="pt",
            ).to(self.args.device)

            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": self.wandb_eval_generation_max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": self.wandb_eval_generation_do_sample,
            }
            if self.wandb_eval_generation_do_sample:
                gen_kwargs["temperature"] = self.wandb_eval_generation_temperature
                gen_kwargs["top_p"] = self.wandb_eval_generation_top_p

            generated = model.generate(**gen_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            responses = self.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            table = wandb.Table(columns=["global_step", "eval_index", "raw_prompt", "model_input", "generation"])
            for idx, raw_prompt, model_input, response in zip(indices, raw_prompts, model_inputs_text, responses):
                table.add_data(step, int(idx), raw_prompt, model_input, response)
            wandb.log({"eval/generation_samples": table, "eval/generation_step": step}, step=step)
            print(f"[CPC-MODPO] logged {n} eval generations to wandb at step={step}", flush=True)
        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Custom evaluation step for RCS/CPC preference batches.

        The default Transformers Trainer.prediction_step calls model(**inputs).
        However, our batch contains keys such as chosen_input_ids and
        rejected_input_ids, which are not valid inputs to LlamaForCausalLM.
        Therefore, evaluation must use the same custom preference loss path
        as training.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, metrics = self._compute_cpc_loss_and_metrics(
                model,
                inputs,
                train_eval="eval",
            )

        # Optional: log eval diagnostics. Trainer will handle eval_loss
        # from the returned loss.
        step = int(getattr(self.state, "global_step", 0))
        interval = max(int(self.debug_print_every), 1)
        if step % interval == 0:
            self.log(metrics)

        # Return format required by Trainer:
        # (loss, logits, labels)
        # We do not need logits/labels for preference validation loss.
        return loss.detach(), None, None
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self._log_eval_generations_to_wandb(dataloader)
        return super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
