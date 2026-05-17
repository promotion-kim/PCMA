from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput


DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"


def _format_prompt(prompt_template: str, raw_prompt: str) -> str:
    return prompt_template.format(raw_prompt=raw_prompt)


@dataclass
class BPPMOADataCollatorWithPadding:
    """
    Standalone DPO-style collator for BPP-MOA.

    It does NOT depend on src.trainer.dpo_trainer or TRL.
    Each dataset item must contain:
      - raw_prompt
      - chosen
      - rejected
      - bpp_rho
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    label_pad_token_id: int = -100

    def _encode_prompt_response(self, raw_prompt: str, response: str) -> Dict[str, List[int]]:
        prompt = _format_prompt(self.prompt_template, raw_prompt)
        response = response or ""

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        eos = self.tokenizer.eos_token_id
        if eos is not None:
            if len(response_ids) == 0 or response_ids[-1] != eos:
                response_ids = response_ids + [eos]

        input_ids = prompt_ids + response_ids
        labels = [self.label_pad_token_id] * len(prompt_ids) + response_ids

        # Right truncation. This matches the simple MODPO-style max_length behavior.
        # If a prompt is too long and masks all response labels, keep the last token trainable
        # to avoid zero-token log-prob sums.
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        if all(x == self.label_pad_token_id for x in labels):
            labels[-1] = input_ids[-1]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _pad(self, encoded_list: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in encoded_list)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = []
        attention_mask = []
        labels = []

        for x in encoded_list:
            n = len(x["input_ids"])
            pad_n = max_len - n

            input_ids.append(x["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(x["attention_mask"] + [0] * pad_n)
            labels.append(x["labels"] + [self.label_pad_token_id] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        for key in ["raw_prompt", "chosen", "rejected", "bpp_rho"]:
            if key not in features[0]:
                raise KeyError(
                    f"BPP-MOA dataset item must contain '{key}'. "
                    "Did build_bpp_targets.py finish successfully?"
                )

        chosen_encoded = [
            self._encode_prompt_response(str(f["raw_prompt"]), str(f["chosen"]))
            for f in features
        ]
        rejected_encoded = [
            self._encode_prompt_response(str(f["raw_prompt"]), str(f["rejected"]))
            for f in features
        ]

        chosen = self._pad(chosen_encoded)
        rejected = self._pad(rejected_encoded)

        batch = {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            "bpp_rho": torch.tensor([float(f["bpp_rho"]) for f in features], dtype=torch.float32),
        }
        return batch


class BPPMOATrainer(Trainer):
    """
    Standalone BPP-MOA trainer.

    This trainer implements the DPO implicit policy logit:

        beta * [
          log pi_theta(y_chosen | x) - log pi_ref(y_chosen | x)
          - log pi_theta(y_rejected | x) + log pi_ref(y_rejected | x)
        ]

    and trains it against the BPP-MOA posterior-pooled soft target rho_w:

        L = - rho_w log sigmoid(logit)
            - (1-rho_w) log sigmoid(-logit)

    Important:
    - No import from src.trainer.dpo_trainer.
    - No import from trl.
    - No import from deepspeed.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        args: TrainingArguments = None,
        tokenize_map_func: Optional[Any] = None,  # kept only for old-call compatibility
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Any] = None,
        peft_config: Optional[Dict] = None,  # unused, kept for compatibility
        disable_dropout: bool = True,
        max_length: Optional[int] = 1024,
        num_proc: Optional[int] = 4,  # unused, kept for compatibility
        generate_during_eval: bool = False,  # unused, kept for compatibility
        compute_metrics: Optional[Any] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        target_eps: float = 1e-4,
    ):
        if loss_type != "sigmoid":
            raise ValueError("BPPMOATrainer supports only sigmoid soft-label loss.")

        if tokenizer is None:
            raise ValueError("BPPMOATrainer requires a tokenizer.")

        self.beta = float(beta)
        self.label_pad_token_id = label_pad_token_id
        self.target_eps = float(target_eps)
        self.ref_model = ref_model
        self.generate_during_eval = bool(generate_during_eval)
        self.wandb_eval_generation_n = 8
        self.wandb_eval_generation_max_new_tokens = 128
        self.wandb_eval_generation_do_sample = False
        self._last_wandb_generation_step = -1
        self.prompt_template = prompt_template
        self.max_length = int(max_length or 1024)

        if disable_dropout and model is not None:
            self._disable_dropout(model)
        if disable_dropout and ref_model is not None:
            self._disable_dropout(ref_model)

        if data_collator is None:
            data_collator = BPPMOADataCollatorWithPadding(
                tokenizer=tokenizer,
                max_length=max_length or 1024,
                prompt_template=prompt_template,
                label_pad_token_id=label_pad_token_id,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
        )

        # If a separate ref_model is provided, freeze it.
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _disable_dropout(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def _sequence_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

        # Predict token t using logits at t-1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].clone()

        loss_mask = shift_labels.ne(self.label_pad_token_id)
        safe_labels = shift_labels.masked_fill(~loss_mask, 0)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)

        token_logps = token_logps * loss_mask.to(token_logps.dtype)
        return token_logps.sum(dim=-1)

    def _unwrap_for_adapter_control(self, model: nn.Module) -> nn.Module:
        """
        In distributed training, Trainer/Accelerate may pass a DDP-wrapped model
        into compute_loss. The PEFT methods such as disable_adapter() live on the
        underlying module, not on DistributedDataParallel itself.
        """
        candidates = [model]

        if hasattr(model, "module"):
            candidates.append(model.module)

        if hasattr(self, "accelerator"):
            try:
                candidates.append(self.accelerator.unwrap_model(model))
            except Exception:
                pass

        # Also check one level deeper for common wrappers.
        expanded = []
        for m in candidates:
            if m is None:
                continue
            expanded.append(m)
            for attr in ["model", "base_model"]:
                child = getattr(m, attr, None)
                if child is not None:
                    expanded.append(child)

        for m in expanded:
            if hasattr(m, "disable_adapter") or hasattr(m, "disable_adapters"):
                return m

        return model

    def _reference_context(self, model: nn.Module):
        """
        For PEFT models, use the same model with adapters disabled as pi_ref.
        This avoids deepcopying a 7B model and avoids TRL entirely.

        Important:
        Under DDP, `model` is usually a DistributedDataParallel wrapper.
        Therefore we first unwrap it and call disable_adapter() on the
        underlying PEFT model.
        """
        if self.ref_model is not None:
            return nullcontext(self.ref_model)

        peft_model = self._unwrap_for_adapter_control(model)

        if hasattr(peft_model, "disable_adapter"):
            return peft_model.disable_adapter()

        if hasattr(peft_model, "disable_adapters"):
            return peft_model.disable_adapters()

        raise RuntimeError(
            "No ref_model was provided and no PEFT disable_adapter() method was found. "
            f"model type={type(model)}, unwrapped type={type(peft_model)}. "
            "This usually means the model is not a PEFT/LoRA model or prepare_model_for_peft "
            "returned a plain causal LM."
        )

    def _get_reference_logps(
        self,
        model: nn.Module,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.ref_model is not None:
                ref = self.ref_model
                ref = ref.to(chosen_input_ids.device)
                chosen_ref_logps = self._sequence_logps(
                    ref,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                )
                rejected_ref_logps = self._sequence_logps(
                    ref,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                )
            else:
                # PEFT path: disable LoRA adapter and use base SFT as reference.
                with self._reference_context(model):
                    chosen_ref_logps = self._sequence_logps(
                        model,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_labels,
                    )
                    rejected_ref_logps = self._sequence_logps(
                        model,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_labels,
                    )

        return chosen_ref_logps, rejected_ref_logps


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """
        Custom evaluation step for pairwise BPP-MOA batches.

        The default transformers.Trainer.prediction_step calls:
            model(**inputs)

        That fails because our batch contains keys such as:
            chosen_input_ids, rejected_input_ids, bpp_rho

        Therefore eval must also go through compute_loss().
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            context_manager = (
                self.compute_loss_context_manager()
                if hasattr(self, "compute_loss_context_manager")
                else nullcontext()
            )
            with context_manager:
                loss = self.compute_loss(model, inputs)

        loss = loss.detach()

        # We only need eval_loss. No logits/labels are returned.
        return loss, None, None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        chosen_labels = inputs["chosen_labels"]

        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        rejected_labels = inputs["rejected_labels"]

        target_probs = inputs["bpp_rho"].to(chosen_input_ids.device)
        target_probs = target_probs.clamp(self.target_eps, 1.0 - self.target_eps)

        # Compute reference log-probs first.
        # Reference forward is no_grad, so doing it before policy forward avoids
        # running the reference model while policy activations are still kept for backward.
        # This significantly lowers the peak memory of DPO/BPP-MOA training.
        reference_chosen_logps, reference_rejected_logps = self._get_reference_logps(
            model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        policy_chosen_logps = self._sequence_logps(
            model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_labels,
        )
        policy_rejected_logps = self._sequence_logps(
            model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_labels,
        )

        logits = self.beta * (
            (policy_chosen_logps - reference_chosen_logps)
            - (policy_rejected_logps - reference_rejected_logps)
        )

        target_probs = target_probs.to(dtype=logits.dtype)
        losses = (
            -target_probs * F.logsigmoid(logits)
            -(1.0 - target_probs) * F.logsigmoid(-logits)
        )
        loss = losses.mean()

        with torch.no_grad():
            policy_probs = torch.sigmoid(logits)
            prefix = "train" if model.training else "eval"
            self.log(
                {
                    f"{prefix}/bpp_loss": loss.detach().float().item(),
                    f"{prefix}/bpp_target_mean": target_probs.detach().float().mean().item(),
                    f"{prefix}/bpp_policy_prob_mean": policy_probs.detach().float().mean().item(),
                    f"{prefix}/bpp_abs_error": (policy_probs - target_probs).detach().float().abs().mean().item(),
                    f"{prefix}/bpp_logit_mean": logits.detach().float().mean().item(),
                }
            )

        if return_outputs:
            return loss, {
                "logits": logits.detach(),
                "policy_probs": policy_probs.detach(),
                "target_probs": target_probs.detach(),
            }
        return loss

    def _format_prompt_for_generation(self, raw_prompt: str) -> str:
        text = str(raw_prompt)
        if "BEGINNING OF CONVERSATION:" in text and "ASSISTANT:" in text:
            return text
        return _format_prompt(self.prompt_template, text)

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

        raw_prompts = [str(dataset[int(idx)]["raw_prompt"]) for idx in indices]
        model_inputs = [self._format_prompt_for_generation(p) for p in raw_prompts]

        model = self.model
        was_training = model.training
        model.eval()

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            inputs = self.tokenizer(
                model_inputs,
                padding=True,
                truncation=True,
                max_length=min(self.max_length, 1024),
                return_tensors="pt",
            ).to(self.args.device)

            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.wandb_eval_generation_max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=self.wandb_eval_generation_do_sample,
            )
            prompt_len = inputs["input_ids"].shape[1]
            responses = self.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

            table = wandb.Table(columns=["global_step", "eval_index", "raw_prompt", "model_input", "generation"])
            for idx, raw_prompt, model_input, response in zip(indices, raw_prompts, model_inputs, responses):
                table.add_data(step, int(idx), raw_prompt, model_input, response)

            wandb.log({"eval/generation_samples": table, "eval/generation_step": step}, step=step)
        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

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
