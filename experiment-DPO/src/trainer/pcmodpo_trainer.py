"""Posterior-calibrated MODPO trainer.

This trainer is a minimal extension of the existing MODPOTrainer. It replaces
fixed scalarization weights w with prompt-dependent posterior-calibrated weights
alpha_i(x,w) produced by `LaplacePosteriorCalibrator`.
"""

from __future__ import annotations

import random
import textwrap
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import wandb

from src.trainer.modpo_trainer import MODPOTrainer
from src.utils import RewardWrapperInput
from src.utils.posterior_calibration import HFPromptFeatureExtractor, LaplacePosteriorCalibrator


class PCMODPOTrainer(MODPOTrainer):
    """MODPO with posterior-calibrated prompt-dependent weights.

    Assumes the training dataset is the anchor objective dataset, e.g. `better`,
    and the margin reward wrappers correspond to all non-anchor objectives, e.g.
    `safer`.

    For two objectives and anchor_idx=0, this implements

        logit = 1/alpha_0(x) * [ beta Δlog(pi/ref) - alpha_1(x) Δr_margin ].
    """

    def __init__(
        self,
        *args,
        alpha_min: float = 0.05,
        debug_print_alpha_every: int = 50,
        debug_print_alpha_n: int = 3,
        wandb_eval_generation_n: int = 5,
        wandb_eval_generation_max_new_tokens: int = 128,
        wandb_eval_generation_do_sample: bool = True,
        wandb_eval_generation_temperature: float = 0.7,
        wandb_eval_generation_top_p: float = 0.9,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Alpha stabilization.
        # We use epsilon-smoothing instead of plain clamp so that alpha remains
        # a normalized scalarization weight.
        self.alpha_min = float(alpha_min)

        # Terminal alpha debugging.
        self.debug_print_alpha_every = debug_print_alpha_every
        self.debug_print_alpha_n = debug_print_alpha_n

        # W&B eval generation logging.
        self.wandb_eval_generation_n = wandb_eval_generation_n
        self.wandb_eval_generation_max_new_tokens = wandb_eval_generation_max_new_tokens
        self.wandb_eval_generation_do_sample = wandb_eval_generation_do_sample
        self.wandb_eval_generation_temperature = wandb_eval_generation_temperature
        self.wandb_eval_generation_top_p = wandb_eval_generation_top_p
        self.prompt_template = prompt_template

        self._last_alpha_print_step = None
        self._last_wandb_generation_step = None

    def set_posterior_calibrator(
        self,
        posterior_calibrator: LaplacePosteriorCalibrator,
        prompt_feature_extractor: HFPromptFeatureExtractor,
        posterior_num_samples: int = 0,
        anchor_objective_idx: int = 0,
    ) -> None:
        self.posterior_calibrator = posterior_calibrator
        self.prompt_feature_extractor = prompt_feature_extractor
        self.posterior_num_samples = int(posterior_num_samples)
        self.anchor_objective_idx = int(anchor_objective_idx)

    '''def _stabilize_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """Clip alpha and renormalize it to stay on the simplex.

        This uses the simple rule:

            alpha <- clamp_min(alpha, alpha_min)
            alpha <- alpha / sum(alpha)

        Note:
            After renormalization, the final minimum value may be slightly smaller
            than alpha_min. This is acceptable as a simple numerical stabilization.
        """
        alpha = alpha.float()

        # First normalize in case expected_alpha has small numerical drift.
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        eps = float(getattr(self, "alpha_min", 0.0))
        if eps <= 0.0:
            return alpha

        alpha = alpha.clamp_min(eps)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        return alpha'''

    def _batch_alpha(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        batch_size: int,
    ) -> torch.Tensor:
        if not hasattr(self, "posterior_calibrator"):
            raise RuntimeError("Call set_posterior_calibrator(...) before training.")

        raw_prompt = batch["raw_prompt"]

        # MODPODataCollator duplicates raw prompts as:
        # [chosen prompts] + [rejected prompts].
        prompts = raw_prompt[:batch_size] if len(raw_prompt) >= 2 * batch_size else raw_prompt

        features = self.prompt_feature_extractor.encode(
            prompts,
            batch_size=min(32, len(prompts)),
            device=self.accelerator.device,
        )

        alpha = self.posterior_calibrator.expected_alpha(
            features,
            base_w=self.w.detach().float().cpu(),
            num_samples=self.posterior_num_samples,
            device=self.accelerator.device,
        )

        #alpha = self._stabilize_alpha(alpha)
        return alpha.to(self.accelerator.device)

    def pcmodpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_margin_reward: torch.FloatTensor,
        rejected_margin_reward: torch.FloatTensor,
        alpha: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        m = self.w.numel()
        anchor = self.anchor_objective_idx
        margin_indices = [i for i in range(m) if i != anchor]

        if chosen_margin_reward.shape[-1] != len(margin_indices):
            raise ValueError(
                f"Expected {len(margin_indices)} margin rewards, "
                f"got {chosen_margin_reward.shape[-1]}. "
                "The margin reward wrappers must correspond to all non-anchor objectives."
            )

        alpha = alpha.to(dtype=policy_chosen_logps.dtype, device=policy_chosen_logps.device)
        alpha_anchor = alpha[:, anchor].clamp_min(1e-6)
        alpha_margin = alpha[:, margin_indices]

        chosen_margin = (chosen_margin_reward * alpha_margin).sum(dim=-1)
        rejected_margin = (rejected_margin_reward * alpha_margin).sum(dim=-1)

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps)
            - chosen_margin
        ) / alpha_anchor

        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps)
            - rejected_margin
        ) / alpha_anchor

        logits = chosen_rewards - rejected_rewards

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. "
                "Should be one of ['sigmoid', 'hinge']"
            )

        return losses, chosen_rewards.detach(), rejected_rewards.detach(), alpha.detach()

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
        ) = self.forward(model, batch)

        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.forward(self.ref_model, batch)

        margin_reward_list = self.wrapped_margin_reward_model_list(
            RewardWrapperInput(
                raw_prompt=batch["raw_prompt"],
                response=batch["response"],
            )
        )

        margin_rewards = torch.stack(margin_reward_list, dim=-1).to(
            policy_chosen_logps.dtype
        ).to(self.accelerator.device)

        chosen_margin_rewards, rejected_margin_rewards = margin_rewards.chunk(2)

        batch_size = policy_chosen_logps.shape[0]
        alpha = self._batch_alpha(batch, batch_size=batch_size)

        raw_prompts_for_alpha = self._get_unique_raw_prompts(batch, batch_size)

        self._print_alpha_samples(
            raw_prompts=raw_prompts_for_alpha,
            alpha=alpha,
            train_eval=train_eval,
        )

        losses, chosen_rewards, rejected_rewards, alpha = self.pcmodpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_margin_rewards,
            rejected_margin_rewards,
            alpha,
        )

        accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""

        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu()
        metrics[f"{prefix}logps/margins"] = (
            policy_chosen_logps - policy_rejected_logps
        ).detach().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu()

        # Alpha diagnostics.
        for i in range(alpha.shape[-1]):
            metrics[f"{prefix}pcmodpo/alpha_{i}_mean"] = alpha[:, i].mean().detach().cpu()
            metrics[f"{prefix}pcmodpo/alpha_{i}_min"] = alpha[:, i].min().detach().cpu()
            metrics[f"{prefix}pcmodpo/alpha_{i}_max"] = alpha[:, i].max().detach().cpu()

        # Margin reward diagnostics. Useful for detecting reward-scale explosion.
        margin_delta = chosen_margin_rewards - rejected_margin_rewards
        for j in range(margin_delta.shape[-1]):
            metrics[f"{prefix}pcmodpo/margin_delta_{j}_mean"] = (
                margin_delta[:, j].mean().detach().cpu()
            )
            metrics[f"{prefix}pcmodpo/margin_delta_{j}_min"] = (
                margin_delta[:, j].min().detach().cpu()
            )
            metrics[f"{prefix}pcmodpo/margin_delta_{j}_max"] = (
                margin_delta[:, j].max().detach().cpu()
            )
            metrics[f"{prefix}pcmodpo/margin_delta_{j}_abs_mean"] = (
                margin_delta[:, j].abs().mean().detach().cpu()
            )

        if train_eval == "train":
            metrics[f"{prefix}accuracy"] = accuracies.detach().cpu()

        return losses.mean(), metrics

    def _get_unique_raw_prompts(self, batch, batch_size: int):
        """Keep only the first B prompts.

        MODPO collator duplicates raw_prompt for chosen/rejected:
            [chosen prompts] + [rejected prompts]
        """
        raw_prompts = batch.get("raw_prompt", None)
        if raw_prompts is None:
            return None
        return list(raw_prompts[:batch_size])

    def _print_alpha_samples(self, raw_prompts, alpha: torch.Tensor, train_eval: str):
        """Print stabilized posterior-calibrated alpha values for a few prompts."""
        if not self.accelerator.is_local_main_process:
            return
        if raw_prompts is None:
            return
        if self.debug_print_alpha_every is None or self.debug_print_alpha_every <= 0:
            return

        step = int(getattr(self.state, "global_step", 0))
        key = (train_eval, step)

        # Avoid printing repeatedly for the same step.
        if self._last_alpha_print_step == key:
            return

        if step % self.debug_print_alpha_every != 0:
            return

        self._last_alpha_print_step = key

        n = min(self.debug_print_alpha_n, len(raw_prompts), alpha.shape[0])

        print("\n" + "=" * 100, flush=True)
        print(
            f"[PC-MODPO stabilized alpha samples] "
            f"split={train_eval}, global_step={step}",
            flush=True,
        )

        for i in range(n):
            prompt = str(raw_prompts[i]).replace("\n", " ")
            prompt = textwrap.shorten(prompt, width=220, placeholder=" ...")

            alpha_str = " ".join(
                [
                    f"obj{j}={alpha[i, j].detach().float().item():.4f}"
                    for j in range(alpha.shape[1])
                ]
            )

            print(f"[sample {i}] prompt: {prompt}", flush=True)
            print(f"           alpha: {alpha_str}", flush=True)

        print("=" * 100 + "\n", flush=True)

    def _format_prompt_for_generation(self, raw_prompt: str) -> str:
        text = str(raw_prompt)

        # Avoid double-formatting if prompt is already templated.
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
        raise KeyError("Eval dataset example must contain `raw_prompt` or `prompt`.")

    @torch.no_grad()
    def _log_eval_generations_to_wandb(self, dataloader):
        """Sample eval prompts, generate with current model, and log to W&B."""
        if not self.accelerator.is_local_main_process:
            return
        if self.wandb_eval_generation_n is None or self.wandb_eval_generation_n <= 0:
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

        raw_prompts = [
            self._extract_raw_prompt_from_eval_example(dataset[int(idx)])
            for idx in indices
        ]
        model_inputs_text = [
            self._format_prompt_for_generation(p)
            for p in raw_prompts
        ]

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        was_training = unwrapped_model.training
        unwrapped_model.eval()

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
            ).to(self.accelerator.device)

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

            generated = unwrapped_model.generate(**gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            responses = self.tokenizer.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
            )

            table = wandb.Table(
                columns=[
                    "global_step",
                    "eval_index",
                    "raw_prompt",
                    "model_input",
                    "generation",
                ]
            )

            for idx, raw_prompt, model_input, response in zip(
                indices,
                raw_prompts,
                model_inputs_text,
                responses,
            ):
                table.add_data(
                    step,
                    int(idx),
                    raw_prompt,
                    model_input,
                    response,
                )

            wandb.log(
                {
                    "eval/generation_samples": table,
                    "eval/generation_step": step,
                },
                step=step,
            )

            print(
                f"[PC-MODPO] logged {n} eval generations to wandb at step={step}",
                flush=True,
            )

        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                unwrapped_model.train()

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """Run normal evaluation and also log generation samples to W&B."""
        self._log_eval_generations_to_wandb(dataloader)
        self.accelerator.wait_for_everyone()

        return super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )