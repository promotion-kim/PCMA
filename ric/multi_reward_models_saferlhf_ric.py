"""SafeRLHF/Beaver reward model wrapper for RiC baselines.

This file is intentionally separate from the original HH-RLHF RiC code.
It uses the same reward models as the SafeRLHF MORLHF/PC-MORLHF experiments:

  helpful  = PKU-Alignment/beaver-7b-v1.0-reward
  harmless = - PKU-Alignment/beaver-7b-v1.0-cost

The class returns one score list per objective. If ``reward_stats_path`` is
provided and ``normalize=True``, scores are normalized by the saved training
mean/std so that they match the score tokens used in RiC prompts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from safe_rlhf.models import AutoModelForScore


def _as_float_list(x: Sequence[float] | str) -> List[float]:
    if isinstance(x, str):
        return [float(v.strip()) for v in x.split(",") if v.strip()]
    return [float(v) for v in x]


class SafeRLHFRewardModels:
    """Batched scorer for Beaver reward/cost models.

    Parameters
    ----------
    reward_model_names:
        Hugging Face ids or local paths for objective scorers.
    reward_tokenizer_names:
        Tokenizer ids. Defaults to ``reward_model_names``.
    reward_signs:
        Sign applied to each model output. Use ``1,-1`` for reward,cost.
    reward_stats_path:
        Optional ``all_reward_stat.npy`` saved by the dataset preparation script.
        Expected shape: ``(num_rewards, 2)`` with columns ``mean, std``.
    """

    def __init__(
        self,
        reward_model_names: Sequence[str],
        reward_tokenizer_names: Optional[Sequence[str]] = None,
        reward_signs: Optional[Sequence[float] | str] = None,
        gpu_id: int = 0,
        reward_stats_path: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.reward_model_names = [str(x).strip() for x in reward_model_names]
        self.reward_tokenizer_names = [str(x).strip() for x in (reward_tokenizer_names or reward_model_names)]
        self.reward_signs = _as_float_list(reward_signs or [1.0] * len(self.reward_model_names))
        self.num_rewards = len(self.reward_model_names)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        if not (len(self.reward_model_names) == len(self.reward_tokenizer_names) == len(self.reward_signs)):
            raise ValueError("reward_model_names, reward_tokenizer_names, and reward_signs must have same length.")

        self.reward_stats = None
        if reward_stats_path:
            stats = np.load(reward_stats_path)
            if stats.shape[0] != self.num_rewards or stats.shape[1] != 2:
                raise ValueError(
                    f"reward_stats_path has shape {stats.shape}; expected ({self.num_rewards}, 2)."
                )
            self.reward_stats = stats.astype(np.float32)

        self.models = []
        self.tokenizers = []
        print("Loading SafeRLHF reward/cost models...")
        for model_name, tokenizer_name in zip(self.reward_model_names, self.reward_tokenizer_names):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForScore.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map={"": gpu_id} if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()
            self.tokenizers.append(tokenizer)
            self.models.append(model)
        print("Loaded scorers:", self.reward_model_names)

    @staticmethod
    def _extract_score(output: object) -> torch.Tensor:
        scores = getattr(output, "end_scores", None)
        if scores is None:
            scores = getattr(output, "scores", None)
        if scores is None:
            raise RuntimeError(f"AutoModelForScore output has no end_scores/scores: {output}")
        if scores.ndim == 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        elif scores.ndim >= 2:
            scores = scores[:, -1]
        return scores.float()

    @torch.no_grad()
    def get_reward_model_scores(
        self,
        queries_responses: Sequence[Tuple[str, str]],
        normalize: bool = False,
    ) -> List[List[float]]:
        """Return ``[scores_for_obj1, scores_for_obj2, ...]``.

        ``queries_responses`` should contain formatted prompt/query and response.
        The Beaver scorer receives their concatenation, matching the existing
        SafeRLHF MORLHF code path.
        """
        texts = [str(q) + str(r) for q, r in queries_responses]
        all_scores: List[List[float]] = []

        for obj_idx, (model, tokenizer, sign) in enumerate(zip(self.models, self.tokenizers, self.reward_signs)):
            obj_scores: List[float] = []
            loader = DataLoader(texts, batch_size=self.batch_size, shuffle=False)
            for batch_texts in tqdm(loader, desc=f"scoring objective {obj_idx+1}", dynamic_ncols=True):
                inputs = tokenizer(
                    list(batch_texts),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = model(**inputs)
                values = self._extract_score(output).detach().cpu().numpy().astype(np.float32)
                values = values * float(sign)

                if normalize:
                    if self.reward_stats is None:
                        raise ValueError("normalize=True requires reward_stats_path at construction time.")
                    mean, std = self.reward_stats[obj_idx]
                    if float(std) == 0.0:
                        raise ValueError(f"Reward std for objective {obj_idx} is zero; cannot normalize.")
                    values = (values - float(mean)) / float(std)

                obj_scores.extend([float(v) for v in values])
            all_scores.append(obj_scores)
        return all_scores
