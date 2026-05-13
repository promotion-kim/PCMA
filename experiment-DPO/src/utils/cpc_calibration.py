"""Objective-level calibration utilities for Confidence-Preserving Calibration (CPC).

This module intentionally implements the prompt-independent default from the
paper revision:

    a_i \in R,      kappa_i = exp(a_i),
    c_i(w, gamma) = w_i * exp(gamma * a_i).

It does not compute normalized alpha.  If you need the normalized relative
weights only for logging, use `relative_weights(...)`; training should use
`coefficients(...)`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class CPCFitConfig:
    num_steps: int = 2000
    lr: float = 3e-3
    batch_size: int = 512
    prior_sigma_a: float = 2.0
    log_every: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CPCMetadata:
    num_objectives: int
    objective_names: Optional[List[str]] = None
    objective_adapter_names: Optional[List[str]] = None
    objective_dataset_names: Optional[List[str]] = None
    scorer_type: str = "implicit_logratio_gap"
    fit_config: Optional[Dict[str, Any]] = None


class ObjectiveLogPrecisionCalibrator:
    """Prompt-independent objective-level log-precision calibrator.

    The fitted likelihood is

        P(z_{in}=1 | gap_{in}, a_i) = sigmoid(exp(a_i) * signed_gap_{in}).

    The downstream CPC coefficients are

        c_i(w, gamma) = w_i * exp(gamma * a_i).

    Parameters
    ----------
    log_precision:
        Tensor of shape [m].  Entry i is a_i.
    metadata:
        Optional experiment metadata saved with the calibrator.
    """

    def __init__(self, log_precision: torch.Tensor, metadata: Optional[CPCMetadata] = None):
        if log_precision.ndim != 1:
            raise ValueError(f"log_precision must be 1D, got shape={tuple(log_precision.shape)}")
        self.log_precision = log_precision.detach().float().cpu()
        self.metadata = metadata or CPCMetadata(num_objectives=int(log_precision.numel()))
        if self.metadata.num_objectives != int(log_precision.numel()):
            raise ValueError(
                f"metadata.num_objectives={self.metadata.num_objectives} does not match "
                f"log_precision size={int(log_precision.numel())}"
            )

    @property
    def num_objectives(self) -> int:
        return int(self.log_precision.numel())

    @classmethod
    def fit(
        cls,
        *,
        objective_idx: torch.Tensor,
        signed_gaps: torch.Tensor,
        num_objectives: int,
        fit_cfg: CPCFitConfig,
        objective_names: Optional[Sequence[str]] = None,
        objective_adapter_names: Optional[Sequence[str]] = None,
        objective_dataset_names: Optional[Sequence[str]] = None,
    ) -> "ObjectiveLogPrecisionCalibrator":
        """Fit objective-level log precision by MAP.

        objective_idx: Long tensor [N] with values in {0, ..., m-1}.
        signed_gaps: Float tensor [N], already signed so that positive means the
            labeled/chosen response is preferred for that objective.
        """
        if objective_idx.ndim != 1 or signed_gaps.ndim != 1:
            raise ValueError("objective_idx and signed_gaps must be 1D tensors.")
        if objective_idx.numel() != signed_gaps.numel():
            raise ValueError("objective_idx and signed_gaps must have the same length.")
        if signed_gaps.numel() == 0:
            raise ValueError("Cannot fit CPC calibrator with zero examples.")

        device = torch.device(fit_cfg.device)
        torch.manual_seed(fit_cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(fit_cfg.seed)

        obj = objective_idx.long().to(device)
        gaps = signed_gaps.float().to(device)
        a = torch.zeros(num_objectives, dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([a], lr=fit_cfg.lr)

        n = int(gaps.numel())
        batch_size = int(fit_cfg.batch_size) if fit_cfg.batch_size and fit_cfg.batch_size > 0 else n
        log_every = int(fit_cfg.log_every) if fit_cfg.log_every is not None else 0

        for step in range(int(fit_cfg.num_steps)):
            if batch_size >= n:
                batch_obj = obj
                batch_gaps = gaps
            else:
                idx = torch.randint(0, n, (batch_size,), device=device)
                batch_obj = obj[idx]
                batch_gaps = gaps[idx]

            logits = torch.exp(a[batch_obj]) * batch_gaps
            nll = F.softplus(-logits).mean()
            # The paper objective averages NLL by data size and scales the prior
            # by 1/N.  With minibatches, using full N here keeps the prior weak and
            # comparable to full-batch MAP.
            prior = torch.sum(a ** 2) / (2.0 * float(n) * (fit_cfg.prior_sigma_a ** 2))
            loss = nll + prior

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if log_every > 0 and (step % log_every == 0 or step == fit_cfg.num_steps - 1):
                with torch.no_grad():
                    full_logits = torch.exp(a[obj]) * gaps
                    full_nll = F.softplus(-full_logits).mean().item()
                    print(
                        f"[CPC fit] step={step:05d} loss={loss.item():.6f} "
                        f"batch_nll={nll.item():.6f} full_nll={full_nll:.6f} "
                        f"a={a.detach().cpu().tolist()}",
                        flush=True,
                    )

        metadata = CPCMetadata(
            num_objectives=num_objectives,
            objective_names=list(objective_names) if objective_names is not None else None,
            objective_adapter_names=list(objective_adapter_names) if objective_adapter_names is not None else None,
            objective_dataset_names=list(objective_dataset_names) if objective_dataset_names is not None else None,
            fit_config=asdict(fit_cfg),
        )
        return cls(a.detach().cpu(), metadata=metadata)

    def coefficients(
        self,
        base_w: torch.Tensor,
        *,
        gamma: float = 1.0,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        coefficient_floor: float = 0.0,
        coefficient_ceiling: Optional[float] = None,
    ) -> torch.Tensor:
        """Return CPC coefficients c_i = w_i exp(gamma a_i).

        This is the object that should replace fixed scalarization coefficients
        in downstream optimizers.
        """
        if base_w.ndim != 1:
            raise ValueError(f"base_w must be 1D, got shape={tuple(base_w.shape)}")
        if base_w.numel() != self.num_objectives:
            raise ValueError(f"base_w has {base_w.numel()} entries, expected {self.num_objectives}.")
        dev = torch.device(device) if device is not None else base_w.device
        dt = dtype or base_w.dtype
        a = self.log_precision.to(device=dev, dtype=dt)
        w = base_w.to(device=dev, dtype=dt)
        coeff = w * torch.exp(torch.as_tensor(gamma, device=dev, dtype=dt) * a)
        if coefficient_floor and coefficient_floor > 0:
            coeff = coeff.clamp_min(float(coefficient_floor))
        if coefficient_ceiling is not None:
            coeff = coeff.clamp_max(float(coefficient_ceiling))
        return coeff

    def batch_coefficients(
        self,
        batch_size: int,
        base_w: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        coeff = self.coefficients(base_w, **kwargs)
        return coeff.unsqueeze(0).expand(int(batch_size), -1).contiguous()

    def relative_weights(
        self,
        base_w: torch.Tensor,
        *,
        gamma: float = 1.0,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return normalized relative weights for logging only.

        alpha_i = c_i / sum_j c_j.  Do not use this for CPC training unless you
        also preserve rho=sum_j c_j in the logit.
        """
        coeff = self.coefficients(base_w, gamma=gamma, device=device, dtype=dtype)
        denom = coeff.sum().clamp_min(1e-12)
        return coeff / denom

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "objective_log_precision_cpc_v1",
            "log_precision": self.log_precision.tolist(),
            "precision": torch.exp(self.log_precision).tolist(),
            "metadata": asdict(self.metadata),
        }
        with open(output_dir / "cpc_calibrator.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ObjectiveLogPrecisionCalibrator":
        path = Path(path)
        json_path = path / "cpc_calibrator.json" if path.is_dir() else path
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") != "objective_log_precision_cpc_v1":
            raise ValueError(f"Unsupported CPC calibrator format: {payload.get('format')}")
        log_precision = torch.tensor(payload["log_precision"], dtype=torch.float32)
        metadata_dict = payload.get("metadata", {})
        metadata = CPCMetadata(**metadata_dict)
        return cls(log_precision, metadata=metadata)
