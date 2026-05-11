"""
Posterior calibration utilities for PC-MODPO.

This module implements the finite-dimensional calibration model from the note:

    a_i(x) = rho(x) + delta_i(x)
    rho(x) = mu + c^T phi(x)
    delta(x) = P_perp (u + W phi(x))

where sum_i delta_i(x) = 0.  The posterior-calibrated weights use only
relative calibration:

    alpha_i(x, w; theta) = w_i exp(delta_i(x; theta)) /
        sum_j w_j exp(delta_j(x; theta)).

The posterior is approximated by MAP + diagonal Laplace approximation.  The
implementation is intentionally conservative: it uses a damped/clipped diagonal
precision because the logistic likelihood in log-precision can be non-convex.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibrationPriorConfig:
    """Gaussian prior scales for theta=(mu, c, u, W)."""

    mu_mean: float = 0.0
    mu_std: float = 1.0
    c_std: float = 1.0
    u_std: float = 0.5
    w_std: float = 0.5
    min_precision: float = 1e-4
    laplace_damping: float = 1e-3


@dataclass
class CalibrationFitConfig:
    num_steps: int = 2000
    lr: float = 3e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    log_every: int = 100
    laplace_batch_size: int = 2048
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PosteriorCalibrationModel(nn.Module):
    """Finite-dimensional log-calibration model.

    Args:
        num_objectives: number of objectives m.
        feature_dim: dimension of prompt features phi(x).
    """

    def __init__(self, num_objectives: int, feature_dim: int):
        super().__init__()
        if num_objectives < 2:
            raise ValueError("PC calibration needs at least two objectives.")
        self.num_objectives = int(num_objectives)
        self.feature_dim = int(feature_dim)

        self.mu = nn.Parameter(torch.zeros(()))
        self.c = nn.Parameter(torch.zeros(feature_dim))
        self.u = nn.Parameter(torch.zeros(num_objectives))
        self.W = nn.Parameter(torch.zeros(num_objectives, feature_dim))

    @property
    def flat_dim(self) -> int:
        return 1 + self.feature_dim + self.num_objectives + self.num_objectives * self.feature_dim

    def delta(self, features: torch.Tensor) -> torch.Tensor:
        """Return relative log-calibration delta(x), shape (B, m)."""
        raw = self.u.unsqueeze(0) + features @ self.W.t()
        return raw - raw.mean(dim=-1, keepdim=True)

    def rho(self, features: torch.Tensor) -> torch.Tensor:
        """Return absolute log-calibration rho(x), shape (B,)."""
        return self.mu + features @ self.c

    def a(self, features: torch.Tensor) -> torch.Tensor:
        """Return full log-precision a_i(x), shape (B, m)."""
        return self.rho(features).unsqueeze(-1) + self.delta(features)

    def alpha(self, features: torch.Tensor, base_w: torch.Tensor) -> torch.Tensor:
        """Return alpha_i(x,w), shape (B, m)."""
        base_w = base_w.to(device=features.device, dtype=features.dtype)
        if base_w.ndim != 1 or base_w.numel() != self.num_objectives:
            raise ValueError(f"base_w must have shape ({self.num_objectives},), got {tuple(base_w.shape)}")
        delta = self.delta(features)
        logits = torch.log(base_w.clamp_min(1e-12)).unsqueeze(0) + delta
        return torch.softmax(logits, dim=-1)

    def prior_loss(self, prior: CalibrationPriorConfig) -> torch.Tensor:
        loss = 0.5 * ((self.mu - prior.mu_mean) / prior.mu_std).pow(2)
        loss = loss + 0.5 * (self.c / prior.c_std).pow(2).sum()
        loss = loss + 0.5 * (self.u / prior.u_std).pow(2).sum()
        loss = loss + 0.5 * (self.W / prior.w_std).pow(2).sum()
        return loss

    def nll(self, features: torch.Tensor, objective_idx: torch.Tensor, signed_gaps: torch.Tensor) -> torch.Tensor:
        """Negative BT log-likelihood.

        signed_gaps should be z * (g_i(x,y1)-g_i(x,y2)), so positive values mean
        the objective-specific policy already agrees with the observed preference.
        """
        a_all = self.a(features)
        a_i = a_all.gather(1, objective_idx.long().view(-1, 1)).squeeze(1)
        t = signed_gaps.to(a_i.dtype) * torch.exp(a_i)
        return F.softplus(-t).mean()

    def map_loss(
        self,
        features: torch.Tensor,
        objective_idx: torch.Tensor,
        signed_gaps: torch.Tensor,
        prior: CalibrationPriorConfig,
    ) -> torch.Tensor:
        return self.nll(features, objective_idx, signed_gaps) + self.prior_loss(prior) / max(1, features.shape[0])

    def flatten_parameters(self) -> torch.Tensor:
        return torch.cat([
            self.mu.reshape(-1),
            self.c.reshape(-1),
            self.u.reshape(-1),
            self.W.reshape(-1),
        ])

    def load_flat_parameters(self, flat: torch.Tensor) -> None:
        d = self.feature_dim
        m = self.num_objectives
        pos = 0
        with torch.no_grad():
            self.mu.copy_(flat[pos].reshape(())); pos += 1
            self.c.copy_(flat[pos:pos + d]); pos += d
            self.u.copy_(flat[pos:pos + m]); pos += m
            self.W.copy_(flat[pos:pos + m * d].reshape(m, d)); pos += m * d
        if pos != flat.numel():
            raise ValueError("flat parameter length mismatch")

    def flat_prior_precision(self, prior: CalibrationPriorConfig, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.mu.device
        d = self.feature_dim
        m = self.num_objectives
        pieces = [
            torch.full((1,), 1.0 / prior.mu_std**2, device=device),
            torch.full((d,), 1.0 / prior.c_std**2, device=device),
            torch.full((m,), 1.0 / prior.u_std**2, device=device),
            torch.full((m * d,), 1.0 / prior.w_std**2, device=device),
        ]
        return torch.cat(pieces)


def _standardize_features(features: torch.Tensor, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mean is None:
        mean = features.mean(dim=0)
    if std is None:
        std = features.std(dim=0).clamp_min(1e-6)
    return (features - mean) / std, mean, std


@dataclass
class LaplacePosteriorState:
    num_objectives: int
    feature_dim: int
    theta_map: List[float]
    precision_diag: List[float]
    feature_mean: List[float]
    feature_std: List[float]
    prior: Dict[str, Any]
    feature_model_name: Optional[str] = None


class LaplacePosteriorCalibrator:
    """MAP + diagonal Laplace posterior for prompt-dependent calibration."""

    def __init__(
        self,
        model: PosteriorCalibrationModel,
        precision_diag: torch.Tensor,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
        prior: Optional[CalibrationPriorConfig] = None,
        feature_model_name: Optional[str] = None,
    ):
        self.model = model
        self.precision_diag = precision_diag.detach().float().cpu()
        self.feature_mean = feature_mean.detach().float().cpu() if feature_mean is not None else torch.zeros(model.feature_dim)
        self.feature_std = feature_std.detach().float().cpu() if feature_std is not None else torch.ones(model.feature_dim)
        self.prior = prior or CalibrationPriorConfig()
        self.feature_model_name = feature_model_name

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        objective_idx: torch.Tensor,
        signed_gaps: torch.Tensor,
        num_objectives: int,
        prior: Optional[CalibrationPriorConfig] = None,
        fit_cfg: Optional[CalibrationFitConfig] = None,
        feature_model_name: Optional[str] = None,
    ) -> "LaplacePosteriorCalibrator":
        prior = prior or CalibrationPriorConfig()
        fit_cfg = fit_cfg or CalibrationFitConfig()
        torch.manual_seed(fit_cfg.seed)

        features = features.float()
        features, feat_mean, feat_std = _standardize_features(features)
        objective_idx = objective_idx.long()
        signed_gaps = signed_gaps.float()

        device = torch.device(fit_cfg.device)
        model = PosteriorCalibrationModel(num_objectives=num_objectives, feature_dim=features.shape[-1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=fit_cfg.lr, weight_decay=fit_cfg.weight_decay)

        n = features.shape[0]
        x = features.to(device)
        obj = objective_idx.to(device)
        gaps = signed_gaps.to(device)

        for step in range(1, fit_cfg.num_steps + 1):
            perm = torch.randint(0, n, (min(fit_cfg.batch_size, n),), device=device)
            loss = model.map_loss(x[perm], obj[perm], gaps[perm], prior)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if fit_cfg.log_every and step % fit_cfg.log_every == 0:
                print(f"[posterior-calibration] step={step} loss={loss.item():.6f}")

        precision_diag = cls._diagonal_laplace_precision(model, x, obj, gaps, prior, fit_cfg)
        return cls(
            model=model.cpu(),
            precision_diag=precision_diag.cpu(),
            feature_mean=feat_mean,
            feature_std=feat_std,
            prior=prior,
            feature_model_name=feature_model_name,
        )

    @staticmethod
    def _design_diag_terms(model: PosteriorCalibrationModel, features: torch.Tensor, objective_idx: torch.Tensor) -> torch.Tensor:
        """Return psi^2 for selected objective, shape (B, P)."""
        B, d = features.shape
        m = model.num_objectives
        device = features.device
        obj = objective_idx.long()

        # theta order: mu, c[d], u[m], W[m,d]
        P = model.flat_dim
        psi = torch.zeros(B, P, device=device, dtype=features.dtype)
        psi[:, 0] = 1.0                       # d a_i / d mu
        psi[:, 1:1 + d] = features             # d a_i / d c

        # P_perp row for selected objective: e_i - 1/m * 1
        row = -torch.ones(B, m, device=device, dtype=features.dtype) / float(m)
        row.scatter_add_(1, obj.view(-1, 1), torch.ones(B, 1, device=device, dtype=features.dtype))
        u_start = 1 + d
        W_start = u_start + m
        psi[:, u_start:W_start] = row
        psi[:, W_start:] = (row.unsqueeze(-1) * features.unsqueeze(1)).reshape(B, m * d)
        return psi.pow(2)

    @classmethod
    def _diagonal_laplace_precision(
        cls,
        model: PosteriorCalibrationModel,
        features: torch.Tensor,
        objective_idx: torch.Tensor,
        signed_gaps: torch.Tensor,
        prior: CalibrationPriorConfig,
        fit_cfg: CalibrationFitConfig,
    ) -> torch.Tensor:
        """Damped diagonal negative Hessian of log posterior.

        The likelihood is not globally log-concave in log-precision.  We therefore
        clamp the diagonal precision to a small positive value after adding the
        Gaussian prior precision and damping.
        """
        device = features.device
        P = model.flat_dim
        precision = model.flat_prior_precision(prior, device=device).clone()

        bs = min(fit_cfg.laplace_batch_size, features.shape[0])
        with torch.no_grad():
            for start in range(0, features.shape[0], bs):
                end = min(start + bs, features.shape[0])
                xb = features[start:end]
                ob = objective_idx[start:end]
                gb = signed_gaps[start:end].to(xb.dtype)
                a_i = model.a(xb).gather(1, ob.long().view(-1, 1)).squeeze(1)
                t = gb * torch.exp(a_i)
                # Hessian wrt a of negative log-likelihood = - d^2 log sigma(t)/d a^2.
                h_a = -t * torch.sigmoid(-t) * (1.0 - t * torch.sigmoid(t))
                psi2 = cls._design_diag_terms(model, xb, ob)
                precision = precision + (h_a.unsqueeze(1) * psi2).sum(dim=0)

        precision = precision + prior.laplace_damping
        precision = precision.clamp_min(prior.min_precision)
        if precision.numel() != P:
            raise RuntimeError("internal precision dimension mismatch")
        return precision.detach()

    def _standardize_for_inference(self, features: torch.Tensor, device: torch.device) -> torch.Tensor:
        mean = self.feature_mean.to(device=device, dtype=features.dtype)
        std = self.feature_std.to(device=device, dtype=features.dtype)
        return (features - mean) / std.clamp_min(1e-6)

    def expected_alpha(
        self,
        features: torch.Tensor,
        base_w: Sequence[float] | torch.Tensor,
        num_samples: int = 0,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """Posterior expectation E_q[alpha(x,w;theta)].

        If num_samples <= 0, returns plug-in MAP alpha.  This is often the most
        stable option early in experiments.  If num_samples > 0, samples theta
        from the diagonal Laplace posterior and averages alpha.
        """
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = self.model.to(device)
        features = features.to(device=device, dtype=torch.float32)
        features = self._standardize_for_inference(features, device)
        base_w = torch.as_tensor(base_w, device=device, dtype=torch.float32)
        base_w = base_w / base_w.sum().clamp_min(1e-12)

        if num_samples <= 0:
            with torch.no_grad():
                return model.alpha(features, base_w)

        theta_map = model.flatten_parameters().detach().to(device)
        std = torch.rsqrt(self.precision_diag.to(device).clamp_min(self.prior.min_precision))
        acc = torch.zeros(features.shape[0], model.num_objectives, device=device)
        original = theta_map.clone()
        with torch.no_grad():
            for _ in range(num_samples):
                eps = torch.randn_like(theta_map)
                model.load_flat_parameters(theta_map + eps * std)
                acc += model.alpha(features, base_w)
            model.load_flat_parameters(original)
        return acc / float(num_samples)

    def save_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        state = LaplacePosteriorState(
            num_objectives=self.model.num_objectives,
            feature_dim=self.model.feature_dim,
            theta_map=self.model.flatten_parameters().detach().cpu().tolist(),
            precision_diag=self.precision_diag.detach().cpu().tolist(),
            feature_mean=self.feature_mean.detach().cpu().tolist(),
            feature_std=self.feature_std.detach().cpu().tolist(),
            prior=asdict(self.prior),
            feature_model_name=self.feature_model_name,
        )
        with open(os.path.join(output_dir, "posterior_calibrator.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "LaplacePosteriorCalibrator":
        if os.path.isdir(path):
            path = os.path.join(path, "posterior_calibrator.json")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        prior = CalibrationPriorConfig(**raw.get("prior", {}))
        model = PosteriorCalibrationModel(raw["num_objectives"], raw["feature_dim"])
        model.load_flat_parameters(torch.tensor(raw["theta_map"], dtype=torch.float32))
        return cls(
            model=model,
            precision_diag=torch.tensor(raw["precision_diag"], dtype=torch.float32),
            feature_mean=torch.tensor(raw["feature_mean"], dtype=torch.float32),
            feature_std=torch.tensor(raw["feature_std"], dtype=torch.float32),
            prior=prior,
            feature_model_name=raw.get("feature_model_name"),
        )


class FrozenCausalLMPromptFeatureExtractor:
    """Prompt feature extractor using frozen SFT/reference causal-LM hidden states.

    This is the self-contained alternative to an external sentence encoder. If
    the model is a PeftModel, disable_adapter=True extracts features from the
    adapter-free reference policy.
    """

    def __init__(self, model: nn.Module, tokenizer, max_length: int = 256, device: Optional[str] = None, pooling: str = "mean", disable_adapter: bool = True, prompt_template: Optional[str] = None):
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be one of {'mean', 'last'}")
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pooling = pooling
        self.disable_adapter = bool(disable_adapter)
        self.prompt_template = prompt_template
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

    def _adapter_disabled_context(self):
        from contextlib import nullcontext
        target = self.model.module if hasattr(self.model, "module") else self.model
        if self.disable_adapter and hasattr(target, "disable_adapter"):
            return target.disable_adapter()
        return nullcontext()

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], batch_size: int = 8, device: Optional[str] = None) -> torch.Tensor:
        out: List[torch.Tensor] = []
        target_device = torch.device(device) if device is not None else self.device
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, len(prompts), batch_size):
                texts = list(prompts[start:start + batch_size])
                if self.prompt_template is not None:
                    texts = [self.prompt_template.format(raw_prompt=t) for t in texts]
                toks = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with self._adapter_disabled_context():
                    outputs = self.model(
                        input_ids=toks["input_ids"],
                        attention_mask=toks["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
                if getattr(outputs, "hidden_states", None) is None:
                    raise RuntimeError("Model did not return hidden_states. Make sure output_hidden_states=True is supported.")
                hidden = outputs.hidden_states[-1]
                mask = toks["attention_mask"].to(hidden.dtype)
                if self.pooling == "mean":
                    pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                else:
                    last_idx = mask.long().sum(dim=1).clamp_min(1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx]
                out.append(pooled.detach().float().cpu())
        finally:
            if was_training:
                self.model.train()
        return torch.cat(out, dim=0).to(target_device)


class HFPromptFeatureExtractor:
    """Simple HF encoder/LM prompt feature extractor.

    For encoder models, this uses masked mean pooling over the last hidden state.
    For causal LMs, the same masked mean pooling is used.  The extractor is kept
    outside the Laplace calibrator so that training can reuse the same pretrained
    model without coupling it to the posterior JSON.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 256,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], batch_size: int = 32, device: Optional[str | torch.device] = None) -> torch.Tensor:
        out: List[torch.Tensor] = []
        target_device = torch.device(device) if device is not None else self.device
        for start in range(0, len(prompts), batch_size):
            texts = list(prompts[start:start + batch_size])
            toks = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**toks).last_hidden_state
            mask = toks["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            out.append(pooled.detach().float().cpu())
        return torch.cat(out, dim=0).to(target_device)
