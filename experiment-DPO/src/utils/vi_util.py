"""
Utilities for Variational-Inference MOD (VI-MOD).

This file contains:
  1. LLM-based prompt feature extraction, phi(x)
  2. Sequence log-prob and signed margin computation
  3. Mean-field VI calibrator for a_i(x) = mu_a + u_i + b_i^T phi(x)
  4. VIMODFusionModel, a drop-in FusionModel-style decoder using posterior-averaged MOD weights

Expected calibration JSONL format for vi_mod.py:
  {"objective": 0, "prompt": "...", "y1": "...", "y2": "...", "z": 1}
where z=+1 means y1 preferred to y2 for that objective, and z=-1 means y2 preferred.
Alternatively, rows with {"chosen": "...", "rejected": "..."} are accepted and treated as z=+1.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

try:
    from .util_decode import FusionModel
except Exception:  # pragma: no cover
    FusionModel = None

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    projection_dim: int = 128
    normalize: bool = True
    seed: int = 42
    batch_size: int = 4
    max_length: int = 512
    pooling: str = "mean"  # "mean" or "last"


class LLMPromptFeatureExtractor:
    """Extract prompt features phi(x) from a frozen LLM.

    We use the reference/base LLM hidden states as prompt embeddings. To keep the
    calibrator small, hidden states are optionally projected with a fixed random
    projection matrix.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: FeatureConfig = FeatureConfig(),
        projector: Optional[Tensor] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.projector = projector
        self._hidden_size: Optional[int] = None

    @property
    def feature_dim(self) -> int:
        if self.config.projection_dim and self.config.projection_dim > 0:
            return self.config.projection_dim
        if self._hidden_size is None:
            hidden = getattr(self.model.config, "hidden_size", None)
            if hidden is None:
                raise ValueError("Cannot infer hidden_size before extracting features.")
            self._hidden_size = int(hidden)
        return self._hidden_size

    def _maybe_init_projector(self, hidden_size: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        proj_dim = int(self.config.projection_dim or 0)
        if proj_dim <= 0 or proj_dim == hidden_size:
            return None
        if self.projector is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.config.seed)
            # Gaussian random projection; scaling preserves norms in expectation.
            P = torch.randn(hidden_size, proj_dim, generator=gen, dtype=torch.float32) / math.sqrt(float(proj_dim))
            self.projector = P
        return self.projector.to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], device: Optional[torch.device] = None) -> Tensor:
        if len(prompts) == 0:
            return torch.empty(0, self.feature_dim)
        self.model.eval()
        model_device = device if device is not None else next(self.model.parameters()).device
        outputs: List[Tensor] = []

        for start in tqdm(
            range(0, len(prompts), self.config.batch_size),
            desc=f"[features] encode {len(prompts)} prompts",
            unit="batch",
            dynamic_ncols=True,
                          ):
            batch = list(prompts[start : start + self.config.batch_size])
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(model_device)

            # If using a PEFT model, prompt features should come from the reference/base model.
            if hasattr(self.model, "disable_adapter"):
                ctx = self.model.disable_adapter()
            else:
                ctx = _NullContext()
            with ctx:
                out = self.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
            h = out.hidden_states[-1]  # [B, T, H]
            self._hidden_size = h.shape[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)

            if self.config.pooling == "mean":
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            elif self.config.pooling == "last":
                lengths = enc["attention_mask"].sum(dim=1).clamp_min(1) - 1
                pooled = h[torch.arange(h.shape[0], device=h.device), lengths]
            else:
                raise ValueError(f"Unknown pooling={self.config.pooling}")

            P = self._maybe_init_projector(pooled.shape[-1], pooled.device, pooled.dtype)
            if P is not None:
                pooled = pooled @ P
            if self.config.normalize:
                pooled = F.normalize(pooled.float(), p=2, dim=-1).to(h.dtype)
            outputs.append(pooled.detach().float().cpu())
        return torch.cat(outputs, dim=0)

    def save_projector(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"config": asdict(self.config), "projector": self.projector.cpu() if self.projector is not None else None}
        torch.save(payload, path)

    @staticmethod
    def load_projector(path: str) -> Tuple[FeatureConfig, Optional[Tensor]]:
        payload = torch.load(path, map_location="cpu")
        return FeatureConfig(**payload["config"]), payload.get("projector", None)


class _NullContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False


# -----------------------------------------------------------------------------
# Log-prob and margin utilities
# -----------------------------------------------------------------------------

def apply_prompt_template(raw_prompt: str, template: Optional[str]) -> str:
    if template is None or template == "":
        return raw_prompt
    if "{raw_prompt}" in template:
        return template.format(raw_prompt=raw_prompt)
    return template + raw_prompt


@torch.no_grad()
def sequence_logprob(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    adapter_name: Optional[str] = None,
    use_reference: bool = False,
    batch_size: int = 1,
    max_length: int = 1024,
    length_normalize: bool = True,
) -> Tensor:
    """Compute log p(response | prompt) for each pair.

    If adapter_name is given, model.set_adapter(adapter_name) is used.
    If use_reference=True and the model supports disable_adapter(), adapters are disabled.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length.")
    device = next(model.parameters()).device
    model.eval()
    vals: List[Tensor] = []

    total_batches = math.ceil(len(prompts) / batch_size)

    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=total_batches,
        desc=f"[logprob] adapter={adapter_name if adapter_name else 'ref'} use_ref={use_reference}",
        unit="batch",
        dynamic_ncols=True,
        ):
        ps = list(prompts[start : start + batch_size])
        rs = list(responses[start : start + batch_size])
        full = [p + r for p, r in zip(ps, rs)]

        full_enc = tokenizer(full, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        prompt_enc = tokenizer(ps, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)

        if adapter_name is not None and hasattr(model, "set_adapter") and not use_reference:
            model.set_adapter(adapter_name)

        ctx = model.disable_adapter() if (use_reference and hasattr(model, "disable_adapter")) else _NullContext()
        with ctx:
            out = model(**full_enc, use_cache=False, return_dict=True)

        logits = out.logits[:, :-1, :]
        labels = full_enc["input_ids"][:, 1:]
        mask = full_enc["attention_mask"][:, 1:].bool()

        token_logp = F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        # Response tokens start at prompt_len. Since labels are shifted by one, token position j
        # predicts full input token j+1. We include j+1 >= prompt_len.
        pos = torch.arange(labels.shape[1], device=device).unsqueeze(0) + 1
        resp_mask = pos >= prompt_lens.unsqueeze(1)
        final_mask = mask & resp_mask
        denom = final_mask.sum(dim=1).clamp_min(1)
        seq_lp = (token_logp * final_mask.float()).sum(dim=1)
        if length_normalize:
            seq_lp = seq_lp / denom
        vals.append(seq_lp.detach().cpu())
    return torch.cat(vals, dim=0)


@torch.no_grad()
def compute_signed_margins_for_rows(
    model,
    tokenizer,
    rows: Sequence[Dict],
    objective_adapter_name: str,
    prompt_template: Optional[str] = None,
    batch_size: int = 1,
    max_length: int = 1024,
    length_normalize: bool = True,
) -> Tensor:
    """Compute Delta r = z * [(lp_i(y1)-lp_ref(y1)) - (lp_i(y2)-lp_ref(y2))]."""
    prompts: List[str] = []
    y1s: List[str] = []
    y2s: List[str] = []
    zs: List[float] = []
    for row in rows:
        prompt = row.get("prompt", row.get("raw_prompt", row.get("input", "")))
        prompt = apply_prompt_template(prompt, prompt_template)
        if "chosen" in row and "rejected" in row:
            y1, y2, z = row["chosen"], row["rejected"], 1.0
        else:
            y1 = row.get("y1", row.get("response1", row.get("answer1")))
            y2 = row.get("y2", row.get("response2", row.get("answer2")))
            z = float(row.get("z", row.get("label", 1)))
        if y1 is None or y2 is None:
            raise ValueError(f"Row missing y1/y2 or chosen/rejected: {row.keys()}")
        prompts.append(prompt)
        y1s.append(y1)
        y2s.append(y2)
        zs.append(z)

    lp_i_y1 = sequence_logprob(model, tokenizer, prompts, y1s, objective_adapter_name, False, batch_size, max_length, length_normalize)
    lp_i_y2 = sequence_logprob(model, tokenizer, prompts, y2s, objective_adapter_name, False, batch_size, max_length, length_normalize)
    lp_0_y1 = sequence_logprob(model, tokenizer, prompts, y1s, None, True, batch_size, max_length, length_normalize)
    lp_0_y2 = sequence_logprob(model, tokenizer, prompts, y2s, None, True, batch_size, max_length, length_normalize)

    d = (lp_i_y1 - lp_0_y1) - (lp_i_y2 - lp_0_y2)
    return torch.tensor(zs, dtype=torch.float32) * d.float()


def load_calibration_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_rows_by_objective(rows: Sequence[Dict], num_objectives: int) -> List[List[Dict]]:
    grouped: List[List[Dict]] = [[] for _ in range(num_objectives)]
    for row in rows:
        obj = int(row.get("objective", row.get("objective_id", 0)))
        if obj < 0 or obj >= num_objectives:
            raise ValueError(f"objective index {obj} out of range [0, {num_objectives})")
        grouped[obj].append(row)
    return grouped


# -----------------------------------------------------------------------------
# Mean-field VI calibrator
# -----------------------------------------------------------------------------

@dataclass
class VIPriorConfig:
    m0: float = 0.0
    s0: float = 1.0
    sigma_u: float = 0.5
    sigma_b: float = 0.1


class MeanFieldVICalibrator(nn.Module):
    """Mean-field VI posterior for a_i(x)=mu_a+u_i+b_i^T phi(x)."""

    def __init__(
        self,
        num_objectives: int,
        feature_dim: int,
        prior: VIPriorConfig = VIPriorConfig(),
        init_log_std: float = -3.0,
    ) -> None:
        super().__init__()
        self.num_objectives = int(num_objectives)
        self.feature_dim = int(feature_dim)
        self.prior = prior

        self.mu_mean = nn.Parameter(torch.tensor(0.0))
        self.mu_log_std = nn.Parameter(torch.tensor(float(init_log_std)))
        self.u_mean = nn.Parameter(torch.zeros(num_objectives))
        self.u_log_std = nn.Parameter(torch.full((num_objectives,), float(init_log_std)))
        if feature_dim > 0:
            self.b_mean = nn.Parameter(torch.zeros(num_objectives, feature_dim))
            self.b_log_std = nn.Parameter(torch.full((num_objectives, feature_dim), float(init_log_std)))
        else:
            self.register_parameter("b_mean", None)
            self.register_parameter("b_log_std", None)

    @staticmethod
    def _std(log_std: Tensor) -> Tensor:
        return F.softplus(log_std) + 1e-5

    @property
    def mu_std(self) -> Tensor:
        return self._std(self.mu_log_std)

    @property
    def u_std(self) -> Tensor:
        return self._std(self.u_log_std)

    @property
    def b_std(self) -> Optional[Tensor]:
        if self.b_log_std is None:
            return None
        return self._std(self.b_log_std)

    def sample_theta(self, num_samples: int = 1) -> Dict[str, Tensor]:
        device = self.mu_mean.device
        dtype = self.mu_mean.dtype
        mu = self.mu_mean + self.mu_std * torch.randn(num_samples, device=device, dtype=dtype)
        u = self.u_mean.unsqueeze(0) + self.u_std.unsqueeze(0) * torch.randn(num_samples, self.num_objectives, device=device, dtype=dtype)
        if self.feature_dim > 0:
            b = self.b_mean.unsqueeze(0) + self.b_std.unsqueeze(0) * torch.randn(num_samples, self.num_objectives, self.feature_dim, device=device, dtype=dtype)
        else:
            b = torch.empty(num_samples, self.num_objectives, 0, device=device, dtype=dtype)
        return {"mu": mu, "u": u, "b": b}

    def a_samples(self, phi: Tensor, num_samples: int = 1) -> Tensor:
        if phi.dim() != 2 or phi.shape[-1] != self.feature_dim:
            raise ValueError(f"phi must have shape [B,{self.feature_dim}], got {tuple(phi.shape)}")
        theta = self.sample_theta(num_samples)
        if self.feature_dim > 0:
            b_phi = torch.einsum("smd,bd->sbm", theta["b"], phi)
        else:
            b_phi = torch.zeros(num_samples, phi.shape[0], self.num_objectives, device=phi.device, dtype=phi.dtype)
        return theta["mu"][:, None, None] + theta["u"][:, None, :] + b_phi

    def a_moments(self, phi: Tensor) -> Tuple[Tensor, Tensor]:
        if phi.dim() != 2 or phi.shape[-1] != self.feature_dim:
            raise ValueError(f"phi must have shape [B,{self.feature_dim}], got {tuple(phi.shape)}")
        B = phi.shape[0]
        mean = self.mu_mean + self.u_mean.unsqueeze(0).expand(B, -1)
        var = self.mu_std.pow(2) + self.u_std.pow(2).unsqueeze(0).expand(B, -1)
        if self.feature_dim > 0:
            mean = mean + phi @ self.b_mean.T
            var = var + phi.pow(2) @ self.b_std.pow(2).T
        return mean, var

    def kl_to_prior(self) -> Tensor:
        device = self.mu_mean.device
        dtype = self.mu_mean.dtype
        total = torch.zeros((), device=device, dtype=dtype)

        s0 = torch.tensor(self.prior.s0, device=device, dtype=dtype)
        m0 = torch.tensor(self.prior.m0, device=device, dtype=dtype)
        total = total + torch.log(s0 / self.mu_std) + (self.mu_std.pow(2) + (self.mu_mean - m0).pow(2)) / (2 * s0.pow(2)) - 0.5

        sigma_u = torch.tensor(self.prior.sigma_u, device=device, dtype=dtype)
        total = total + torch.sum(torch.log(sigma_u / self.u_std) + (self.u_std.pow(2) + self.u_mean.pow(2)) / (2 * sigma_u.pow(2)) - 0.5)

        if self.feature_dim > 0:
            sigma_b = torch.tensor(self.prior.sigma_b, device=device, dtype=dtype)
            total = total + torch.sum(torch.log(sigma_b / self.b_std) + (self.b_std.pow(2) + self.b_mean.pow(2)) / (2 * sigma_b.pow(2)) - 0.5)
        return total

    def negative_elbo(self, features_by_obj: List[Tensor], margins_by_obj: List[Tensor], num_mc_samples: int = 1) -> Tensor:
        expected_nll = torch.zeros((), device=self.mu_mean.device, dtype=self.mu_mean.dtype)
        for i, (phi_i, margin_i) in enumerate(zip(features_by_obj, margins_by_obj)):
            phi_i = phi_i.to(self.mu_mean.device, dtype=self.mu_mean.dtype)
            margin_i = margin_i.to(self.mu_mean.device, dtype=self.mu_mean.dtype).view(-1)
            if phi_i.numel() == 0 and phi_i.shape[-1] != self.feature_dim:
                phi_i = phi_i.view(margin_i.shape[0], self.feature_dim)
            a_s = self.a_samples(phi_i, num_samples=num_mc_samples)[:, :, i]
            logits = torch.exp(torch.clamp(a_s, -20, 20)) * margin_i.unsqueeze(0)
            expected_nll = expected_nll + F.softplus(-logits).mean(dim=0).sum()
        return expected_nll + self.kl_to_prior()

    @torch.no_grad()
    def expected_objective_weights(self, user_weights: Union[List[float], Tensor], phi: Tensor, num_samples: int = 32, eps: float = 1e-8) -> Tensor:
        if not torch.is_tensor(user_weights):
            user_weights = torch.tensor(user_weights, dtype=self.mu_mean.dtype, device=self.mu_mean.device)
        else:
            user_weights = user_weights.to(self.mu_mean.device, dtype=self.mu_mean.dtype)
        phi = phi.to(self.mu_mean.device, dtype=self.mu_mean.dtype)
        a_s = self.a_samples(phi, num_samples=num_samples)  # [S,B,M]
        kappa = torch.exp(torch.clamp(a_s, -20, 20))
        raw = kappa * user_weights.view(1, 1, -1)
        w_s = raw / raw.sum(dim=-1, keepdim=True).clamp_min(eps)
        return w_s.mean(dim=0)  # [B,M]


def train_vi_calibrator_from_margins(
    features_by_obj: List[Tensor],
    margins_by_obj: List[Tensor],
    num_steps: int = 2000,
    lr: float = 1e-2,
    num_mc_samples: int = 1,
    prior: VIPriorConfig = VIPriorConfig(),
    device: str = "cuda",
    print_every: int = 100,
) -> MeanFieldVICalibrator:
    feature_dim = int(features_by_obj[0].shape[-1])
    model = MeanFieldVICalibrator(len(features_by_obj), feature_dim, prior=prior).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(num_steps):
        opt.zero_grad(set_to_none=True)
        loss = model.negative_elbo(features_by_obj, margins_by_obj, num_mc_samples=num_mc_samples)
        loss.backward()
        opt.step()
        if print_every and (step % print_every == 0 or step == num_steps - 1):
            print(f"[VI] step={step:05d} neg_elbo={float(loss):.4f}")
    return model


def save_vi_calibrator(model: MeanFieldVICalibrator, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "num_objectives": model.num_objectives,
        "feature_dim": model.feature_dim,
        "prior": asdict(model.prior),
    }
    torch.save(payload, path)


def load_vi_calibrator(path: str, map_location: Union[str, torch.device] = "cpu") -> MeanFieldVICalibrator:
    payload = torch.load(path, map_location=map_location)
    model = MeanFieldVICalibrator(payload["num_objectives"], payload["feature_dim"], VIPriorConfig(**payload["prior"]))
    model.load_state_dict(payload["state_dict"])
    return model


# -----------------------------------------------------------------------------
# VI-MOD FusionModel
# -----------------------------------------------------------------------------

class VIMODFusionModel(FusionModel):
    """FusionModel with VI posterior-averaged objective weights.

    This keeps the original MOD generation flow but replaces fixed weights by
    E_q[ normalize(w_i exp(a_i(x))) ].
    """

    def __init__(
        self,
        model,
        user_weights: Sequence[float],
        f_type: str,
        vi_calibrator: MeanFieldVICalibrator,
        rho: float = 1.0,
        num_vi_samples: int = 32,
    ) -> None:
        if FusionModel is None:
            raise ImportError("Could not import FusionModel from util_decode.py")
        super().__init__(model=model, weights=list(user_weights), f_type=f_type)
        self.user_weights = torch.tensor(list(user_weights), dtype=torch.float32)
        self.vi_calibrator = vi_calibrator
        self.rho = float(rho)
        self.num_vi_samples = int(num_vi_samples)
        self._vi_weights: Optional[Tensor] = None

    def __call__(self, past_key_values, **model_inputs):
        # Same as original FusionModel, but fixes the reference past_key_values index.
        outputs = [None] * self.num_models
        for idx in range(self.num_models - 1):
            self.model.set_adapter("model_" + str(idx))
            with torch.no_grad():
                pkv = past_key_values[idx] if past_key_values is not None else None
                outputs[idx] = self.model(past_key_values=pkv, **model_inputs)
        ref_idx = self.num_models - 1
        with self.model.disable_adapter():
            with torch.no_grad():
                pkv = past_key_values[ref_idx] if past_key_values is not None else None
                outputs[ref_idx] = self.model(past_key_values=pkv, **model_inputs)
        output = outputs[0]
        output.logits = [outputs[k].logits for k in range(self.num_models)]
        output.past_key_values = [outputs[k].past_key_values for k in range(self.num_models)]
        return output

    def set_prompt_features(self, prompt_features: Tensor) -> None:
        weights = self.vi_calibrator.expected_objective_weights(
            user_weights=self.user_weights,
            phi=prompt_features,
            num_samples=self.num_vi_samples,
        )
        self._vi_weights = weights.to(next(self.model.parameters()).device)

    def generate(self, *args, prompt_features: Optional[Tensor] = None, **kwargs):
        if prompt_features is None:
            if self.vi_calibrator.feature_dim == 0:
                # batch size inferred later is not available here; assume one prompt if missing.
                prompt_features = torch.empty(1, 0, device=next(self.model.parameters()).device)
            else:
                raise ValueError("prompt_features must be passed for prompt-dependent VI-MOD.")
        self.set_prompt_features(prompt_features)
        return super().generate(*args, **kwargs)

    def f_value(self, logp_list: List[Tensor]) -> Tensor:
        if self._vi_weights is None:
            raise RuntimeError("VI weights are not initialized. Call generate(..., prompt_features=...) first.")
        obj_logps = logp_list[: self.num_models - 1]
        ref_logp = logp_list[-1]
        W = self._vi_weights.to(ref_logp.device, dtype=ref_logp.dtype)  # [B,M]

        # logp tensors are either [B,V] or [B, num_beams*V]. Broadcast W over last dim.
        if self.f_type in {"reverse_kl", "jsd"}:
            score = torch.zeros_like(obj_logps[0])
            for i, lp in enumerate(obj_logps):
                score = score + W[:, i].unsqueeze(-1) * lp
            if self.rho < 1.0:
                score = self.rho * score + (1.0 - self.rho) * ref_logp
            return score

        # Keep original forward/alpha style but replace fixed weights with batch-wise VI weights.
        if self.f_type == "forward_kl":
            parts = []
            for i, lp in enumerate(obj_logps):
                wi = W[:, i].clamp_min(1e-12).unsqueeze(-1)
                parts.append(-lp + torch.log(wi))
            score = -torch.logsumexp(torch.stack(parts, dim=0), dim=0)
            if self.rho < 1.0:
                score = self.rho * score + (1.0 - self.rho) * ref_logp
            return score

        if "-divergence" in self.f_type:
            alpha = float(self.f_type.split("-")[0])
            parts = []
            for i, lp in enumerate(obj_logps):
                wi = W[:, i].clamp_min(1e-12).unsqueeze(-1)
                parts.append(-alpha * lp + torch.log(wi))
            score = -torch.logsumexp(torch.stack(parts, dim=0), dim=0)
            if self.rho < 1.0:
                score = self.rho * score + (1.0 - self.rho) * ref_logp
            return score
        raise NotImplementedError(f"Unsupported f_type={self.f_type}")