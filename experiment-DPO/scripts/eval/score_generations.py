#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_REWARD_MODEL = "PKU-Alignment/beaver-7b-v1.0-reward"
DEFAULT_COST_MODEL = "PKU-Alignment/beaver-7b-v1.0-cost"
DEFAULT_PROMPT_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
DEFAULT_ARMORM_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

# Attribute order used by RLHFlow/ArmoRM-Llama3-8B-v0.1.
# Source: the model card demo code.
ARMORM_ATTRIBUTES = [
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
]

ARMORM_OBJECTIVE_ALIASES = {
    "helpful": "helpsteer-helpfulness",
    "helpfulness": "helpsteer-helpfulness",
    "uf-helpfulness": "ultrafeedback-helpfulness",
    "ultrafeedback-helpful": "ultrafeedback-helpfulness",
    "safe": "beavertails-is_safe",
    "safety": "beavertails-is_safe",
    "harmless": "beavertails-is_safe",
    "harmlessness": "beavertails-is_safe",
    "truthful": "ultrafeedback-truthfulness",
    "truthfulness": "ultrafeedback-truthfulness",
    "honest": "ultrafeedback-honesty",
    "honesty": "ultrafeedback-honesty",
    "overall": "ultrafeedback-overall_score",
    "overall_score": "ultrafeedback-overall_score",
    "instruction_following": "ultrafeedback-instruction_following",
    "instruction-following": "ultrafeedback-instruction_following",
}


@dataclass
class GenerationSet:
    name: str
    path: str
    metadata: Dict[str, Any]
    data: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Single-method interface. This is the recommended interface for evaluating
    # one generation file, including JSONL files from Reward Soup.
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Single generation file path (.json or .jsonl). Example: rs_output_h0.0_s1.0.jsonl",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Method name for the single generation file. Example: rs, modpo, bppmoa",
    )

    # Backward-compatible multi-input interface. If exactly two inputs are given,
    # the script also writes paired comparison output.
    parser.add_argument(
        "--input_jsons",
        type=str,
        default=None,
        help="Comma-separated generation json/jsonl paths. Example: modpo.json,pcmodpo.json",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Comma-separated names matching input_jsons. Example: modpo,pcmodpo",
    )

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--reward_type",
        type=str,
        choices=["beaver", "armorm"],
        default="beaver",
        help=(
            "Scoring backend. 'beaver' keeps the original PKU reward/cost models. "
            "'armorm' uses RLHFlow ArmoRM and writes armorm_preference_score plus selected objective scores."
        ),
    )
    parser.add_argument("--reward_model_name", type=str, default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--cost_model_name", type=str, default=DEFAULT_COST_MODEL)
    parser.add_argument("--armorm_model_name", type=str, default=DEFAULT_ARMORM_MODEL)
    parser.add_argument(
        "--armorm_helpful_objective",
        type=str,
        default="helpsteer-helpfulness",
        help=(
            "ArmoRM objective used as helpful_score when --reward_type armorm. "
            "Aliases such as helpful/helpfulness are allowed."
        ),
    )
    parser.add_argument(
        "--armorm_harmless_objective",
        type=str,
        default="beavertails-is_safe",
        help=(
            "ArmoRM objective used as harmless_score when --reward_type armorm. "
            "Aliases such as harmless/safety/safe are allowed."
        ),
    )
    parser.add_argument(
        "--armorm_save_all_objectives",
        action="store_true",
        help="When --reward_type armorm, save all 19 ArmoRM objective rewards in each scored row.",
    )

    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        help="Use 'none' for single GPU .to(cuda), or 'auto' for HF device_map='auto'.",
    )

    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument(
        "--preprocess_generations",
        action="store_true",
        help=(
            "Clean generated responses before scoring. This removes unrelated instruction "
            "leakage and trims obvious repeated/degenerate continuations."
        ),
    )
    parser.add_argument(
        "--no_preprocess_generations",
        dest="preprocess_generations",
        action="store_false",
        help="Disable generation preprocessing before scoring.",
    )
    parser.set_defaults(preprocess_generations=True)

    parser.add_argument(
        "--keep_original_generation",
        action="store_true",
        help="When preprocessing is enabled, keep the raw text in original_generation.",
    )
    parser.set_defaults(keep_original_generation=True)

    parser.add_argument(
        "--min_clean_words",
        type=int,
        default=4,
        help="If cleaning leaves fewer than this many words, fall back to the original generation.",
    )

    parser.add_argument("--save_scored_json", action="store_true")
    parser.add_argument("--no_save_scored_json", dest="save_scored_json", action="store_false")
    parser.set_defaults(save_scored_json=True)

    return parser.parse_args()


def resolve_input_args(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Resolve either --input_json/--name or --input_jsons/--names."""
    using_single = args.input_json is not None or args.name is not None
    using_multi = args.input_jsons is not None or args.names is not None

    if using_single and using_multi:
        raise ValueError("Use either --input_json/--name or --input_jsons/--names, not both.")

    if using_single:
        if not args.input_json:
            raise ValueError("--input_json is required when using the single-method interface.")
        name = args.name
        if not name:
            stem = os.path.basename(args.input_json)
            name = re.sub(r"\.(jsonl|json)$", "", stem, flags=re.IGNORECASE)
        return [args.input_json], [name]

    if using_multi:
        if not args.input_jsons or not args.names:
            raise ValueError("Both --input_jsons and --names are required for the multi-input interface.")
        return split_csv_arg(args.input_jsons), split_csv_arg(args.names)

    raise ValueError("Provide either --input_json/--name or --input_jsons/--names.")

def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def split_csv_arg(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def normalize_generation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common generation formats into the schema expected by the scorer.

    Supported examples:
      - MODPO/BPPMOA JSON rows: {raw_prompt, generation, ...}
      - Reward Soup JSONL rows: {raw_prompt, prompt, response, full_text, weight_helpful, weight_harmless}
      - Generic rows: {prompt, output/answer/text, ...}
    """
    out = dict(row)

    if out.get("generation") is None:
        for key in ["response", "output", "answer", "completion", "generated_text"]:
            if out.get(key) is not None:
                out["generation"] = out[key]
                break

    # If only a full prompt is available, keep it as model_input for Beaver-style scoring.
    if out.get("model_input") is None and out.get("prompt") is not None:
        out["model_input"] = out["prompt"]

    # Best-effort fallback: split full_text into model_input and generation if needed.
    if out.get("generation") is None and out.get("full_text") is not None:
        full_text = str(out["full_text"])
        marker = "ASSISTANT:"
        if marker in full_text:
            before, after = full_text.rsplit(marker, 1)
            out.setdefault("model_input", before + marker)
            out["generation"] = after.strip()
        else:
            out["generation"] = full_text

    if out.get("generation") is None:
        raise ValueError(
            "Could not find generation text in row. Expected one of: "
            "generation, response, output, answer, completion, generated_text, full_text. "
            f"Available keys: {sorted(out.keys())}"
        )

    return out


def load_generation_set(name: str, path: str, limit: int = -1) -> GenerationSet:
    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Expected rs_v2 format:
        #   Prompt and response
        #   BEGINNING OF CONVERSATION: USER: ... ASSISTANT: ...
        chunks = re.split(r"\n\s*Prompt and response\s*\n", text)
        data: List[Dict[str, Any]] = []
        for chunk in chunks:
            full_text = chunk.strip()
            if not full_text:
                continue

            row: Dict[str, Any] = {"full_text": full_text}

            m = re.search(r"ASSISTANT\s*:", full_text)
            if m is not None:
                split_idx = m.end()
                row["model_input"] = full_text[:split_idx].strip() + " "
                row["generation"] = full_text[split_idx:].strip()
                user_m = re.search(r"USER\s*:(.*?)ASSISTANT\s*:", full_text, flags=re.DOTALL)
                if user_m is not None:
                    row["raw_prompt"] = user_m.group(1).strip()
            else:
                row["generation"] = full_text

            data.append(normalize_generation_row(row))

        metadata = {"format": "txt"}
    elif suffix == ".jsonl":
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"Each JSONL line must be an object. Got {type(obj)} at line {line_no}")
                data.append(normalize_generation_row(obj))
        metadata = {"format": "jsonl"}
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
        raw_data = obj.get("data", obj if isinstance(obj, list) else []) if isinstance(obj, dict) else obj

        if not isinstance(raw_data, list):
            raise ValueError(f"Invalid generation JSON format: {path}")
        data = [normalize_generation_row(x) for x in raw_data]

    if limit is not None and limit > 0:
        data = data[:limit]

    return GenerationSet(name=name, path=path, metadata=metadata, data=data)

# These patterns are designed for generation artifacts observed in MODPO/CPC-MODPO outputs:
#   answer\n##\n11. Instruction: ...
#   answer\n### 10. Instruction: ...
#   answer\n## See also ...
# They are intentionally conservative: they only remove the tail from the first strong marker.
UNRELATED_TAIL_PATTERNS = [
    re.compile(r"\n\s*#{1,6}\s*\n\s*\d+\s*[\.)]?\s*Instruction\s*:", re.IGNORECASE),
    re.compile(r"\n\s*#{1,6}\s*\d+\s*[\.)]?\s*Instruction\s*:", re.IGNORECASE),
    re.compile(r"\n\s*\d+\s*[\.)]?\s*Instruction\s*:", re.IGNORECASE),
    re.compile(r"\n\s*(?:Input|Output|Evaluation)\s*:\s*", re.IGNORECASE),
    re.compile(r"\n\s*#{1,6}\s*See also\b", re.IGNORECASE),
]

# Long social-media-like hashtag tails are often degeneration rather than answer content.
HASHTAG_TAIL_RE = re.compile(r"(?:\s+#[-\w]+){4,}\s*$", re.IGNORECASE)

# Emoji loops often appear at the end of degenerate generations.
EMOJI_LOOP_RE = re.compile(r"(\s*[😆🤣😈🤡😂😭😍😊👍🔥💀]+\s*){6,}$")


def normalize_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_unrelated_tail(text: str) -> str:
    """Cut off obvious appended dataset/instruction artifacts."""
    cut = len(text)
    for pattern in UNRELATED_TAIL_PATTERNS:
        match = pattern.search(text)
        if match:
            cut = min(cut, match.start())
    return text[:cut].strip()


def strip_degenerate_tail(text: str) -> str:
    """Remove common non-semantic tails such as hashtag spam and emoji loops."""
    previous = None
    cleaned = text.strip()
    while previous != cleaned:
        previous = cleaned
        cleaned = HASHTAG_TAIL_RE.sub("", cleaned).strip()
        cleaned = EMOJI_LOOP_RE.sub("", cleaned).strip()
    return cleaned


def dedupe_repeated_paragraphs(text: str) -> str:
    """Remove exact repeated paragraphs while preserving first occurrences."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    seen = set()
    kept = []
    for paragraph in paragraphs:
        key = re.sub(r"\s+", " ", paragraph).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept).strip()


def dedupe_consecutive_lines(text: str) -> str:
    """Remove immediately repeated lines, which often occur in copied artifacts."""
    lines = [line.rstrip() for line in text.split("\n")]
    kept = []
    prev_key = None
    for line in lines:
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key and key == prev_key:
            continue
        kept.append(line)
        prev_key = key if key else None
    return "\n".join(kept).strip()


def truncate_repeated_sentence_loop(text: str, max_repeats: int = 2) -> str:
    """
    Truncate when the same sentence-like unit appears too many times.
    This targets pathological loops without trying to judge semantic quality.
    """
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    kept = []
    counts: Dict[str, int] = {}
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        key = re.sub(r"\s+", " ", piece).strip().lower()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > max_repeats:
            break
        kept.append(piece)
    return " ".join(kept).strip()


def truncate_repeated_ngram_tail(text: str, min_ngram: int = 5, max_ngram: int = 40, max_repeats: int = 3) -> str:
    """
    If the answer ends with a repeated word n-gram loop, keep only one occurrence.
    Example: "... #nasty #talk #to #me #insults ..." repeated many times.
    """
    tokens = text.split()
    if len(tokens) < min_ngram * max_repeats:
        return text.strip()

    max_ngram = min(max_ngram, len(tokens) // max_repeats)
    for n in range(min_ngram, max_ngram + 1):
        suffix = tokens[-n:]
        repeats = 1
        pos = len(tokens) - n
        while pos - n >= 0 and tokens[pos - n : pos] == suffix:
            repeats += 1
            pos -= n
        if repeats >= max_repeats:
            return " ".join(tokens[: pos + n]).strip()
    return text.strip()


def clean_generation_text(generation: str, min_clean_words: int = 4) -> Dict[str, Any]:
    """
    Clean a generated answer before reward/cost scoring.

    The goal is not to make unsafe answers safe. It only removes evaluation noise:
    unrelated instruction leakage, obvious copied benchmark artifacts, and repeated loops.
    """
    original = str(generation or "").strip()
    cleaned = normalize_space(original)
    cleaned = strip_unrelated_tail(cleaned)
    cleaned = strip_degenerate_tail(cleaned)
    cleaned = dedupe_repeated_paragraphs(cleaned)
    cleaned = dedupe_consecutive_lines(cleaned)
    cleaned = truncate_repeated_sentence_loop(cleaned)
    cleaned = truncate_repeated_ngram_tail(cleaned)
    cleaned = normalize_space(cleaned)

    # Avoid accidentally scoring an empty/nearly empty answer if a rule was too aggressive.
    if len(cleaned.split()) < min_clean_words and len(original.split()) >= min_clean_words:
        cleaned = normalize_space(original)

    return {
        "generation": cleaned,
        "changed": cleaned != normalize_space(original),
        "original_chars": len(original),
        "cleaned_chars": len(cleaned),
        "removed_chars": max(0, len(original) - len(cleaned)),
    }


def preprocess_generation_set(
    gs: GenerationSet,
    min_clean_words: int = 4,
    keep_original_generation: bool = True,
) -> Dict[str, Any]:
    """Apply generation cleaning in-place and return summary statistics."""
    changed = 0
    removed_chars = 0

    for row in gs.data:
        original_generation = str(row.get("generation", ""))
        result = clean_generation_text(original_generation, min_clean_words=min_clean_words)

        if keep_original_generation and result["changed"]:
            row["original_generation"] = original_generation

        row["generation"] = result["generation"]
        row["preprocess_changed"] = bool(result["changed"])
        row["preprocess_original_chars"] = int(result["original_chars"])
        row["preprocess_cleaned_chars"] = int(result["cleaned_chars"])
        row["preprocess_removed_chars"] = int(result["removed_chars"])

        changed += int(result["changed"])
        removed_chars += int(result["removed_chars"])

    return {
        "name": gs.name,
        "n": len(gs.data),
        "changed": changed,
        "changed_ratio": changed / max(1, len(gs.data)),
        "removed_chars": removed_chars,
        "avg_removed_chars": removed_chars / max(1, len(gs.data)),
    }


def build_score_input(row: Dict[str, Any], prompt_template: str) -> str:
    generation = str(row.get("generation", "")).strip()

    if row.get("model_input") is not None:
        model_input = str(row["model_input"])
    elif row.get("prompt") is not None:
        model_input = str(row["prompt"])
    elif row.get("raw_prompt") is not None:
        model_input = prompt_template.format(raw_prompt=str(row["raw_prompt"]))
    else:
        model_input = ""

    # Most prompts end with "ASSISTANT:"; add a space before the generated answer.
    if model_input and not model_input.endswith((" ", "\n")):
        model_input = model_input + " "

    return model_input + generation


def extract_prompt_response(row: Dict[str, Any], prompt_template: str) -> Tuple[str, str]:
    """Return a raw user prompt and assistant response for chat-template based reward models."""
    response = str(row.get("generation", "")).strip()

    if row.get("raw_prompt") is not None:
        prompt = str(row["raw_prompt"]).strip()
    else:
        model_input = ""
        if row.get("model_input") is not None:
            model_input = str(row["model_input"]).strip()
        elif row.get("prompt") is not None:
            model_input = str(row["prompt"]).strip()

        match = re.search(
            r"BEGINNING OF CONVERSATION:\s*USER:\s*(.*?)\s*ASSISTANT:\s*$",
            model_input,
            flags=re.IGNORECASE | re.DOTALL,
        )
        prompt = match.group(1).strip() if match else model_input

    return prompt, response

def build_armorm_chat_text(row: Dict[str, Any], tokenizer: AutoTokenizer, prompt_template: str) -> str:
    """Format a prompt/response pair with the tokenizer's chat template for ArmoRM."""
    prompt, response = extract_prompt_response(row, prompt_template=prompt_template)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def resolve_armorm_objective(name: str) -> Tuple[str, int]:
    """Resolve an ArmoRM objective name or common alias into (canonical_name, index)."""
    key = str(name).strip()
    if not key:
        raise ValueError("Empty ArmoRM objective name.")

    canonical = ARMORM_OBJECTIVE_ALIASES.get(key.lower(), key)
    if canonical not in ARMORM_ATTRIBUTES:
        allowed = ", ".join(ARMORM_ATTRIBUTES)
        aliases = ", ".join(sorted(ARMORM_OBJECTIVE_ALIASES))
        raise ValueError(
            f"Unknown ArmoRM objective: {name!r}.\n"
            f"Allowed objectives: {allowed}\n"
            f"Useful aliases: {aliases}"
        )
    return canonical, ARMORM_ATTRIBUTES.index(canonical)


def load_armorm_model(model_name: str, torch_dtype: torch.dtype, device_map: str, tokenizer: AutoTokenizer):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if device_map != "none":
        model_kwargs["device_map"] = device_map

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

    # The custom ArmoRM forward supports batching only when pad_token_id is set.
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    return model, device


@torch.no_grad()
def score_armorm_rows(
    rows: Sequence[Dict[str, Any]],
    model_name: str,
    batch_size: int,
    max_length: int,
    torch_dtype: torch.dtype,
    device_map: str,
    prompt_template: str,
    helpful_objective: str,
    harmless_objective: str,
    save_all_objectives: bool,
    desc: str,
) -> List[Dict[str, Any]]:
    """
    Score rows with RLHFlow/ArmoRM-Llama3-8B-v0.1.

    Returned fields intentionally preserve the old output schema:
      - helpful_score: selected ArmoRM helpful objective
      - harmless_score: selected ArmoRM safety/harmless objective
      - cost_score: -harmless_score, for compatibility with older analysis code
      - mip_0p5: 0.5 * helpful_score + 0.5 * harmless_score
      - armorm_preference_score: MoE-gated scalar preference score from ArmoRM
    """
    helpful_name, helpful_idx = resolve_armorm_objective(helpful_objective)
    harmless_name, harmless_idx = resolve_armorm_objective(harmless_objective)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, device = load_armorm_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        tokenizer=tokenizer,
    )

    scored: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(rows), batch_size), desc=desc, dynamic_ncols=True):
        batch_rows = list(rows[start : start + batch_size])
        batch_texts = [
            build_armorm_chat_text(row, tokenizer=tokenizer, prompt_template=prompt_template)
            for row in batch_rows
        ]

        toks = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}

        outputs = model(**toks)
        if not hasattr(outputs, "rewards") or not hasattr(outputs, "score"):
            raise RuntimeError(
                "ArmoRM output should contain `.rewards` and `.score`. "
                f"Got output type: {type(outputs)}"
            )

        rewards = outputs.rewards.detach().float().cpu()
        preference_scores = outputs.score.detach().float().view(-1).cpu()

        if rewards.ndim != 2 or rewards.shape[1] != len(ARMORM_ATTRIBUTES):
            raise RuntimeError(
                f"Unexpected ArmoRM rewards shape: {tuple(rewards.shape)}. "
                f"Expected (batch, {len(ARMORM_ATTRIBUTES)})."
            )

        for i in range(len(batch_rows)):
            helpful = float(rewards[i, helpful_idx].item())
            harmless = float(rewards[i, harmless_idx].item())
            item: Dict[str, Any] = {
                "helpful_score": helpful,
                "harmless_score": harmless,
                "cost_score": -harmless,
                "mip_0p5": 0.5 * helpful + 0.5 * harmless,
                "armorm_preference_score": float(preference_scores[i].item()),
                "armorm_helpful_objective": helpful_name,
                "armorm_harmless_objective": harmless_name,
            }
            if save_all_objectives:
                item["armorm_rewards"] = {
                    attr: float(rewards[i, j].item()) for j, attr in enumerate(ARMORM_ATTRIBUTES)
                }
            scored.append(item)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scored


def extract_scores(outputs: Any) -> torch.Tensor:
    """
    Supports Safe-RLHF AutoModelForScore and common HF sequence-classification outputs.
    Expected output shape: (B,)
    """
    value = None

    if hasattr(outputs, "end_scores"):
        value = outputs.end_scores
    elif hasattr(outputs, "scores"):
        value = outputs.scores
    elif hasattr(outputs, "logits"):
        value = outputs.logits
    elif isinstance(outputs, dict):
        for key in ["end_scores", "scores", "logits"]:
            if key in outputs:
                value = outputs[key]
                break
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        value = outputs[0]

    if value is None:
        raise RuntimeError(f"Could not extract score from model output type: {type(outputs)}")

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    # Typical shapes: (B, 1), (B,), or sometimes (B, T, 1).
    if value.ndim == 3:
        value = value[:, -1, :]
    if value.ndim == 2 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    elif value.ndim == 2 and value.shape[-1] > 1:
        # Fallback for unexpected multi-logit heads.
        value = value[:, 0]

    return value.float().view(-1)


def load_score_model(model_name: str, torch_dtype: torch.dtype, device_map: str):
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if device_map != "none":
        model_kwargs["device_map"] = device_map

    from safe_rlhf.models import AutoModelForScore

    try:
        model = AutoModelForScore.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("trust_remote_code", None)
        model = AutoModelForScore.from_pretrained(model_name, **model_kwargs)

    if device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device

    model.eval()
    return model, device


@torch.no_grad()
def score_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    torch_dtype: torch.dtype,
    device_map: str,
    desc: str,
) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Score models generally work fine with right padding; end_scores should use attention_mask.
    tokenizer.padding_side = "right"

    model, device = load_score_model(model_name, torch_dtype=torch_dtype, device_map=device_map)

    scores: List[float] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, dynamic_ncols=True):
        batch_texts = list(texts[start : start + batch_size])

        toks = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        toks = {k: v.to(device) for k, v in toks.items()}

        outputs = model(**toks)
        batch_scores = extract_scores(outputs)

        if batch_scores.numel() != len(batch_texts):
            raise RuntimeError(
                f"Score count mismatch: got {batch_scores.numel()}, expected {len(batch_texts)}"
            )

        scores.extend([float(x) for x in batch_scores.detach().cpu().tolist()])

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def std(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def stderr(xs: Sequence[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return std(xs) / math.sqrt(len(xs))


def summarize_rows(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    helpful = [float(r["helpful_score"]) for r in rows]
    cost = [float(r["cost_score"]) for r in rows]
    harmless = [float(r["harmless_score"]) for r in rows]
    mip_05 = [0.5 * h + 0.5 * s for h, s in zip(helpful, harmless)]

    summary = {
        "name": name,
        "n": len(rows),
        "helpful_mean": mean(helpful),
        "helpful_std": std(helpful),
        "helpful_stderr": stderr(helpful),
        "cost_mean": mean(cost),
        "cost_std": std(cost),
        "cost_stderr": stderr(cost),
        "harmless_mean": mean(harmless),
        "harmless_std": std(harmless),
        "harmless_stderr": stderr(harmless),
        "mip_0p5_mean": mean(mip_05),
        "mip_0p5_std": std(mip_05),
        "mip_0p5_stderr": stderr(mip_05),
    }

    if rows and "armorm_preference_score" in rows[0]:
        pref = [float(r["armorm_preference_score"]) for r in rows]
        summary.update(
            {
                "armorm_preference_mean": mean(pref),
                "armorm_preference_std": std(pref),
                "armorm_preference_stderr": stderr(pref),
                "armorm_helpful_objective": rows[0].get("armorm_helpful_objective"),
                "armorm_harmless_objective": rows[0].get("armorm_harmless_objective"),
            }
        )

    return summary


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_summary_csv(path: str, summaries: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    fieldnames = [
        "name",
        "n",
        "helpful_mean",
        "helpful_std",
        "helpful_stderr",
        "harmless_mean",
        "harmless_std",
        "harmless_stderr",
        "cost_mean",
        "cost_std",
        "cost_stderr",
        "mip_0p5_mean",
        "mip_0p5_std",
        "mip_0p5_stderr",
        "armorm_preference_mean",
        "armorm_preference_std",
        "armorm_preference_stderr",
        "armorm_helpful_objective",
        "armorm_harmless_objective",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})


def write_paired_comparison(
    path: str,
    a_name: str,
    a_rows: List[Dict[str, Any]],
    b_name: str,
    b_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = min(len(a_rows), len(b_rows))
    paired = []

    helpful_diffs = []
    harmless_diffs = []
    mip_diffs = []

    same_prompt_count = 0

    for i in range(n):
        ar = a_rows[i]
        br = b_rows[i]

        same_prompt = ar.get("raw_prompt") == br.get("raw_prompt")
        if same_prompt:
            same_prompt_count += 1

        a_helpful = float(ar["helpful_score"])
        b_helpful = float(br["helpful_score"])
        a_harmless = float(ar["harmless_score"])
        b_harmless = float(br["harmless_score"])

        helpful_diff = b_helpful - a_helpful
        harmless_diff = b_harmless - a_harmless

        a_mip = 0.5 * a_helpful + 0.5 * a_harmless
        b_mip = 0.5 * b_helpful + 0.5 * b_harmless
        mip_diff = b_mip - a_mip

        helpful_diffs.append(helpful_diff)
        harmless_diffs.append(harmless_diff)
        mip_diffs.append(mip_diff)

        paired.append(
            {
                "index": i,
                "same_prompt": same_prompt,
                "raw_prompt": ar.get("raw_prompt"),
                f"{a_name}_generation": ar.get("generation"),
                f"{b_name}_generation": br.get("generation"),
                f"{a_name}_helpful": a_helpful,
                f"{b_name}_helpful": b_helpful,
                "delta_helpful": helpful_diff,
                f"{a_name}_harmless": a_harmless,
                f"{b_name}_harmless": b_harmless,
                "delta_harmless": harmless_diff,
                f"{a_name}_mip_0p5": a_mip,
                f"{b_name}_mip_0p5": b_mip,
                "delta_mip_0p5": mip_diff,
            }
        )

    result = {
        "metadata": {
            "baseline_name": a_name,
            "candidate_name": b_name,
            "n": n,
            "same_prompt_count": same_prompt_count,
            "same_prompt_ratio": same_prompt_count / max(1, n),
            "delta_is_candidate_minus_baseline": True,
        },
        "summary": {
            "delta_helpful_mean": mean(helpful_diffs),
            "delta_helpful_stderr": stderr(helpful_diffs),
            "delta_harmless_mean": mean(harmless_diffs),
            "delta_harmless_stderr": stderr(harmless_diffs),
            "delta_mip_0p5_mean": mean(mip_diffs),
            "delta_mip_0p5_stderr": stderr(mip_diffs),
        },
        "data": paired,
    }

    write_json(path, result)
    return result


def main() -> None:
    args = parse_args()

    input_jsons, names = resolve_input_args(args)

    if len(input_jsons) != len(names):
        raise ValueError(
            f"--input_jsons and --names must have same length. "
            f"Got {len(input_jsons)} input_jsons and {len(names)} names."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = get_dtype(args.torch_dtype)

    gen_sets = [
        load_generation_set(name=name, path=path, limit=args.limit)
        for name, path in zip(names, input_jsons)
    ]

    for gs in gen_sets:
        print("=" * 100)
        print(f"[load] name={gs.name}")
        print(f"[load] path={gs.path}")
        print(f"[load] n={len(gs.data)}")

    preprocess_summaries: List[Dict[str, Any]] = []
    if args.preprocess_generations:
        print("=" * 100)
        print("[preprocess] cleaning unrelated tails and repeated/degenerate generations")
        for gs in gen_sets:
            prep_summary = preprocess_generation_set(
                gs,
                min_clean_words=args.min_clean_words,
                keep_original_generation=args.keep_original_generation,
            )
            preprocess_summaries.append(prep_summary)
            print(
                f"[preprocess] {gs.name}: "
                f"changed={prep_summary['changed']}/{prep_summary['n']} "
                f"({prep_summary['changed_ratio']:.2%}), "
                f"removed_chars={prep_summary['removed_chars']} "
                f"avg_removed={prep_summary['avg_removed_chars']:.1f}"
            )

        preprocess_path = os.path.join(args.output_dir, "preprocess_summary.json")
        write_json(preprocess_path, {"summaries": preprocess_summaries})
        print(f"[save] {preprocess_path}")

    # Build all texts.
    texts_by_name: Dict[str, List[str]] = {}
    for gs in gen_sets:
        texts_by_name[gs.name] = [
            build_score_input(row, prompt_template=args.prompt_template)
            for row in gs.data
        ]

    summaries = []
    scored_rows_by_name: Dict[str, List[Dict[str, Any]]] = {}

    if args.reward_type == "armorm":
        print("=" * 100)
        print(f"[score] ArmoRM model: {args.armorm_model_name}")
        print(f"[score] helpful objective: {args.armorm_helpful_objective}")
        print(f"[score] harmless objective: {args.armorm_harmless_objective}")

        for gs in gen_sets:
            armorm_scores = score_armorm_rows(
                gs.data,
                model_name=args.armorm_model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                torch_dtype=dtype,
                device_map=args.device_map,
                prompt_template=args.prompt_template,
                helpful_objective=args.armorm_helpful_objective,
                harmless_objective=args.armorm_harmless_objective,
                save_all_objectives=args.armorm_save_all_objectives,
                desc=f"armorm/{gs.name}",
            )

            if len(armorm_scores) != len(gs.data):
                raise RuntimeError(f"Score length mismatch for {gs.name}")

            scored_rows = []
            for row, score_fields in zip(gs.data, armorm_scores):
                new_row = dict(row)
                new_row.update(score_fields)
                scored_rows.append(new_row)

            scored_rows_by_name[gs.name] = scored_rows
            summaries.append(summarize_rows(gs.name, scored_rows))

            if args.save_scored_json:
                out_path = os.path.join(args.output_dir, f"{gs.name}_scored.json")
                write_json(
                    out_path,
                    {
                        "metadata": {
                            "name": gs.name,
                            "source_path": gs.path,
                            "source_metadata": gs.metadata,
                            "reward_type": args.reward_type,
                            "armorm_model_name": args.armorm_model_name,
                            "armorm_helpful_objective": summaries[-1].get("armorm_helpful_objective"),
                            "armorm_harmless_objective": summaries[-1].get("armorm_harmless_objective"),
                            "max_length": args.max_length,
                            "preprocess_generations": bool(args.preprocess_generations),
                            "preprocess_summary": next(
                                (s for s in preprocess_summaries if s.get("name") == gs.name),
                                None,
                            ),
                            "n": len(scored_rows),
                        },
                        "summary": summaries[-1],
                        "data": scored_rows,
                    },
                )
                print(f"[save] {out_path}")

    else:
        # Score helpful reward.
        print("=" * 100)
        print(f"[score] helpful reward model: {args.reward_model_name}")
        helpful_by_name = {}
        for gs in gen_sets:
            helpful_by_name[gs.name] = score_texts(
                texts_by_name[gs.name],
                model_name=args.reward_model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                torch_dtype=dtype,
                device_map=args.device_map,
                desc=f"helpful/{gs.name}",
            )

        # Score cost.
        print("=" * 100)
        print(f"[score] cost model: {args.cost_model_name}")
        cost_by_name = {}
        for gs in gen_sets:
            cost_by_name[gs.name] = score_texts(
                texts_by_name[gs.name],
                model_name=args.cost_model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                torch_dtype=dtype,
                device_map=args.device_map,
                desc=f"cost/{gs.name}",
            )

        for gs in gen_sets:
            helpful_scores = helpful_by_name[gs.name]
            cost_scores = cost_by_name[gs.name]

            if len(helpful_scores) != len(gs.data) or len(cost_scores) != len(gs.data):
                raise RuntimeError(f"Score length mismatch for {gs.name}")

            scored_rows = []
            for row, helpful, cost in zip(gs.data, helpful_scores, cost_scores):
                new_row = dict(row)
                new_row["helpful_score"] = float(helpful)
                new_row["cost_score"] = float(cost)
                new_row["harmless_score"] = float(-cost)
                new_row["mip_0p5"] = 0.5 * float(helpful) + 0.5 * float(-cost)
                scored_rows.append(new_row)

            scored_rows_by_name[gs.name] = scored_rows
            summaries.append(summarize_rows(gs.name, scored_rows))

            if args.save_scored_json:
                out_path = os.path.join(args.output_dir, f"{gs.name}_scored.json")
                write_json(
                    out_path,
                    {
                        "metadata": {
                            "name": gs.name,
                            "source_path": gs.path,
                            "source_metadata": gs.metadata,
                            "reward_type": args.reward_type,
                            "reward_model_name": args.reward_model_name,
                            "cost_model_name": args.cost_model_name,
                            "max_length": args.max_length,
                            "preprocess_generations": bool(args.preprocess_generations),
                            "preprocess_summary": next(
                                (s for s in preprocess_summaries if s.get("name") == gs.name),
                                None,
                            ),
                            "n": len(scored_rows),
                        },
                        "summary": summaries[-1],
                        "data": scored_rows,
                    },
                )
                print(f"[save] {out_path}")

    summary_json_path = os.path.join(args.output_dir, "summary.json")
    summary_csv_path = os.path.join(args.output_dir, "summary.csv")

    write_json(
        summary_json_path,
        {
            "reward_type": args.reward_type,
            "preprocess_generations": bool(args.preprocess_generations),
            "preprocess_summaries": preprocess_summaries,
            "summaries": summaries,
        },
    )
    write_summary_csv(summary_csv_path, summaries)

    print("=" * 100)
    print("[summary]")
    for s in summaries:
        pref_text = ""
        if "armorm_preference_mean" in s:
            pref_text = (
                f" | armorm_pref={s['armorm_preference_mean']:.4f} "
                f"± {s['armorm_preference_stderr']:.4f}"
            )
        print(
            f"{s['name']:>12s} | "
            f"n={s['n']} | "
            f"helpful={s['helpful_mean']:.4f} ± {s['helpful_stderr']:.4f} | "
            f"harmless={s['harmless_mean']:.4f} ± {s['harmless_stderr']:.4f} | "
            f"cost={s['cost_mean']:.4f} | "
            f"MIP@0.5={s['mip_0p5_mean']:.4f} ± {s['mip_0p5_stderr']:.4f}"
            f"{pref_text}"
        )

    print(f"[save] {summary_json_path}")
    print(f"[save] {summary_csv_path}")

    # If exactly two models, write paired comparison.
    if len(gen_sets) == 2:
        a = gen_sets[0]
        b = gen_sets[1]
        paired_path = os.path.join(args.output_dir, f"paired_{a.name}_vs_{b.name}.json")
        paired_result = write_paired_comparison(
            paired_path,
            a.name,
            scored_rows_by_name[a.name],
            b.name,
            scored_rows_by_name[b.name],
        )

        print("=" * 100)
        print(f"[paired] baseline={a.name}, candidate={b.name}")
        print(
            f"same_prompt_ratio={paired_result['metadata']['same_prompt_ratio']:.4f} "
            f"({paired_result['metadata']['same_prompt_count']}/{paired_result['metadata']['n']})"
        )
        print(
            f"delta helpful={paired_result['summary']['delta_helpful_mean']:.4f} "
            f"± {paired_result['summary']['delta_helpful_stderr']:.4f}"
        )
        print(
            f"delta harmless={paired_result['summary']['delta_harmless_mean']:.4f} "
            f"± {paired_result['summary']['delta_harmless_stderr']:.4f}"
        )
        print(
            f"delta MIP@0.5={paired_result['summary']['delta_mip_0p5_mean']:.4f} "
            f"± {paired_result['summary']['delta_mip_0p5_stderr']:.4f}"
        )
        print(f"[save] {paired_path}")


if __name__ == "__main__":
    main()
