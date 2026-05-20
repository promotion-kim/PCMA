import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


OBJECTIVES = ["helpful", "harmless", "humor"]


def split_hh_conversation(text: str) -> Tuple[str, str]:
    """
    Anthropic HH format:
      \\n\\nHuman: ... \\n\\nAssistant: response

    Returns:
      prompt_hh: everything up to and including the last '\\n\\nAssistant:'
      response: assistant response after the last marker
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        raise ValueError("Cannot find Assistant marker in HH conversation.")

    prompt_hh = text[: idx + len(marker)]
    response = text[idx + len(marker):].strip()
    return prompt_hh, response


def hh_prompt_to_messages(prompt_hh: str) -> List[Dict[str, str]]:
    """
    Convert HH prompt prefix into chat messages.

    Example HH:
      \\n\\nHuman: hi\\n\\nAssistant: hello\\n\\nHuman: question\\n\\nAssistant:

    Output:
      [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "question"}
      ]
    """
    text = prompt_hh.strip()

    if text.endswith("Assistant:"):
        text = text[: -len("Assistant:")].strip()

    parts = text.split("\n\n")
    messages = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("Human:"):
            messages.append(
                {
                    "role": "user",
                    "content": part[len("Human:"):].strip(),
                }
            )
        elif part.startswith("Assistant:"):
            messages.append(
                {
                    "role": "assistant",
                    "content": part[len("Assistant:"):].strip(),
                }
            )

    return messages


def format_raw_prompt(
    prompt_hh: str,
    prompt_format: str,
    template_tokenizer: Optional[AutoTokenizer],
) -> str:
    """
    prompt_format:
      - hh:   save original HH prompt.
      - chat: apply tokenizer chat template and save model-specific prompt.

    If you train Llama-3/Qwen Instruct with --prompt_template "{raw_prompt}",
    prompt_format=chat is usually cleaner.
    """
    if prompt_format == "hh":
        return prompt_hh

    if prompt_format == "chat":
        if template_tokenizer is None:
            raise ValueError("--policy_model_for_template is required when --prompt_format chat")

        messages = hh_prompt_to_messages(prompt_hh)
        return template_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def collect_hh_pairs(
    dataset_name: str,
    split_name: str,
    prompt_format: str,
    template_tokenizer: Optional[AutoTokenizer],
    max_samples: int,
) -> List[Dict]:
    ds = load_dataset(dataset_name, split=split_name)

    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    rows = []
    skipped = 0

    for idx, ex in enumerate(tqdm(ds, desc=f"collect {split_name}")):
        try:
            prompt_chosen, chosen_resp = split_hh_conversation(ex["chosen"])
            prompt_rejected, rejected_resp = split_hh_conversation(ex["rejected"])
        except Exception:
            skipped += 1
            continue

        # In HH-RLHF, chosen/rejected should share the same prompt.
        # If not exactly equal, we still use the chosen-side prompt.
        prompt_hh = prompt_chosen

        try:
            raw_prompt = format_raw_prompt(
                prompt_hh=prompt_hh,
                prompt_format=prompt_format,
                template_tokenizer=template_tokenizer,
            )
        except Exception:
            skipped += 1
            continue

        if not chosen_resp or not rejected_resp:
            skipped += 1
            continue

        rows.append(
            {
                "source_idx": idx,
                "source_split": split_name,
                "prompt_hh": prompt_hh,
                "raw_prompt": raw_prompt,
                "original_chosen": chosen_resp,
                "original_rejected": rejected_resp,
                "raw_scores": {},
                "norm_scores": {},
            }
        )

    print(f"[info] split={split_name}, collected={len(rows)}, skipped={skipped}")
    return rows


def build_reward_texts(rows: List[Dict], objective: str, side: str) -> List[str]:
    """
    side is one of:
      - original_chosen
      - original_rejected

    helpful/harmless reward models get prompt + response.
    humor classifier gets response only.
    """
    texts = []

    for row in rows:
        response = row[side]

        if objective in ["helpful", "harmless"]:
            text = row["prompt_hh"] + " " + response
        elif objective == "humor":
            text = response
        else:
            raise ValueError(f"Unknown objective: {objective}")

        texts.append(text)

    return texts


@torch.no_grad()
def score_texts(
    model_name: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str,
    trust_remote_code: bool = False,
) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    model.to(device)
    model.eval()
    ensure_pad_token(tokenizer, model)

    scores = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"score {model_name}"):
        batch = texts[start : start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**inputs).logits

        if logits.shape[-1] == 1:
            batch_scores = logits[:, 0]
        else:
            # For binary classifiers, use the positive-class logit.
            batch_scores = logits[:, -1]

        scores.extend(batch_scores.detach().float().cpu().tolist())

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def score_objective_for_splits(
    objective: str,
    model_name: str,
    split_rows: Dict[str, List[Dict]],
    batch_size: int,
    max_length: int,
    device: str,
    trust_remote_code: bool,
):
    """
    Load one reward model at a time, score train/test, then unload.
    This avoids holding all reward models in GPU memory simultaneously.
    """
    all_texts = []
    metadata = []

    for split_name, rows in split_rows.items():
        for side in ["original_chosen", "original_rejected"]:
            texts = build_reward_texts(rows, objective=objective, side=side)
            all_texts.extend(texts)
            metadata.extend([(split_name, i, side) for i in range(len(rows))])

    scores = score_texts(
        model_name=model_name,
        texts=all_texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    assert len(scores) == len(metadata)

    for score, (split_name, idx, side) in zip(scores, metadata):
        row = split_rows[split_name][idx]

        if objective not in row["raw_scores"]:
            row["raw_scores"][objective] = {}

        row["raw_scores"][objective][side] = score


def compute_train_normalization_stats(train_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    stats = {}

    for obj in OBJECTIVES:
        values = []

        for row in train_rows:
            values.append(row["raw_scores"][obj]["original_chosen"])
            values.append(row["raw_scores"][obj]["original_rejected"])

        t = torch.tensor(values, dtype=torch.float32)
        mean = t.mean().item()
        std = t.std(unbiased=False).item()

        if std < 1e-8:
            std = 1.0

        stats[obj] = {
            "mean": mean,
            "std": std,
        }

    return stats


def apply_normalization(split_rows: Dict[str, List[Dict]], stats: Dict[str, Dict[str, float]]):
    for rows in split_rows.values():
        for row in rows:
            for obj in OBJECTIVES:
                row["norm_scores"][obj] = {}

                for side in ["original_chosen", "original_rejected"]:
                    raw = row["raw_scores"][obj][side]
                    mean = stats[obj]["mean"]
                    std = stats[obj]["std"]
                    row["norm_scores"][obj][side] = (raw - mean) / std


def build_objective_preference_rows(rows: List[Dict], objective: str, tau: float) -> List[Dict]:
    out = []

    for row in rows:
        sc = row["norm_scores"][objective]["original_chosen"]
        sr = row["norm_scores"][objective]["original_rejected"]
        gap = sc - sr

        if gap > tau:
            out.append(
                {
                    "raw_prompt": row["raw_prompt"],
                    "chosen": row["original_chosen"],
                    "rejected": row["original_rejected"],
                    "objective": objective,
                    "source_idx": row["source_idx"],
                    "source_split": row["source_split"],
                    "score_chosen": sc,
                    "score_rejected": sr,
                    "gap": gap,
                    "original_chosen_is_objective_chosen": True,
                }
            )
        elif gap < -tau:
            out.append(
                {
                    "raw_prompt": row["raw_prompt"],
                    "chosen": row["original_rejected"],
                    "rejected": row["original_chosen"],
                    "objective": objective,
                    "source_idx": row["source_idx"],
                    "source_split": row["source_split"],
                    "score_chosen": sr,
                    "score_rejected": sc,
                    "gap": -gap,
                    "original_chosen_is_objective_chosen": False,
                }
            )

    return out


def save_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def strip_debug_fields_for_scored_pool(rows: List[Dict]) -> List[Dict]:
    """
    Save scored pool for debugging / BPP-MOA reuse.
    Keep both raw and normalized objective scores.
    """
    out = []

    for row in rows:
        out.append(
            {
                "source_idx": row["source_idx"],
                "source_split": row["source_split"],
                "prompt_hh": row["prompt_hh"],
                "raw_prompt": row["raw_prompt"],
                "original_chosen": row["original_chosen"],
                "original_rejected": row["original_rejected"],
                "raw_scores": row["raw_scores"],
                "norm_scores": row["norm_scores"],
            }
        )

    return out


def print_summary(split_rows: Dict[str, List[Dict]], output_dir: str, tau: float):
    print("\n=== Dataset summary ===")
    print(f"output_dir = {output_dir}")
    print(f"tau        = {tau}")

    for split_name, rows in split_rows.items():
        print(f"\n[{split_name}] source pairs = {len(rows)}")

        for obj in OBJECTIVES:
            pref_rows = build_objective_preference_rows(rows, obj, tau=tau)
            gaps = [r["gap"] for r in pref_rows]
            flipped = [not r["original_chosen_is_objective_chosen"] for r in pref_rows]

            if len(gaps) == 0:
                print(f"  {obj:9s}: n=0")
                continue

            gap_t = torch.tensor(gaps, dtype=torch.float32)
            flip_rate = sum(flipped) / len(flipped)

            print(
                f"  {obj:9s}: n={len(pref_rows):7d}, "
                f"gap_mean={gap_t.mean().item():.4f}, "
                f"gap_p50={gap_t.median().item():.4f}, "
                f"flip_rate={flip_rate:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--prompt_format",
        type=str,
        default="hh",
        choices=["hh", "chat"],
        help="hh: save HH prompt. chat: apply policy tokenizer chat template.",
    )
    parser.add_argument(
        "--policy_model_for_template",
        type=str,
        default=None,
        help="Only tokenizer is loaded. Required if --prompt_format chat.",
    )

    parser.add_argument("--helpful_rm", type=str, default="Ray2333/gpt2-large-helpful-reward_model")
    parser.add_argument("--harmless_rm", type=str, default="Ray2333/gpt2-large-harmless-reward_model")
    parser.add_argument("--humor_rm", type=str, default="mohameddhiab/humor-no-humor")

    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument("--rm_max_length", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.0)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--trust_remote_code", action="store_true")

    # Default -1 means use all examples.
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)

    args = parser.parse_args()

    template_tokenizer = None
    if args.prompt_format == "chat":
        if args.policy_model_for_template is None:
            raise ValueError("--policy_model_for_template is required when --prompt_format chat")

        print(f"[info] loading tokenizer for chat template: {args.policy_model_for_template}")
        template_tokenizer = AutoTokenizer.from_pretrained(
            args.policy_model_for_template,
            trust_remote_code=True,
        )

    # HH-RLHF commonly has train/test. If a dataset has validation instead,
    # use validation as the held-out split.
    split_names = list(load_dataset(args.dataset_name).keys())
    if "train" not in split_names:
        raise ValueError(f"Dataset has no train split. Available splits: {split_names}")

    if "test" in split_names:
        heldout_split = "test"
    elif "validation" in split_names:
        heldout_split = "validation"
    else:
        raise ValueError(f"Dataset has neither test nor validation split. Available splits: {split_names}")

    print(f"[info] using train split: train")
    print(f"[info] using heldout split: {heldout_split}")

    split_rows = {
        "train": collect_hh_pairs(
            dataset_name=args.dataset_name,
            split_name="train",
            prompt_format=args.prompt_format,
            template_tokenizer=template_tokenizer,
            max_samples=args.max_train_samples,
        ),
        "test": collect_hh_pairs(
            dataset_name=args.dataset_name,
            split_name=heldout_split,
            prompt_format=args.prompt_format,
            template_tokenizer=template_tokenizer,
            max_samples=args.max_test_samples,
        ),
    }

    reward_model_names = {
        "helpful": args.helpful_rm,
        "harmless": args.harmless_rm,
        "humor": args.humor_rm,
    }

    for obj in OBJECTIVES:
        print(f"\n[info] scoring objective={obj}, model={reward_model_names[obj]}")
        score_objective_for_splits(
            objective=obj,
            model_name=reward_model_names[obj],
            split_rows=split_rows,
            batch_size=args.score_batch_size,
            max_length=args.rm_max_length,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )

    stats = compute_train_normalization_stats(split_rows["train"])
    apply_normalization(split_rows, stats)

    os.makedirs(args.output_dir, exist_ok=True)

    save_json(
        os.path.join(args.output_dir, "normalization_stats.json"),
        stats,
    )

    # Save scored pool.
    for split_name, rows in split_rows.items():
        save_jsonl(
            os.path.join(args.output_dir, f"scored_pool_{split_name}.jsonl"),
            strip_debug_fields_for_scored_pool(rows),
        )

    # Save objective-specific preference datasets.
    for obj in OBJECTIVES:
        obj_dir = os.path.join(args.output_dir, f"hh3-{obj}")
        os.makedirs(obj_dir, exist_ok=True)

        train_pref = build_objective_preference_rows(split_rows["train"], obj, tau=args.tau)
        test_pref = build_objective_preference_rows(split_rows["test"], obj, tau=args.tau)

        save_jsonl(os.path.join(obj_dir, "train.jsonl"), train_pref)
        save_jsonl(os.path.join(obj_dir, "test.jsonl"), test_pref)

        # For compatibility with existing DPO/MODPO code that calls split="validation".
        save_jsonl(os.path.join(obj_dir, "validation.jsonl"), test_pref)

    print_summary(split_rows, args.output_dir, args.tau)
    print("\n[done]")


if __name__ == "__main__":
    main()