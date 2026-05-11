# src/data/vi_calibration.py
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def get_response_pair(ex):
    """
    Robustly extract response_0 / response_1 from PKU-SafeRLHF-style examples.
    """
    if "response_0" in ex and "response_1" in ex:
        return ex["response_0"], ex["response_1"]

    if "responses" in ex and isinstance(ex["responses"], (list, tuple)) and len(ex["responses"]) >= 2:
        return ex["responses"][0], ex["responses"][1]

    raise KeyError(f"Cannot find response pair. Available keys: {list(ex.keys())}")


def get_prompt(ex):
    for key in ["prompt", "raw_prompt", "input", "query"]:
        if key in ex and ex[key] is not None:
            return ex[key]
    raise KeyError(f"Cannot find prompt field. Available keys: {list(ex.keys())}")


def get_id(ex, candidates):
    for key in candidates:
        if key in ex and ex[key] is not None:
            value = ex[key]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return value
            return int(value)
    return None


def build_pair_from_id(ex, objective):
    """
    objective:
      - better: use better_response_id
      - safer: use safer_response_id

    Assumes ids are 0 or 1.
    """
    r0, r1 = get_response_pair(ex)

    if objective == "better":
        chosen_id = get_id(ex, ["better_response_id", "better_id"])
    elif objective == "safer":
        chosen_id = get_id(ex, ["safer_response_id", "safer_id"])
    else:
        raise ValueError(f"Unknown objective: {objective}")

    if chosen_id is None:
        return None

    if chosen_id not in [0, 1]:
        return None

    chosen = r0 if chosen_id == 0 else r1
    rejected = r1 if chosen_id == 0 else r0
    return chosen, rejected


def build_safer_pair_from_safety_flags(ex):
    """
    Fallback when safer_response_id does not exist.
    Uses is_response_0_safe / is_response_1_safe.
    If both are same, returns None because no pairwise safer preference exists.
    """
    if "is_response_0_safe" not in ex or "is_response_1_safe" not in ex:
        return None

    safe0 = bool(ex["is_response_0_safe"])
    safe1 = bool(ex["is_response_1_safe"])

    if safe0 == safe1:
        return None

    r0, r1 = get_response_pair(ex)
    if safe0 and not safe1:
        return r0, r1
    if safe1 and not safe0:
        return r1, r0

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="PKU-Alignment/PKU-SafeRLHF-10K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="data/vi_calibration.jsonl")
    parser.add_argument("--max_examples_per_objective", type=int, default=None)
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}, split={args.split}")
    ds = load_dataset(args.dataset_name, split=args.split)

    print("Dataset size:", len(ds))
    print("Example keys:", list(ds[0].keys()))
    print("First example:", ds[0])

    if args.sanity_check:
        ds = ds.select(range(min(100, len(ds))))

    objective_specs = [
        ("better", 0),
        ("safer", 1),
    ]

    counts = {0: 0, 1: 0}
    skipped = {0: 0, 1: 0}

    with out_path.open("w", encoding="utf-8") as f:
        for objective_name, objective_id in objective_specs:
            for ex in ds:
                if args.max_examples_per_objective is not None:
                    if counts[objective_id] >= args.max_examples_per_objective:
                        break

                prompt = get_prompt(ex)

                pair = build_pair_from_id(ex, objective_name)

                if pair is None and objective_name == "safer":
                    pair = build_safer_pair_from_safety_flags(ex)

                if pair is None:
                    skipped[objective_id] += 1
                    continue

                chosen, rejected = pair

                item = {
                    "objective": objective_id,
                    "objective_name": objective_name,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[objective_id] += 1

    print(f"Saved to {out_path}")
    print("Counts:", counts)
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()