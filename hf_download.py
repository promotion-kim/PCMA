#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError


def format_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    mb = num_bytes / (1024 ** 2)

    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{mb:.2f} MB"


def count_files_and_size(folder: Path):
    files = []
    total_size = 0

    for path in folder.rglob("*"):
        if path.is_file():
            files.append(path)
            total_size += path.stat().st_size

    return len(files), total_size


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repo to a local directory."
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face model repo id, e.g. promotion/morlhf_saferlhf_h0.5_s0.5",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Local directory where the model will be saved.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch, tag, or commit hash to download. Default: main",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, uses HF_TOKEN env variable or cached login.",
    )

    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if files already exist.",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel download workers. Default: 8",
    )

    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Only download files matching these patterns, e.g. '*.json' '*.safetensors'",
    )

    parser.add_argument(
        "--ignore_patterns",
        nargs="*",
        default=None,
        help="Ignore files matching these patterns, e.g. 'optimizer.pt' 'scheduler.pt'",
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN")

    if token is not None:
        login(token=token)

    print("=" * 80)
    print(f"[HF repo id]  {args.repo_id}")
    print(f"[revision]    {args.revision}")
    print(f"[save dir]    {save_dir}")
    print(f"[workers]     {args.max_workers}")
    print("=" * 80)

    try:
        print("\nDownloading from Hugging Face Hub...")
        print("Progress bars will be shown below.\n")

        local_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
            local_dir=str(save_dir),
            token=token,
            force_download=args.force_download,
            max_workers=args.max_workers,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
        )

        num_files, total_size = count_files_and_size(save_dir)

        print("\nDone!")
        print(f"Downloaded path: {local_path}")
        print(f"Saved files: {num_files}")
        print(f"Total size: {format_size(total_size)}")

    except HfHubHTTPError as e:
        print("\nHugging Face Hub error occurred.")
        print(e)
        print(
            "\nCheck whether the repo_id is correct, "
            "whether the model is private, and whether your HF token has access."
        )
        raise

    except Exception as e:
        print("\nUnexpected error occurred.")
        print(e)
        raise


if __name__ == "__main__":
    main()