"""Upload a completed checkpoint directory to the Hugging Face Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("repo_id", help="For example: SrulyRosenblat/human-chess-mlx")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    required = {
        "LICENSE",
        "NOTICE",
        "model.safetensors",
        "config.json",
        "vocab.json",
        "tokenizer_config.json",
        "chess_model.py",
        "legal_inference.py",
        "README.md",
    }
    missing = sorted(name for name in required if not (args.checkpoint / name).is_file())
    if missing:
        raise SystemExit(f"Checkpoint is not publishable; missing: {', '.join(missing)}")

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.checkpoint),
        ignore_patterns=[
            "optimizer.safetensors",
            "training_state.json",
            "checkpoints/**",
            "*.log",
            "snapshot_*.json",
        ],
    )
    print(f"Uploaded https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
