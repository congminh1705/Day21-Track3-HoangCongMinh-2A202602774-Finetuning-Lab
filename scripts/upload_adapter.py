"""B5: upload the prepared adapter after the owner chooses a public HF repo.

Authenticate with `hf auth login` or HF_TOKEN first; do not put credentials in code.
Usage: python scripts/upload_adapter.py account/lab21-qwen35-triage-vi
"""
import argparse
import pathlib


def main():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    args = parser.parse_args()
    validate_repo_id(args.repo_id)
    if "/" not in args.repo_id:
        parser.error("include the owner: account/repo")
    root = pathlib.Path(__file__).resolve().parents[1]
    folder = root / "adapters" / "correct"
    for name in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        if not (folder / name).exists():
            parser.error(f"missing {name}; finish training and write_report.py first")
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=folder,
                      commit_message="Upload measured Lab 21 LoRA adapter")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
