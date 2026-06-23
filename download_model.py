from pathlib import Path

from modelscope import snapshot_download


MODEL_ID = "Qwen/Qwen3.6-27B"
TARGET_DIR = Path(__file__).resolve().parent / "models" / "Qwen3.6-27B"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        revision="master",
        local_dir=str(TARGET_DIR),
        max_workers=8,
    )
    print(f"Downloaded {MODEL_ID} to {TARGET_DIR}")


if __name__ == "__main__":
    main()
