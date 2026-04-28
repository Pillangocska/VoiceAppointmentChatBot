"""Pre-download Hugging Face models with a visible per-file progress bar.

Running this once after ``uv sync`` populates the local cache for the
faster-whisper Whisper model and the multilingual sentiment classifier,
so the first ``vetbot`` run does not appear to hang while large files
download in the background.

Files are fetched sequentially with ``hf_hub_download`` so the byte-level
``tqdm`` bar from ``huggingface_hub`` is visible per file. The aggregate
``snapshot_download`` API used previously hid per-file progress when it
parallelised downloads, which made multi-gigabyte fetches look frozen.
"""

from huggingface_hub import HfApi, hf_hub_download
import os
import sys

from voiceappointmentchatbot.config import AppConfig

WHISPER_REPO_TEMPLATE = "Systran/faster-whisper-{model}"


def _download_repo(repo_id: str) -> None:
    """Fetch every file in ``repo_id`` sequentially, with progress bars."""
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    print(f"[prefetch] {repo_id}: {len(files)} files")
    sys.stdout.flush()
    for index, filename in enumerate(files, start=1):
        print(f"[prefetch] ({index}/{len(files)}) {filename}")
        sys.stdout.flush()
        hf_hub_download(repo_id=repo_id, filename=filename)


def main() -> None:
    """Download Whisper and sentiment models for the active config."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    config = AppConfig.load()
    whisper_model = config.whisper.model_for(config.device)
    whisper_repo = WHISPER_REPO_TEMPLATE.format(model=whisper_model)

    print(f"[prefetch] device={config.device}")
    sys.stdout.flush()
    _download_repo(whisper_repo)
    _download_repo(config.sentiment.model_name)
    print("[prefetch] done.")


if __name__ == "__main__":
    main()
