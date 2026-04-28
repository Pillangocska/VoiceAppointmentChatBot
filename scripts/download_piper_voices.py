"""Pre-download every Piper voice declared in :class:`PiperConfig`.

Optional one-time helper: ``PiperSpeaker`` already fetches missing voices
on first use, so running this script is only useful when you want to
populate ``models/piper/`` ahead of time (for example before a demo
without internet access).
"""

from urllib.request import urlretrieve

from voiceappointmentchatbot.config import PiperConfig


def main() -> None:
    """Download every voice file listed in the active Piper configuration."""
    config = PiperConfig()
    config.models_dir.mkdir(parents=True, exist_ok=True)

    for language, spec in config.voices.items():
        for filename, url in (
            (spec.model_file, spec.model_url),
            (spec.config_file, spec.config_url),
        ):
            target = config.models_dir / filename
            if target.exists():
                print(f"[skip] {language}: {filename}")
                continue
            print(f"[fetch] {language}: {filename}")
            urlretrieve(url, target)
    print("Done.")


if __name__ == "__main__":
    main()
