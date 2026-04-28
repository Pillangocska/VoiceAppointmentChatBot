"""Runtime configuration for the voice chatbot.

Centralises model choices, audio parameters, and device detection so the
rest of the pipeline reads from a single source of truth. Values default
to sensible built-ins and may be overridden by a local ``config.yaml``
that is excluded from version control (see ``config.yaml.example``).
"""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Literal, Mapping, Optional, Tuple
from pathlib import Path
import os

import yaml

from voiceappointmentchatbot import gpu_runtime  # noqa: F401  (side-effect import)

Device = Literal["cuda", "cpu"]
DEFAULT_CONFIG_PATH = Path("config.yaml")


def detect_device() -> Device:
    """Return ``cuda`` when a working GPU is available, ``cpu`` otherwise.

    Detection goes through ctranslate2 because faster-whisper drives the
    GPU directly via CTranslate2 rather than PyTorch. This keeps the
    dependency surface small in week one (no CUDA torch wheel needed).

    Returns:
        The device identifier accepted by faster-whisper.
    """
    try:
        import ctranslate2
    except ImportError:
        return "cpu"
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class AudioConfig:
    """Microphone capture parameters.

    Attributes:
        sample_rate: Sampling rate in Hz used for capture and ASR input.
        channels: Number of input channels (mono is sufficient for speech).
        dtype: NumPy dtype string for the captured PCM samples.
        block_size: Frames per audio callback block.
    """

    sample_rate: int = 16_000
    channels: int = 1
    dtype: str = "float32"
    block_size: int = 1024


@dataclass(frozen=True)
class WhisperConfig:
    """Selection rules for the faster-whisper model and runtime.

    Attributes:
        gpu_model: Whisper model id used when CUDA is available.
        cpu_model: Smaller model id used as the CPU fallback.
        gpu_compute_type: CTranslate2 compute type on GPU.
        cpu_compute_type: CTranslate2 compute type on CPU.
    """

    gpu_model: str = "large-v3"
    cpu_model: str = "small"
    gpu_compute_type: str = "float16"
    cpu_compute_type: str = "int8"

    def model_for(self, device: Device) -> str:
        """Return the appropriate model id for the given device."""
        return self.gpu_model if device == "cuda" else self.cpu_model

    def compute_type_for(self, device: Device) -> str:
        """Return the appropriate CTranslate2 compute type for the device."""
        return self.gpu_compute_type if device == "cuda" else self.cpu_compute_type


@dataclass(frozen=True)
class SentimentConfig:
    """Configuration for the multilingual text-based sentiment classifier.

    Attributes:
        model_name: Hugging Face model id supporting EN and HU.
    """

    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


_PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass(frozen=True)
class PiperVoiceSpec:
    """Filenames and download URLs for a single Piper voice.

    Attributes:
        model_file: ONNX weights filename inside the models directory.
        config_file: JSON config filename inside the models directory.
        model_url: Public URL the ONNX file can be fetched from.
        config_url: Public URL the JSON file can be fetched from.
    """

    model_file: str
    config_file: str
    model_url: str
    config_url: str


def _voice(model_file: str, config_file: str, hf_path: str) -> PiperVoiceSpec:
    """Build a ``PiperVoiceSpec`` rooted at the rhasspy/piper-voices repo."""
    return PiperVoiceSpec(
        model_file=model_file,
        config_file=config_file,
        model_url=f"{_PIPER_HF_BASE}/{hf_path}/{model_file}",
        config_url=f"{_PIPER_HF_BASE}/{hf_path}/{config_file}",
    )


@dataclass(frozen=True)
class PiperConfig:
    """Locations and download URLs for the Piper voice models.

    Attributes:
        models_dir: Directory containing downloaded ``.onnx`` voice files.
        voices: Mapping from language code to its voice specification.
    """

    models_dir: Path = Path("models/piper")
    voices: Dict[str, PiperVoiceSpec] = field(
        default_factory=lambda: {
            "en": _voice(
                "en_US-lessac-medium.onnx",
                "en_US-lessac-medium.onnx.json",
                "en/en_US/lessac/medium",
            ),
            "hu": _voice(
                "hu_HU-anna-medium.onnx",
                "hu_HU-anna-medium.onnx.json",
                "hu/hu_HU/anna/medium",
            ),
        }
    )

    def voice_for(self, language: str) -> Tuple[Path, Path]:
        """Return absolute paths to the model and config for ``language``.

        Args:
            language: ISO 639-1 code (``en`` or ``hu``).

        Returns:
            Tuple of (model path, config path).

        Raises:
            KeyError: If no voice is registered for ``language``.
        """
        spec = self.voices[language]
        return self.models_dir / spec.model_file, self.models_dir / spec.config_file


@dataclass(frozen=True)
class AppConfig:
    """Aggregate runtime configuration.

    Attributes:
        device: Detected compute device.
        audio: Microphone capture settings.
        whisper: ASR model settings.
        sentiment: Sentiment classifier settings.
        piper: TTS voice settings.
        output_dir: Directory where appointment JSON files are written.
    """

    device: Device = field(default_factory=detect_device)
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    output_dir: Path = Path("output")

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "AppConfig":
        """Build an ``AppConfig`` from defaults overlaid with ``path``.

        When ``path`` exists, its YAML contents override matching default
        values. The Hugging Face token, when set, is exported into the
        environment as ``HF_TOKEN`` so downstream libraries pick it up.

        Args:
            path: Location of the optional YAML override file.

        Returns:
            The merged configuration.
        """
        config = cls()
        if not path.exists():
            return config

        with path.open("r", encoding="utf-8") as handle:
            data: Mapping[str, Any] = yaml.safe_load(handle) or {}

        token = _nested_get(data, ("huggingface", "token"))
        if isinstance(token, str) and token.strip():
            os.environ.setdefault("HF_TOKEN", token.strip())
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token.strip())

        device = _resolve_device(_nested_get(data, ("device", "override")))
        if device is not None:
            config = replace(config, device=device)

        whisper_overrides: Dict[str, Any] = {}
        gpu_model = _nested_get(data, ("whisper", "gpu_model"))
        cpu_model = _nested_get(data, ("whisper", "cpu_model"))
        if isinstance(gpu_model, str) and gpu_model:
            whisper_overrides["gpu_model"] = gpu_model
        if isinstance(cpu_model, str) and cpu_model:
            whisper_overrides["cpu_model"] = cpu_model
        if whisper_overrides:
            config = replace(config, whisper=replace(config.whisper, **whisper_overrides))

        sample_rate = _nested_get(data, ("audio", "sample_rate"))
        if isinstance(sample_rate, int) and sample_rate > 0:
            config = replace(config, audio=replace(config.audio, sample_rate=sample_rate))

        out_dir = _nested_get(data, ("output", "dir"))
        if isinstance(out_dir, str) and out_dir:
            config = replace(config, output_dir=Path(out_dir))

        return config


def _nested_get(data: Mapping[str, Any], keys: Tuple[str, ...]) -> Any:
    """Return ``data[keys[0]][keys[1]]...`` or ``None`` if any key is missing."""
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _resolve_device(value: Any) -> Optional[Device]:
    """Map a config string to a ``Device`` value, ``None`` if unrecognised."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value == "auto":
        return detect_device()
    if value in ("cuda", "cpu"):
        return value  # type: ignore[return-value]
    return None
