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
        warmup_samples: Number of zero-valued samples passed through the
            model during ``warm_up`` to trigger first-decode kernel and
            graph compilation.
    """

    gpu_model: str = "large-v3"
    cpu_model: str = "small"
    gpu_compute_type: str = "float16"
    cpu_compute_type: str = "int8"
    warmup_samples: int = 16_000

    def model_for(self, device: Device) -> str:
        """Return the appropriate model id for the given device."""
        return self.gpu_model if device == "cuda" else self.cpu_model

    def compute_type_for(self, device: Device) -> str:
        """Return the appropriate CTranslate2 compute type for the device."""
        return self.gpu_compute_type if device == "cuda" else self.cpu_compute_type


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Configuration for the bilingual retrieval-augmented knowledge base.

    Attributes:
        embedding_model: Sentence-Transformers model id used to embed
            both the document chunks and incoming questions. The default
            is multilingual and covers EN and HU on CPU comfortably.
        top_k: Number of chunks returned per query.
        chunk_min_chars: Paragraphs shorter than this are merged with
            their neighbours during chunking.
    """

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 3
    chunk_min_chars: int = 80


@dataclass(frozen=True)
class AnthropicConfig:
    """Configuration for the Anthropic Claude API client.

    Attributes:
        api_key: Anthropic API key. Empty string when unset; the
            dialogue manager checks this before making LLM calls so the
            rest of the pipeline can be exercised without a key.
        model: Claude model identifier used for chat completions.
        max_tokens: Maximum response length per turn.
    """

    api_key: str = ""
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 512


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
        anthropic: Claude API client settings.
        knowledge: Retrieval-augmented knowledge base settings.
        domain: Active business domain identifier (e.g. ``vet``).
        domains_dir: Directory containing the domain YAML and KB files.
        output_dir: Directory where appointment JSON files are written.
    """

    device: Device = field(default_factory=detect_device)
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    knowledge: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    domain: str = "vet"
    domains_dir: Path = Path("domains")
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

        anthropic_overrides: Dict[str, Any] = {}
        api_key = _nested_get(data, ("anthropic", "api_key"))
        if isinstance(api_key, str) and api_key.strip():
            anthropic_overrides["api_key"] = api_key.strip()
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key.strip())
        model_id = _nested_get(data, ("anthropic", "model"))
        if isinstance(model_id, str) and model_id.strip():
            anthropic_overrides["model"] = model_id.strip()
        max_tokens = _nested_get(data, ("anthropic", "max_tokens"))
        if isinstance(max_tokens, int) and max_tokens > 0:
            anthropic_overrides["max_tokens"] = max_tokens
        if anthropic_overrides:
            config = replace(
                config, anthropic=replace(config.anthropic, **anthropic_overrides)
            )

        kb_overrides: Dict[str, Any] = {}
        kb_model = _nested_get(data, ("knowledge", "embedding_model"))
        if isinstance(kb_model, str) and kb_model.strip():
            kb_overrides["embedding_model"] = kb_model.strip()
        kb_top_k = _nested_get(data, ("knowledge", "top_k"))
        if isinstance(kb_top_k, int) and kb_top_k > 0:
            kb_overrides["top_k"] = kb_top_k
        if kb_overrides:
            config = replace(config, knowledge=replace(config.knowledge, **kb_overrides))

        domain_name = _nested_get(data, ("domain",))
        if isinstance(domain_name, str) and domain_name.strip():
            config = replace(config, domain=domain_name.strip())

        domains_dir = _nested_get(data, ("domains", "dir"))
        if isinstance(domains_dir, str) and domains_dir:
            config = replace(config, domains_dir=Path(domains_dir))

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
