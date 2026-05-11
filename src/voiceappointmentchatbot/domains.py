"""Domain registry for the booking chatbot.

Each business domain (vet, hairdresser, ...) lives in a YAML file under
``domains/`` and declares its slot list, a short bilingual blurb that
seeds the system prompt, and the markdown file that holds its knowledge
base. Adding a new domain is a configuration change rather than a code
change.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple
from pathlib import Path

import yaml


DEFAULT_DOMAINS_DIR = Path("domains")
SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "hu")


@dataclass(frozen=True)
class SlotSpec:
    """Specification for one slot the dialogue must fill.

    Attributes:
        name: Stable identifier used as the JSON key and tool argument.
        prompts: Mapping from language code to a short noun phrase that
            describes the slot, used inside generated questions.
        type: Optional semantic type, currently ``"phone"`` or ``None``.
            Slots typed as ``"phone"`` go through digit-readback
            confirmation before being marked complete.
    """

    name: str
    prompts: Mapping[str, str]
    type: Optional[str] = None

    def prompt_for(self, language: str) -> str:
        """Return the noun-phrase prompt in ``language`` with EN fallback."""
        return self.prompts.get(language) or self.prompts["en"]


@dataclass(frozen=True)
class Domain:
    """A single business domain the chatbot can run as.

    Attributes:
        name: Stable identifier (e.g. ``vet`` or ``hairdresser``).
        display_names: Human-readable names per language.
        blurbs: Short paragraph per language inserted into the system
            prompt to set the assistant persona.
        slots: Ordered slot specifications.
        knowledge_base_path: Absolute path to the markdown knowledge
            base file for this domain.
        asr_prompts: Per-language ``initial_prompt`` strings forwarded
            to faster-whisper to bias decoding toward domain vocabulary
            (pet names, "foglalni", "rendelő", ...). Empty mapping by
            default; falls back to no biasing when a language is missing.
    """

    name: str
    display_names: Mapping[str, str]
    blurbs: Mapping[str, str]
    slots: Tuple[SlotSpec, ...]
    knowledge_base_path: Path
    asr_prompts: Mapping[str, str] = field(default_factory=dict)

    def display_name(self, language: str) -> str:
        """Return the localised display name with EN fallback."""
        return self.display_names.get(language) or self.display_names["en"]

    def blurb(self, language: str) -> str:
        """Return the localised blurb with EN fallback."""
        return self.blurbs.get(language) or self.blurbs["en"]

    def slot(self, name: str) -> SlotSpec:
        """Return the slot specification with the given ``name``.

        Args:
            name: Slot identifier.

        Returns:
            Matching :class:`SlotSpec`.

        Raises:
            KeyError: If no slot with that name is declared.
        """
        for spec in self.slots:
            if spec.name == name:
                return spec
        raise KeyError(f"unknown slot: {name!r}")

    @property
    def slot_names(self) -> Tuple[str, ...]:
        """Tuple of slot identifiers in declaration order."""
        return tuple(slot.name for slot in self.slots)


def load_domain(name: str, domains_dir: Path = DEFAULT_DOMAINS_DIR) -> Domain:
    """Load a single domain definition from ``domains_dir``.

    Args:
        name: Domain identifier; the loader reads ``<name>.yaml``.
        domains_dir: Directory containing the YAML and markdown files.

    Returns:
        Parsed :class:`Domain`.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the file is structurally invalid.
    """
    yaml_path = domains_dir / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"domain definition not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _parse_domain(data, domains_dir)


def load_all_domains(domains_dir: Path = DEFAULT_DOMAINS_DIR) -> Dict[str, Domain]:
    """Load every ``*.yaml`` domain file under ``domains_dir``.

    Args:
        domains_dir: Directory to scan.

    Returns:
        Mapping from domain name to :class:`Domain`.
    """
    if not domains_dir.is_dir():
        return {}
    domains: Dict[str, Domain] = {}
    for yaml_path in sorted(domains_dir.glob("*.yaml")):
        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        domain = _parse_domain(data, domains_dir)
        domains[domain.name] = domain
    return domains


def _parse_domain(data: Mapping[str, object], domains_dir: Path) -> Domain:
    """Validate and convert a parsed YAML mapping into a :class:`Domain`."""
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("domain file is missing a 'name' string")

    display_names = _require_lang_map(data.get("display_name"), "display_name", name)
    blurbs = _require_lang_map(data.get("blurb"), "blurb", name)

    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError(f"domain {name!r} has no slots")
    slots = tuple(_parse_slot(item, name) for item in raw_slots)

    kb_value = data.get("knowledge_base")
    if not isinstance(kb_value, str) or not kb_value:
        raise ValueError(f"domain {name!r} is missing 'knowledge_base'")

    asr_prompts = _parse_asr_prompts(data.get("asr_prompts"), name)

    return Domain(
        name=name,
        display_names=display_names,
        blurbs={lang: text.strip() for lang, text in blurbs.items()},
        slots=slots,
        knowledge_base_path=domains_dir / kb_value,
        asr_prompts=asr_prompts,
    )


def _parse_asr_prompts(value: object, domain_name: str) -> Dict[str, str]:
    """Validate the optional ``asr_prompts`` mapping.

    Args:
        value: Raw YAML value (typically a mapping or ``None``).
        domain_name: Domain identifier, used in error messages.

    Returns:
        Mapping from supported language code to prompt string. Empty
        mapping when ``value`` is missing.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"domain {domain_name!r}: asr_prompts must be a mapping")
    result: Dict[str, str] = {}
    for lang, text in value.items():
        if not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"domain {domain_name!r}: asr_prompts has unsupported "
                f"language {lang!r}"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"domain {domain_name!r}: asr_prompts[{lang!r}] must be "
                f"a non-empty string"
            )
        result[lang] = text.strip()
    return result


def _parse_slot(item: object, domain_name: str) -> SlotSpec:
    """Convert a single slot mapping from YAML into a :class:`SlotSpec`."""
    if not isinstance(item, Mapping):
        raise ValueError(f"domain {domain_name!r}: slot entry is not a mapping")
    slot_name = item.get("name")
    if not isinstance(slot_name, str) or not slot_name:
        raise ValueError(f"domain {domain_name!r}: slot is missing 'name'")
    prompts = _require_lang_map(item.get("prompt"), f"slot {slot_name!r}.prompt", domain_name)
    slot_type = item.get("type")
    if slot_type is not None and not isinstance(slot_type, str):
        raise ValueError(f"domain {domain_name!r}: slot {slot_name!r} type must be a string")
    return SlotSpec(name=slot_name, prompts=prompts, type=slot_type)


def _require_lang_map(value: object, label: str, domain_name: str) -> Dict[str, str]:
    """Validate a per-language string mapping and ensure English is present."""
    if not isinstance(value, Mapping):
        raise ValueError(f"domain {domain_name!r}: {label} must be a mapping")
    result: Dict[str, str] = {}
    for lang, text in value.items():
        if not isinstance(lang, str) or lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"domain {domain_name!r}: {label} has unsupported language {lang!r}"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"domain {domain_name!r}: {label}[{lang!r}] must be a non-empty string")
        result[lang] = text
    if "en" not in result:
        raise ValueError(f"domain {domain_name!r}: {label} must include English ('en')")
    return result


def available_domains(domains_dir: Path = DEFAULT_DOMAINS_DIR) -> List[str]:
    """Return the sorted list of domain names found under ``domains_dir``."""
    if not domains_dir.is_dir():
        return []
    return sorted(path.stem for path in domains_dir.glob("*.yaml"))
