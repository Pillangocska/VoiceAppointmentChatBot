"""Unit tests for the domain registry."""

from pathlib import Path

import pytest

from voiceappointmentchatbot.domains import (
    Domain,
    SlotSpec,
    available_domains,
    load_all_domains,
    load_domain,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = REPO_ROOT / "domains"


def test_vet_domain_loads_with_expected_slots() -> None:
    """The bundled vet domain parses with the slots required by the assignment."""
    domain = load_domain("vet", DOMAINS_DIR)

    assert domain.name == "vet"
    assert domain.display_name("en") == "Veterinary clinic"
    assert domain.display_name("hu").startswith("Állat")
    assert "Paws & Whiskers" in domain.blurb("en")
    assert {"customer_name", "phone", "pet_name", "species", "complaint", "time"} <= set(
        domain.slot_names
    )
    assert domain.knowledge_base_path == DOMAINS_DIR / "vet.md"
    assert domain.knowledge_base_path.exists()


def test_hairdresser_domain_loads_with_expected_slots() -> None:
    """The bundled hairdresser domain parses with its smaller slot set."""
    domain = load_domain("hairdresser", DOMAINS_DIR)

    assert domain.name == "hairdresser"
    assert {"customer_name", "phone", "service", "time"} == set(domain.slot_names)
    assert domain.knowledge_base_path.exists()


def test_phone_slot_is_typed() -> None:
    """Phone slots carry the 'phone' type so the manager can confirm digits."""
    domain = load_domain("vet", DOMAINS_DIR)

    assert domain.slot("phone").type == "phone"
    assert domain.slot("customer_name").type is None


def test_slot_prompt_falls_back_to_english() -> None:
    """A slot prompt resolves missing languages to the English text."""
    spec = SlotSpec(name="x", prompts={"en": "the answer"})

    assert spec.prompt_for("hu") == "the answer"
    assert spec.prompt_for("en") == "the answer"


def test_load_all_domains_returns_both() -> None:
    """``load_all_domains`` discovers every YAML file in the directory."""
    domains = load_all_domains(DOMAINS_DIR)

    assert set(domains.keys()) == {"vet", "hairdresser"}
    for domain in domains.values():
        assert isinstance(domain, Domain)


def test_available_domains_lists_names() -> None:
    """``available_domains`` returns sorted domain identifiers."""
    assert available_domains(DOMAINS_DIR) == ["hairdresser", "vet"]


def test_unknown_domain_raises_file_not_found() -> None:
    """Loading a missing domain surfaces a clear error."""
    with pytest.raises(FileNotFoundError):
        load_domain("nonexistent", DOMAINS_DIR)


def test_missing_english_blurb_is_rejected(tmp_path: Path) -> None:
    """A domain that omits the English blurb fails validation."""
    yaml_path = tmp_path / "broken.yaml"
    yaml_path.write_text(
        "name: broken\n"
        "display_name:\n  en: Broken\n  hu: Hibás\n"
        "blurb:\n  hu: Csak magyar\n"
        "slots:\n"
        "  - name: foo\n    prompt:\n      en: the foo\n"
        "knowledge_base: broken.md\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must include English"):
        load_domain("broken", tmp_path)


def test_missing_slots_is_rejected(tmp_path: Path) -> None:
    """A domain with no slots fails validation."""
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text(
        "name: empty\n"
        "display_name:\n  en: Empty\n  hu: Üres\n"
        "blurb:\n  en: x\n  hu: y\n"
        "slots: []\n"
        "knowledge_base: empty.md\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no slots"):
        load_domain("empty", tmp_path)
