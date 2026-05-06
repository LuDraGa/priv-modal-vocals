"""File-backed reusable voice profile registry for Dia2.

Profiles live on the Modal Volume beside model weights so they can be reused by
all warm containers and survive deployments.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4


@dataclass
class VoiceProfile:
    """Persistent voice profile metadata."""

    id: str
    name: str
    reference_audio_path: str
    reference_transcript: str
    gender: Optional[str]
    accent: Optional[str]
    language: str
    style_tags: List[str]
    use_case: Optional[str]
    quality_rating: Optional[int]
    notes: Optional[str]
    consent_confirmed: bool
    created_at: str
    reference_duration_sec: Optional[float] = None
    reference_sample_rate: Optional[int] = None


class VoiceProfileStore:
    """Small JSON registry and audio file store for Dia2 voice profiles."""

    def __init__(self, root_path: str = "/models/dia2/profiles"):
        self.root_path = Path(root_path)
        self.audio_path = self.root_path / "audio"
        self.registry_path = self.root_path / "profiles.json"
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.audio_path.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> List[VoiceProfile]:
        """Return all profiles ordered by creation time."""
        registry = self._load_registry()
        profiles = [self._profile_from_dict(data) for data in registry.values()]
        return sorted(profiles, key=lambda profile: profile.created_at)

    def get_profile(self, profile_id: str) -> VoiceProfile:
        """Load one profile or raise KeyError."""
        registry = self._load_registry()
        if profile_id not in registry:
            raise KeyError(profile_id)
        return self._profile_from_dict(registry[profile_id])

    def create_profile(
        self,
        *,
        name: str,
        reference_audio: bytes,
        reference_transcript: str,
        gender: Optional[str],
        accent: Optional[str],
        language: str,
        style_tags: List[str],
        use_case: Optional[str],
        quality_rating: Optional[int],
        notes: Optional[str],
        consent_confirmed: bool,
        reference_duration_sec: Optional[float],
        reference_sample_rate: Optional[int],
    ) -> VoiceProfile:
        """Create and persist a profile and its reference audio."""
        profile_id = self._new_profile_id(name)
        reference_audio_path = self.audio_path / f"{profile_id}.wav"
        reference_audio_path.write_bytes(reference_audio)

        profile = VoiceProfile(
            id=profile_id,
            name=name.strip(),
            reference_audio_path=str(reference_audio_path),
            reference_transcript=reference_transcript.strip(),
            gender=self._clean_optional(gender),
            accent=self._clean_optional(accent),
            language=language.strip().lower() or "en",
            style_tags=style_tags,
            use_case=self._clean_optional(use_case),
            quality_rating=quality_rating,
            notes=self._clean_optional(notes),
            consent_confirmed=consent_confirmed,
            created_at=datetime.now(timezone.utc).isoformat(),
            reference_duration_sec=reference_duration_sec,
            reference_sample_rate=reference_sample_rate,
        )

        registry = self._load_registry()
        registry[profile.id] = asdict(profile)
        self._write_registry(registry)
        return profile

    def delete_profile(self, profile_id: str) -> None:
        """Delete profile metadata and stored reference audio."""
        registry = self._load_registry()
        if profile_id not in registry:
            raise KeyError(profile_id)

        data = registry.pop(profile_id)
        audio_file = data.get("reference_audio_path")
        if audio_file:
            try:
                Path(audio_file).unlink(missing_ok=True)
            except OSError:
                pass

        self._write_registry(registry)

    def _load_registry(self) -> Dict[str, dict]:
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text())

    def _write_registry(self, registry: Dict[str, dict]) -> None:
        self.registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))

    def _profile_from_dict(self, data: dict) -> VoiceProfile:
        return VoiceProfile(
            id=data["id"],
            name=data["name"],
            reference_audio_path=data["reference_audio_path"],
            reference_transcript=data["reference_transcript"],
            gender=data.get("gender"),
            accent=data.get("accent"),
            language=data.get("language", "en"),
            style_tags=list(data.get("style_tags", [])),
            use_case=data.get("use_case"),
            quality_rating=data.get("quality_rating"),
            notes=data.get("notes"),
            consent_confirmed=bool(data.get("consent_confirmed")),
            created_at=data["created_at"],
            reference_duration_sec=data.get("reference_duration_sec"),
            reference_sample_rate=data.get("reference_sample_rate"),
        )

    def _new_profile_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"
        return f"{slug}-{uuid4().hex[:8]}"

    def _clean_optional(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
