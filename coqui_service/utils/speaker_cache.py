"""Speaker metadata caching with stale-while-revalidate pattern.

Implements a 10-day TTL cache for speaker metadata stored on Modal Volume.
Pattern: Return cached data if <10 days old, trigger async refresh if stale.
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog

logger = structlog.get_logger()


class SpeakerCache:
    """Manages speaker metadata caching on Modal Volume."""

    def __init__(self, volume_path: str = "/models/coqui"):
        self.volume_path = Path(volume_path)
        self.cache_file = self.volume_path / "speaker_metadata.json"
        self.ttl_days = 10
        self._refresh_lock = asyncio.Lock()

    def get_cache_metadata(self) -> Optional[Dict[str, Any]]:
        """Read speaker metadata from Volume cache.

        Returns:
            Cache data with speakers list and metadata, or None if not found
        """
        if not self.cache_file.exists():
            logger.info("speaker_cache.miss", reason="file_not_found")
            return None

        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)

            # Validate cache structure
            if not isinstance(data, dict) or "speakers" not in data:
                logger.warning("speaker_cache.invalid", reason="missing_speakers_key")
                return None

            return data

        except (json.JSONDecodeError, IOError) as e:
            logger.error("speaker_cache.read_error", error=str(e))
            return None

    def is_cache_stale(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cache is older than TTL.

        Args:
            cache_data: Cache metadata dict

        Returns:
            True if cache is stale (>10 days old)
        """
        last_updated_str = cache_data.get("last_updated")
        if not last_updated_str:
            return True

        try:
            last_updated = datetime.fromisoformat(last_updated_str)
            age = datetime.utcnow() - last_updated
            is_stale = age > timedelta(days=self.ttl_days)

            logger.info(
                "speaker_cache.age_check",
                age_days=age.days,
                is_stale=is_stale,
                ttl_days=self.ttl_days,
            )
            return is_stale

        except (ValueError, TypeError) as e:
            logger.warning("speaker_cache.invalid_timestamp", error=str(e))
            return True

    def write_cache(self, speakers: List[str], volume) -> None:
        """Write speaker metadata to Volume cache.

        Args:
            speakers: List of speaker names
            volume: Modal Volume instance (for commit)
        """
        cache_data = {
            "speakers": sorted(speakers),
            "count": len(speakers),
            "last_updated": datetime.utcnow().isoformat(),
        }

        try:
            # Ensure directory exists
            self.volume_path.mkdir(parents=True, exist_ok=True)

            # Write cache file
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            # Commit to Volume (persist changes)
            volume.commit()

            logger.info(
                "speaker_cache.written",
                count=len(speakers),
                cache_file=str(self.cache_file),
            )

        except IOError as e:
            logger.error("speaker_cache.write_error", error=str(e))
            raise

    async def get_speakers(
        self,
        tts_model,
        volume,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Get speaker list with stale-while-revalidate pattern.

        Args:
            tts_model: Loaded TTS model instance
            volume: Modal Volume instance
            force_refresh: Force synchronous cache refresh

        Returns:
            Dict with speakers, count, last_updated, cache_age_days
        """
        # Force refresh: synchronous rebuild
        if force_refresh:
            logger.info("speaker_cache.force_refresh")
            return await self._refresh_cache(tts_model, volume)

        # Check cache
        cache_data = self.get_cache_metadata()

        # Cache miss: synchronous rebuild
        if cache_data is None:
            logger.info("speaker_cache.rebuild", reason="cache_miss")
            return await self._refresh_cache(tts_model, volume)

        # Cache hit: check staleness
        is_stale = self.is_cache_stale(cache_data)

        if is_stale:
            # Stale cache: return stale data + trigger async refresh
            logger.info("speaker_cache.stale", action="async_refresh")
            asyncio.create_task(self._async_refresh(tts_model, volume))

        # Return cached data (fresh or stale)
        last_updated = datetime.fromisoformat(cache_data["last_updated"])
        age_days = (datetime.utcnow() - last_updated).days

        return {
            "speakers": cache_data["speakers"],
            "count": cache_data["count"],
            "last_updated": last_updated,
            "cache_age_days": age_days,
        }

    async def _refresh_cache(self, tts_model, volume) -> Dict[str, Any]:
        """Synchronously refresh cache from TTS model.

        Args:
            tts_model: Loaded TTS model instance
            volume: Modal Volume instance

        Returns:
            Refreshed cache data
        """
        async with self._refresh_lock:
            # Extract speakers from loaded model
            speakers = self._discover_speakers(tts_model)

            if not speakers:
                logger.warning("speaker_cache.refresh_failed", reason="no_speakers_found")
                return {
                    "speakers": [],
                    "count": 0,
                    "last_updated": datetime.utcnow(),
                    "cache_age_days": 0,
                }

            # Write to cache
            self.write_cache(speakers, volume)

            return {
                "speakers": speakers,
                "count": len(speakers),
                "last_updated": datetime.utcnow(),
                "cache_age_days": 0,
            }

    async def _async_refresh(self, tts_model, volume) -> None:
        """Asynchronously refresh cache in background.

        Args:
            tts_model: Loaded TTS model instance
            volume: Modal Volume instance
        """
        try:
            logger.info("speaker_cache.async_refresh.start")
            await self._refresh_cache(tts_model, volume)
            logger.info("speaker_cache.async_refresh.complete")
        except Exception as e:
            logger.error("speaker_cache.async_refresh.failed", error=str(e))

    def _discover_speakers(self, tts_model) -> List[str]:
        """Discover speakers from loaded TTS model.

        Uses the pattern from story_reels: tts.speakers attribute.
        Reference: tts_v2/engines/coqui_xtts.py:149-161

        Args:
            tts_model: Loaded TTS model instance

        Returns:
            List of speaker names
        """
        if not hasattr(tts_model, "speakers") or not tts_model.speakers:
            logger.warning("speaker_cache.no_speakers_attr")
            return []

        speakers = tts_model.speakers

        # Handle both list and dict formats
        if isinstance(speakers, list):
            return speakers
        elif isinstance(speakers, dict):
            return list(speakers.keys())

        logger.warning("speaker_cache.unexpected_speakers_type", type=type(speakers).__name__)
        return []
