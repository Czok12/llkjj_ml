#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Cache Invalidation Engine
===========================================

Intelligente Cache-Invalidation für optimale Performance:
- Content-basierte Invalidation bei PDF-Änderungen
- Time-basierte Rules für Business-Logik Updates
- Schema-Version Tracking für Model-Upgrades
- Smart Clean-up für Speicher-Optimierung

Autor: LLKJJ ML Pipeline Team
Version: 4.1.0 (Business Value Maximierung)
Datum: 19. August 2025
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..settings_bridge import Config

logger = logging.getLogger(__name__)


class CacheInvalidationEngine:
    """
    🧹 Intelligente Cache-Invalidation für LLKJJ ML Pipeline.

    Regelt automatische Cache-Löschung basierend auf:
    - Content-Änderungen (PDF-Hash-Vergleich)
    - Zeitbasierte Rules (Max-Age, Business-Zyklen)
    - Schema-Versionen (Model-Updates, Pipeline-Upgrades)
    - Speicher-Limits (Automatic Cleanup bei Bedarf)
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache-Datenbanken
        self.pdf_cache_db = self.cache_dir / "pdf_hash_cache.db"
        self.invalidation_db = self.cache_dir / "cache_invalidation.db"

        # Cache-Rules Configuration
        self.rules: dict[str, int | float | str] = {
            "max_age_days": 30,  # PDFs älter als 30 Tage invalidieren
            "max_cache_size_mb": 500.0,  # Cache-Limit: 500MB
            "schema_version": "v4.1.0",  # Aktuelle Pipeline-Version
            "business_cycle_hours": 24,  # Geschäfts-Zyklen: täglich
            "emergency_cleanup_threshold": 0.9,  # 90% Speicher-Auslastung
        }

        self._init_invalidation_database()
        logger.info("🧹 CacheInvalidationEngine initialisiert")

    def _init_invalidation_database(self) -> None:
        """Initialisiert die Cache-Invalidation-Datenbank."""
        with sqlite3.connect(self.invalidation_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT UNIQUE NOT NULL,
                    rule_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS invalidation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL,
                    invalidation_reason TEXT NOT NULL,
                    invalidated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    files_deleted INTEGER DEFAULT 0,
                    space_freed_mb REAL DEFAULT 0.0
                )
            """
            )

            # Speichere aktuelle Rules
            for rule_name, rule_value in self.rules.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_rules (rule_name, rule_value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                    (rule_name, str(rule_value)),
                )

            conn.commit()

    def check_pdf_content_change(self, pdf_path: str | Path) -> bool:
        """
        Prüft ob PDF-Inhalt sich geändert hat.

        Args:
            pdf_path: Pfad zur PDF-Datei

        Returns:
            True wenn Inhalt geändert, False wenn unverändert
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return True  # Datei gelöscht → Cache invalid

        # Aktueller PDF-Hash
        current_hash = self._calculate_pdf_hash(pdf_path)

        # Gespeicherter Hash aus Cache
        cached_hash = self._get_cached_hash(pdf_path)

        if cached_hash is None:
            # Kein Cache-Eintrag → keine Änderung erkennbar
            return False

        content_changed = current_hash != cached_hash

        if content_changed:
            logger.info(
                "📄 PDF-Inhalt geändert: %s (Hash: %s → %s)",
                pdf_path.name,
                cached_hash[:12],
                current_hash[:12],
            )

        return content_changed

    def _calculate_pdf_hash(self, pdf_path: Path) -> str:
        """Berechnet SHA-256 Hash einer PDF-Datei."""
        hasher = hashlib.sha256()

        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _get_cached_hash(self, pdf_path: Path) -> str | None:
        """Holt gespeicherten PDF-Hash aus Cache-Datenbank."""
        if not self.pdf_cache_db.exists():
            return None

        try:
            with sqlite3.connect(self.pdf_cache_db) as conn:
                cursor = conn.execute(
                    """
                    SELECT pdf_hash FROM pdf_cache
                    WHERE pdf_path = ?
                    ORDER BY processed_at DESC
                    LIMIT 1
                """,
                    (str(pdf_path),),
                )

                row = cursor.fetchone()
                return row[0] if row else None

        except sqlite3.Error as e:
            logger.warning("⚠️ Cache-Hash-Abfrage fehlgeschlagen: %s", e)
            return None

    def invalidate_by_age(self, max_age_days: int | None = None) -> dict[str, Any]:
        """
        Invalidiert Cache-Einträge basierend auf Alter.

        Args:
            max_age_days: Maximales Alter in Tagen (default: aus rules)

        Returns:
            Invalidation-Statistiken
        """
        max_age = max_age_days or int(self.rules["max_age_days"])
        cutoff_date = datetime.now() - timedelta(days=max_age)

        logger.info("🕐 Starte Age-basierte Cache-Invalidation (>%d Tage)", max_age)

        stats: dict[str, Any] = {
            "invalidated_entries": 0,
            "space_freed_mb": 0.0,
            "cutoff_date": cutoff_date.isoformat(),
        }

        if not self.pdf_cache_db.exists():
            return stats

        try:
            with sqlite3.connect(self.pdf_cache_db) as conn:
                # Finde alte Cache-Einträge
                cursor = conn.execute(
                    """
                    SELECT pdf_path, pdf_hash, processed_at
                    FROM pdf_cache
                    WHERE datetime(processed_at) < datetime(?)
                """,
                    (cutoff_date.isoformat(),),
                )

                old_entries = cursor.fetchall()

                for pdf_path, pdf_hash, processed_at in old_entries:
                    # Lösche Cache-Eintrag
                    space_freed = float(self._delete_cache_entry(pdf_path, pdf_hash))
                    stats["invalidated_entries"] = (
                        int(stats.get("invalidated_entries", 0)) + 1
                    )
                    stats["space_freed_mb"] = (
                        float(stats.get("space_freed_mb", 0.0)) + space_freed
                    )

                    # Logge Invalidation
                    self._log_invalidation(
                        cache_key=f"{pdf_path}:{pdf_hash[:12]}",
                        reason=f"Age-based cleanup (>{max_age} days)",
                        space_freed_mb=space_freed,
                    )

                    logger.debug(
                        "🗑️ Cache invalidiert: %s (Alter: %s)",
                        Path(pdf_path).name,
                        processed_at,
                    )

        except sqlite3.Error as e:
            logger.error("❌ Age-basierte Invalidation fehlgeschlagen: %s", e)

        logger.info(
            "✅ Age-Invalidation abgeschlossen: %d Einträge, %.1fMB befreit",
            stats["invalidated_entries"],
            stats["space_freed_mb"],
        )

        return stats

    def invalidate_by_schema_version(
        self, current_version: str | None = None
    ) -> dict[str, Any]:
        """
        Invalidiert Cache bei Schema/Pipeline-Version-Änderungen.

        Args:
            current_version: Aktuelle Pipeline-Version (default: aus rules)

        Returns:
            Invalidation-Statistiken
        """
        current_version = current_version or str(self.rules["schema_version"])

        logger.info("🔄 Starte Schema-Version Cache-Invalidation (%s)", current_version)

        stats = {
            "invalidated_entries": 0,
            "space_freed_mb": 0.0,
            "schema_version": current_version,
        }

        # Prüfe gespeicherte Schema-Version
        with sqlite3.connect(self.invalidation_db) as conn:
            cursor = conn.execute(
                """
                SELECT rule_value FROM cache_rules
                WHERE rule_name = 'schema_version'
            """
            )

            row = cursor.fetchone()
            stored_version = row[0] if row else None

            if stored_version and stored_version != current_version:
                # Schema-Version geändert → komplette Cache-Invalidation
                logger.info(
                    "🔄 Schema-Version-Änderung erkannt: %s → %s",
                    stored_version,
                    current_version,
                )

                stats = self._invalidate_all_cache(
                    reason=f"Schema version change ({stored_version} → {current_version})"
                )

                # Update Schema-Version
                conn.execute(
                    """
                    UPDATE cache_rules
                    SET rule_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE rule_name = 'schema_version'
                """,
                    (current_version,),
                )
                conn.commit()

        logger.info(
            "✅ Schema-Version-Invalidation abgeschlossen: %d Einträge, %.1fMB befreit",
            stats["invalidated_entries"],
            stats["space_freed_mb"],
        )

        return stats

    def emergency_cleanup(self) -> dict[str, Any]:
        """
        Notfall-Cleanup bei kritischer Speicher-Auslastung.

        Returns:
            Cleanup-Statistiken
        """
        logger.warning("🚨 Notfall-Cleanup gestartet!")

        # 1) Aggressiver Age-based cleanup
        age_stats: dict[str, Any] = self.invalidate_by_age(max_age_days=7)

        # 2) Aktuelle Cache-Größe prüfen
        current_size_mb: float = float(self._get_cache_size_mb())
        max_size_mb: float = float(self.rules.get("max_cache_size_mb", 500.0))

        stats: dict[str, Any] = {
            "age_cleanup": age_stats,
            "cache_size_mb": current_size_mb,
            "max_size_mb": max_size_mb,
            "additional_cleanup": {},
        }

        # 3) Falls über Limit, führe LRU-Cleanup aus
        if current_size_mb > max_size_mb:
            additional_stats: dict[str, Any] = self._cleanup_by_lru(
                target_size_mb=max_size_mb * 0.8
            )
            stats["additional_cleanup"] = additional_stats

        # 4) Zusammenrechnen der freigegebenen Größe
        additional_cleanup = stats.get("additional_cleanup")
        additional_cleanup_freed: float = 0.0
        if isinstance(additional_cleanup, dict):
            additional_cleanup_freed = float(
                additional_cleanup.get("space_freed_mb", 0.0)
            )

        age_freed = float(age_stats.get("space_freed_mb", 0.0))
        total_freed: float = age_freed + additional_cleanup_freed

        logger.warning(
            "🚨 Notfall-Cleanup abgeschlossen: %.1fMB befreit, Cache: %.1fMB",
            total_freed,
            self._get_cache_size_mb(),
        )

        return stats

    def _cleanup_by_lru(self, target_size_mb: float) -> dict[str, Any]:
        """Cleanup nach Least-Recently-Used Prinzip."""
        logger.info("🔄 LRU-Cleanup startet (Ziel: %.1fMB)", target_size_mb)

        stats = {
            "invalidated_entries": 0,
            "space_freed_mb": 0.0,
            "target_size_mb": target_size_mb,
        }

        if not self.pdf_cache_db.exists():
            return stats

        try:
            with sqlite3.connect(self.pdf_cache_db) as conn:
                # Sortiere nach last_accessed (älteste zuerst)
                cursor = conn.execute(
                    """
                    SELECT pdf_path, pdf_hash, processed_at
                    FROM pdf_cache
                    ORDER BY datetime(processed_at) ASC
                """
                )

                lru_entries = cursor.fetchall()

                current_size = self._get_cache_size_mb()

                for pdf_path, pdf_hash, processed_at in lru_entries:
                    if current_size <= target_size_mb:
                        break  # Ziel erreicht

                    space_freed = self._delete_cache_entry(pdf_path, pdf_hash)
                    current_size -= space_freed

                    stats["invalidated_entries"] = (
                        int(stats.get("invalidated_entries", 0)) + 1
                    )
                    stats["space_freed_mb"] = float(
                        stats.get("space_freed_mb", 0.0)
                    ) + float(space_freed)

                    # Logge LRU-Cleanup
                    self._log_invalidation(
                        cache_key=f"{pdf_path}:{pdf_hash[:12]}",
                        reason=f"LRU cleanup (last access: {processed_at})",
                        space_freed_mb=space_freed,
                    )

        except sqlite3.Error as e:
            logger.error("❌ LRU-Cleanup fehlgeschlagen: %s", e)

        logger.info(
            "✅ LRU-Cleanup abgeschlossen: %d Einträge, %.1fMB befreit",
            stats["invalidated_entries"],
            stats["space_freed_mb"],
        )

        return stats

    def _invalidate_all_cache(self, reason: str) -> dict[str, Any]:
        """Invalidiert kompletten Cache."""
        logger.warning("🚨 Komplette Cache-Invalidation: %s", reason)

        stats = {"invalidated_entries": 0, "space_freed_mb": 0.0, "reason": reason}

        if not self.pdf_cache_db.exists():
            return stats

        try:
            with sqlite3.connect(self.pdf_cache_db) as conn:
                # Zähle aktuelle Einträge
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_cache")
                stats["invalidated_entries"] = cursor.fetchone()[0]

                # Berechne befreiten Speicher
                stats["space_freed_mb"] = self._get_cache_size_mb()

                # Lösche alle Cache-Einträge
                conn.execute("DELETE FROM pdf_cache")
                conn.commit()

                # Logge komplette Invalidation
                invalidated_count = stats.get("invalidated_entries", 0)
                space_freed = stats.get("space_freed_mb", 0.0)

                self._log_invalidation(
                    cache_key="*ALL*",
                    reason=reason,
                    files_deleted=(
                        int(invalidated_count)
                        if isinstance(invalidated_count, int | float)
                        else 0
                    ),
                    space_freed_mb=(
                        float(space_freed)
                        if isinstance(space_freed, int | float)
                        else 0.0
                    ),
                )

        except sqlite3.Error as e:
            logger.error("❌ Komplette Cache-Invalidation fehlgeschlagen: %s", e)

        return stats

    def _delete_cache_entry(self, pdf_path: str, pdf_hash: str) -> float:
        """
        Löscht einzelnen Cache-Eintrag und gibt befreiten Speicher zurück.

        Returns:
            Befreiter Speicher in MB
        """
        try:
            with sqlite3.connect(self.pdf_cache_db) as conn:
                # Lösche Cache-Eintrag
                conn.execute(
                    """
                    DELETE FROM pdf_cache
                    WHERE pdf_path = ? AND pdf_hash = ?
                """,
                    (pdf_path, pdf_hash),
                )
                conn.commit()

            # Geschätzte Speicher-Befreiung (basierend auf PDF-Größe)
            pdf_size_mb = 0.0
            if Path(pdf_path).exists():
                pdf_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)

            # Cache-Overhead: ~2x PDF-Größe (Text + Embeddings + Metadata)
            return pdf_size_mb * 2.0

        except Exception as e:
            logger.warning("⚠️ Cache-Eintrag-Löschung fehlgeschlagen: %s", e)
            return 0.0

    def _get_cache_size_mb(self) -> float:
        """Ermittelt aktuelle Cache-Größe in MB."""
        total_size = 0.0

        # PDF-Cache-Datenbank
        if self.pdf_cache_db.exists():
            total_size += self.pdf_cache_db.stat().st_size

        # ChromaDB-Verzeichnis
        chroma_dir = Path("data/vectors")
        if chroma_dir.exists():
            for file in chroma_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size

        # Weitere Cache-Dateien
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                total_size += cache_file.stat().st_size

        return total_size / (1024 * 1024)  # Bytes → MB

    def _log_invalidation(
        self,
        cache_key: str,
        reason: str,
        files_deleted: int = 1,
        space_freed_mb: float = 0.0,
    ) -> None:
        """Loggt Cache-Invalidation in Datenbank."""
        try:
            with sqlite3.connect(self.invalidation_db) as conn:
                conn.execute(
                    """
                    INSERT INTO invalidation_log
                    (cache_key, invalidation_reason, files_deleted, space_freed_mb)
                    VALUES (?, ?, ?, ?)
                """,
                    (cache_key, reason, files_deleted, space_freed_mb),
                )
                conn.commit()

        except sqlite3.Error as e:
            logger.warning("⚠️ Invalidation-Log fehlgeschlagen: %s", e)

    def run_scheduled_maintenance(self) -> dict[str, Any]:
        """
        Führt geplante Cache-Wartung aus.

        Returns:
            Wartungs-Statistiken
        """
        logger.info("🔧 Starte geplante Cache-Wartung...")

        maintenance_stats = {
            "timestamp": datetime.now().isoformat(),
            "age_cleanup": {},
            "schema_check": {},
            "emergency_check": {},
            "total_space_freed_mb": 0.0,
        }

        # 1. Age-basierte Cleanup
        maintenance_stats["age_cleanup"] = self.invalidate_by_age()

        # 2. Schema-Version-Check
        maintenance_stats["schema_check"] = self.invalidate_by_schema_version()

        # 3. Notfall-Check bei Speicher-Überlauf
        current_size = self._get_cache_size_mb()
        max_size = float(self.rules["max_cache_size_mb"])
        emergency_threshold = float(self.rules["emergency_cleanup_threshold"])

        if current_size > max_size * emergency_threshold:
            maintenance_stats["emergency_check"] = self.emergency_cleanup()

        # Gesamt-Statistiken
        age_cleanup = maintenance_stats.get("age_cleanup", {}) or {}
        schema_check = maintenance_stats.get("schema_check", {}) or {}
        emergency_check = maintenance_stats.get("emergency_check", {}) or {}

        total_freed = float(
            (
                age_cleanup.get("space_freed_mb", 0)
                if isinstance(age_cleanup, dict)
                else 0
            )
            + (
                schema_check.get("space_freed_mb", 0)
                if isinstance(schema_check, dict)
                else 0
            )
            + (
                emergency_check.get("space_freed_mb", 0)
                if isinstance(emergency_check, dict)
                else 0
            )
        )

        maintenance_stats["total_space_freed_mb"] = total_freed
        maintenance_stats["final_cache_size_mb"] = self._get_cache_size_mb()

        logger.info(
            "✅ Cache-Wartung abgeschlossen: %.1fMB befreit, Cache: %.1fMB",
            total_freed,
            maintenance_stats["final_cache_size_mb"],
        )

        return maintenance_stats

    def get_cache_health_report(self) -> dict[str, Any]:
        """
        Erstellt Cache-Health-Report für Monitoring.

        Returns:
            Cache-Health-Daten
        """
        if not self.pdf_cache_db.exists():
            return {"status": "no_cache", "message": "Keine Cache-Datenbank gefunden"}

        try:
            with sqlite3.connect(self.pdf_cache_db) as conn:
                # Cache-Statistiken
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_entries,
                        MIN(datetime(processed_at)) as oldest_entry,
                        MAX(datetime(processed_at)) as newest_entry,
                        AVG(CASE WHEN processed_at IS NOT NULL
                            THEN (julianday('now') - julianday(processed_at))
                            ELSE NULL END) as avg_days_since_access
                    FROM pdf_cache
                """
                )

                stats = cursor.fetchone()

                # Cache-Größe und Limits
                current_size_mb = self._get_cache_size_mb()
                max_size_mb = float(self.rules["max_cache_size_mb"])
                usage_percent = (current_size_mb / max_size_mb) * 100

                # Health-Status bestimmen
                if usage_percent > 90:
                    status = "critical"
                elif usage_percent > 75:
                    status = "warning"
                elif usage_percent > 50:
                    status = "good"
                else:
                    status = "excellent"

                return {
                    "status": status,
                    "cache_size_mb": current_size_mb,
                    "max_size_mb": max_size_mb,
                    "usage_percent": usage_percent,
                    "total_entries": stats[0],
                    "oldest_entry": stats[1],
                    "newest_entry": stats[2],
                    "avg_days_since_access": stats[3] or 0,
                    "rules": self.rules,
                    "recommendations": self._get_cache_recommendations(
                        usage_percent, stats[0]
                    ),
                }

        except sqlite3.Error as e:
            logger.error("❌ Cache-Health-Report fehlgeschlagen: %s", e)
            return {"status": "error", "message": f"Datenbankfehler: {e}"}

    def _get_cache_recommendations(
        self, usage_percent: float, total_entries: int
    ) -> list[str]:
        """Generiert Cache-Optimierungs-Empfehlungen."""
        recommendations = []

        if usage_percent > 90:
            recommendations.append("🚨 Kritisch: Sofortiges Cleanup erforderlich")
            recommendations.append("💡 Empfehlung: Führen Sie emergency_cleanup() aus")

        elif usage_percent > 75:
            recommendations.append("⚠️ Warnung: Cache-Größe überwachen")
            recommendations.append("💡 Empfehlung: Age-basierte Cleanup erwägen")

        if total_entries > 1000:
            recommendations.append("📊 Info: Viele Cache-Einträge")
            recommendations.append("💡 Empfehlung: LRU-Cleanup für bessere Performance")

        if not recommendations:
            recommendations.append("✅ Cache-Gesundheit optimal")

        return recommendations


async def main() -> None:
    """Test der CacheInvalidationEngine."""
    logging.basicConfig(level=logging.INFO)

    engine = CacheInvalidationEngine()

    # Cache-Health-Report
    health_report = engine.get_cache_health_report()
    print("📊 Cache Health Report:")
    print(json.dumps(health_report, indent=2, ensure_ascii=False))

    # Geplante Wartung
    maintenance_stats = engine.run_scheduled_maintenance()
    print("\n🔧 Wartungs-Statistiken:")
    print(json.dumps(maintenance_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
