#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Cache Invalidation Manager
==============================================

Intelligente Cache-Invalidation-Strategien für optimale Datenqualität:
- Automatisches Refresh bei SKR03-Updates
- Version-basierte Cache-Invalidation
- Content-Change-Detection
- Time-based Cache-Expiry

🚀 QUICK WIN: Proaktive Cache-Qualitätssicherung

Autor: LLKJJ ML Pipeline Team
Version: 4.2.0 (Cache-Invalidation)
Datum: 19. August 2025
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..settings_bridge import ConfigBridge, config_instance

logger = logging.getLogger(__name__)


class CacheInvalidationManager:
    """
    🔄 Intelligente Cache-Invalidation für optimale Datenqualität.

    Features:
    1. SKR03-Update-Detection: Automatische Invalidation bei Regel-Änderungen
    2. Version-based Invalidation: Modell-Updates triggern Cache-Refresh
    3. Content-Change-Detection: Verdächtige Änderungen identifizieren
    4. Time-based Expiry: Automatische Alterung nach konfigurierbarer Zeit
    """

    def __init__(self, config: ConfigBridge | None = None):
        self.config = config or config_instance
        self.cache_db_path = Path("data/cache/pdf_hash_cache.db")
        self.invalidation_log_path = Path("data/cache/invalidation_log.db")

        # 🔄 INVALIDATION-KONFIGURATION
        self.invalidation_rules = {
            "max_cache_age_days": 90,  # Cache automatisch nach 90 Tagen invalidieren
            "skr03_update_threshold": 0.1,  # 10% SKR03-Regel-Änderungen → Full Invalidation
            "confidence_drop_threshold": 0.15,  # 15% Konfidenz-Drop → Verdächtig
            "quality_degradation_threshold": 0.2,  # 20% Qualitäts-Drop → Invalidation
        }

        # 📊 TRACKING-METRIKEN
        self.tracking_metrics: dict[str, Any] = {
            "last_skr03_check": None,
            "last_model_version": None,
            "total_invalidations": 0,
            "skr03_triggered_invalidations": 0,
            "time_triggered_invalidations": 0,
            "quality_triggered_invalidations": 0,
        }

        self._setup_invalidation_database()
        logger.info(
            "🔄 CacheInvalidationManager initialisiert - Proaktive Cache-Qualitätssicherung"
        )

    def _setup_invalidation_database(self) -> None:
        """Setup SQLite database für Invalidation-Tracking."""
        self.invalidation_log_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.invalidation_log_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS invalidation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    invalidation_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    affected_entries INTEGER NOT NULL,
                    trigger_data TEXT,
                    performance_impact_ms INTEGER
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skr03_version_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    skr03_hash TEXT NOT NULL,
                    rules_count INTEGER NOT NULL,
                    changes_detected TEXT,
                    invalidation_triggered BOOLEAN DEFAULT FALSE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    avg_confidence REAL NOT NULL,
                    quality_distribution TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    cache_hit_rate REAL NOT NULL
                )
            """
            )

    def setup_database(self) -> None:
        """Öffentliche Methode zum Setup der Invalidation-Datenbank (für Tests)."""
        self._setup_invalidation_database()
        self._setup_cache_database()

    def _setup_cache_database(self) -> None:
        """Setup der PDF-Cache-Datenbank für Tests."""
        import sqlite3

        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_cache (
                    pdf_hash TEXT PRIMARY KEY,
                    pdf_path TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    result_json TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    extraction_quality TEXT NOT NULL
                )
            """
            )

    def check_skr03_updates(self) -> dict[str, Any]:
        """
        🎯 SKR03-Update-Detection: Prüfe auf Änderungen in SKR03-Regeln.

        Returns:
            Dictionary mit Update-Status und Invalidation-Empfehlungen
        """
        start_time = time.time()

        try:
            # SKR03-Regeln-Hash berechnen
            skr03_config_path = Path("data/config/skr03_regeln.yaml")
            if not skr03_config_path.exists():
                logger.warning(
                    "⚠️ SKR03-Regeln-Datei nicht gefunden: %s", skr03_config_path
                )
                return {"status": "error", "message": "SKR03-Regeln nicht verfügbar"}

            # Hash der aktuellen SKR03-Regeln
            with open(skr03_config_path, encoding="utf-8") as f:
                content = f.read()
                import hashlib

                current_hash = hashlib.sha256(content.encode()).hexdigest()

            # Aktuelle Regeln-Anzahl zählen
            import yaml

            with open(skr03_config_path, encoding="utf-8") as f:
                skr03_data = yaml.safe_load(f)
                if skr03_data is None:
                    skr03_data = {}
                rules_count = len(skr03_data.get("kontierungsregeln", []))

            # Letzten bekannten Hash abrufen
            with sqlite3.connect(self.invalidation_log_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT skr03_hash, rules_count, timestamp
                    FROM skr03_version_tracking
                    ORDER BY timestamp DESC LIMIT 1
                """
                )
                last_record = cursor.fetchone()

            if last_record:
                last_hash, last_rules_count, last_timestamp = last_record

                # Änderungen detektiert?
                hash_changed = current_hash != last_hash
                rules_count_changed = rules_count != last_rules_count

                if hash_changed or rules_count_changed:
                    # Detaillierte Änderungs-Analyse
                    changes_detected = []
                    if hash_changed:
                        changes_detected.append("rule_content_modified")
                    if rules_count_changed:
                        changes_detected.append(
                            f"rules_count: {last_rules_count} → {rules_count}"
                        )

                    # Schweregrad bestimmen
                    rules_change_percent = abs(rules_count - last_rules_count) / max(
                        1, last_rules_count
                    )
                    severe_change = (
                        rules_change_percent
                        > self.invalidation_rules["skr03_update_threshold"]
                    )

                    # Invalidation-Empfehlung
                    invalidation_recommended = severe_change or hash_changed

                    # SKR03-Update protokollieren
                    changes_json = json.dumps(changes_detected)
                    with sqlite3.connect(self.invalidation_log_path) as conn:
                        conn.execute(
                            """
                            INSERT INTO skr03_version_tracking
                            (skr03_hash, rules_count, changes_detected, invalidation_triggered)
                            VALUES (?, ?, ?, ?)
                        """,
                            (
                                current_hash,
                                rules_count,
                                changes_json,
                                invalidation_recommended,
                            ),
                        )

                    check_time_ms = int((time.time() - start_time) * 1000)

                    result = {
                        "status": "changes_detected",
                        "hash_changed": hash_changed,
                        "rules_count_changed": rules_count_changed,
                        "rules_change_percent": round(rules_change_percent, 3),
                        "severe_change": severe_change,
                        "invalidation_recommended": invalidation_recommended,
                        "changes_detected": changes_detected,
                        "current_rules_count": rules_count,
                        "previous_rules_count": last_rules_count,
                        "check_time_ms": check_time_ms,
                    }

                    if invalidation_recommended:
                        logger.warning(
                            "🚨 SKR03-Update detektiert: %s → Cache-Invalidation empfohlen!",
                            changes_detected,
                        )
                    else:
                        logger.info(
                            "📊 SKR03-Update detektiert: %s → Keine Invalidation erforderlich",
                            changes_detected,
                        )

                    return result

                else:
                    # Keine Änderungen
                    check_time_ms = int((time.time() - start_time) * 1000)
                    return {
                        "status": "no_changes",
                        "current_rules_count": rules_count,
                        "last_check": last_timestamp,
                        "check_time_ms": check_time_ms,
                    }

            else:
                # Erste SKR03-Baseline erstellen
                with sqlite3.connect(self.invalidation_log_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO skr03_version_tracking
                        (skr03_hash, rules_count, changes_detected, invalidation_triggered)
                        VALUES (?, ?, ?, ?)
                    """,
                        (current_hash, rules_count, "baseline_created", False),
                    )

                logger.info("📊 SKR03-Baseline erstellt: %d Regeln", rules_count)

                return {
                    "status": "baseline_created",
                    "current_rules_count": rules_count,
                    "check_time_ms": int((time.time() - start_time) * 1000),
                }

        except Exception as e:
            logger.error("❌ SKR03-Update-Check fehlgeschlagen: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "check_time_ms": int((time.time() - start_time) * 1000),
            }

    def invalidate_cache_by_age(
        self, max_age_days: int | None = None
    ) -> dict[str, Any]:
        """
        ⏰ Time-based Cache-Invalidation: Entferne veraltete Cache-Einträge.

        Args:
            max_age_days: Maximales Alter in Tagen (default: config-Wert)

        Returns:
            Dictionary mit Invalidation-Statistiken
        """
        start_time = time.time()
        max_age = max_age_days or self.invalidation_rules["max_cache_age_days"]

        try:
            cutoff_date = datetime.now() - timedelta(days=max_age)

            # Alte Einträge identifizieren und löschen
            with sqlite3.connect(self.cache_db_path) as conn:
                # Zähle betroffene Einträge
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM pdf_cache
                    WHERE processed_at < ?
                """,
                    (cutoff_date,),
                )
                affected_count = cursor.fetchone()[0]

                if affected_count > 0:
                    # Lösche alte Einträge
                    conn.execute(
                        """
                        DELETE FROM pdf_cache
                        WHERE processed_at < ?
                    """,
                        (cutoff_date,),
                    )

            # VACUUM außerhalb der Transaktion für Speicherplatz-Freigabe
            if affected_count > 0:
                with sqlite3.connect(self.cache_db_path) as conn:
                    conn.execute("VACUUM")

            # Invalidation protokollieren
            invalidation_time_ms = int((time.time() - start_time) * 1000)

            if affected_count > 0:
                self._log_invalidation(
                    invalidation_type="time_based",
                    reason=f"cache_age_exceeded_{max_age}_days",
                    affected_entries=affected_count,
                    trigger_data=json.dumps(
                        {
                            "max_age_days": max_age,
                            "cutoff_date": cutoff_date.isoformat(),
                        }
                    ),
                    performance_impact_ms=invalidation_time_ms,
                )

                self.tracking_metrics["time_triggered_invalidations"] += 1
                self.tracking_metrics["total_invalidations"] += 1

                logger.info(
                    "⏰ Time-based Cache-Invalidation: %d Einträge gelöscht (älter als %d Tage) in %dms",
                    affected_count,
                    max_age,
                    invalidation_time_ms,
                )

            return {
                "status": "success",
                "invalidated_entries": affected_count,
                "max_age_days": max_age,
                "cutoff_date": cutoff_date.isoformat(),
                "processing_time_ms": invalidation_time_ms,
            }

        except Exception as e:
            logger.error("❌ Time-based Cache-Invalidation fehlgeschlagen: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    def invalidate_cache_by_quality(self) -> dict[str, Any]:
        """
        📊 Quality-based Cache-Invalidation: Entferne Einträge mit degradierter Qualität.

        Returns:
            Dictionary mit Quality-Invalidation-Statistiken
        """
        start_time = time.time()

        try:
            # Aktuelle Cache-Qualitäts-Metriken sammeln
            with sqlite3.connect(self.cache_db_path) as conn:
                # Durchschnittliche Konfidenz
                cursor = conn.execute("SELECT AVG(confidence_score) FROM pdf_cache")
                avg_confidence = cursor.fetchone()[0] or 0.0

                # Qualitäts-Verteilung
                cursor = conn.execute(
                    """
                    SELECT extraction_quality, COUNT(*)
                    FROM pdf_cache
                    GROUP BY extraction_quality
                """
                )
                quality_dist = dict(cursor.fetchall())

                # Niedrig-Qualitäts-Einträge identifizieren
                low_quality_threshold = 0.7  # <70% Konfidenz als "low quality"
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM pdf_cache
                    WHERE confidence_score < ? OR extraction_quality = 'low'
                """,
                    (low_quality_threshold,),
                )
                low_quality_count = cursor.fetchone()[0]

                # Degradierte Einträge löschen (optional, falls zu viele)
                total_entries = sum(quality_dist.values())
                low_quality_ratio = low_quality_count / max(1, total_entries)

                invalidated_count = 0
                if (
                    low_quality_ratio
                    > self.invalidation_rules["quality_degradation_threshold"]
                ):
                    # Löschen wenn >20% der Einträge low-quality sind
                    cursor = conn.execute(
                        """
                        DELETE FROM pdf_cache
                        WHERE confidence_score < ? OR extraction_quality = 'low'
                    """,
                        (low_quality_threshold,),
                    )
                    invalidated_count = cursor.rowcount

            # VACUUM außerhalb der Transaktion für Cleanup
            if invalidated_count > 0:
                with sqlite3.connect(self.cache_db_path) as conn:
                    conn.execute("VACUUM")

            # Quality-Metriken protokollieren
            quality_metrics = {
                "avg_confidence": round(avg_confidence, 3),
                "quality_distribution": quality_dist,
                "low_quality_count": low_quality_count,
                "low_quality_ratio": round(low_quality_ratio, 3),
                "total_entries": total_entries,
            }

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Quality-Metriken in Datenbank speichern
            with sqlite3.connect(self.invalidation_log_path) as conn:
                conn.execute(
                    """
                    INSERT INTO cache_quality_metrics
                    (avg_confidence, quality_distribution, performance_metrics, cache_hit_rate)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        avg_confidence,
                        json.dumps(quality_dist),
                        json.dumps({"processing_time_ms": processing_time_ms}),
                        0.0,  # Cache-Hit-Rate wird separat berechnet
                    ),
                )

            if invalidated_count > 0:
                # Invalidation protokollieren
                self._log_invalidation(
                    invalidation_type="quality_based",
                    reason=f"quality_degradation_ratio_{low_quality_ratio:.2f}",
                    affected_entries=invalidated_count,
                    trigger_data=json.dumps(quality_metrics),
                    performance_impact_ms=processing_time_ms,
                )

                self.tracking_metrics["quality_triggered_invalidations"] += 1
                self.tracking_metrics["total_invalidations"] += 1

                logger.warning(
                    "📊 Quality-based Cache-Invalidation: %d Einträge gelöscht (Qualitätsratio: %.2f) in %dms",
                    invalidated_count,
                    low_quality_ratio,
                    processing_time_ms,
                )
            else:
                logger.info(
                    "✅ Cache-Qualität OK: Avg-Konfidenz %.3f, Low-Quality-Ratio %.2f",
                    avg_confidence,
                    low_quality_ratio,
                )

            return {
                "status": "success",
                "quality_metrics": quality_metrics,
                "invalidated_entries": invalidated_count,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as e:
            logger.error("❌ Quality-based Cache-Invalidation fehlgeschlagen: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    def force_full_invalidation(self, reason: str = "manual_trigger") -> dict[str, Any]:
        """
        🚨 Vollständige Cache-Invalidation: Kompletter Cache-Reset.

        Args:
            reason: Grund für die Invalidation

        Returns:
            Dictionary mit Full-Invalidation-Statistiken
        """
        start_time = time.time()

        try:
            # Cache-Größe vor Invalidation
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_cache")
                total_entries = cursor.fetchone()[0]

                # Alle Einträge löschen
                conn.execute("DELETE FROM pdf_cache")

            # Database optimieren außerhalb der Transaktion
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("VACUUM")

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Invalidation protokollieren
            self._log_invalidation(
                invalidation_type="full_invalidation",
                reason=reason,
                affected_entries=total_entries,
                trigger_data=json.dumps({"manual_trigger": True}),
                performance_impact_ms=processing_time_ms,
            )

            self.tracking_metrics["total_invalidations"] += 1

            logger.warning(
                "🚨 Vollständige Cache-Invalidation: %d Einträge gelöscht ('%s') in %dms",
                total_entries,
                reason,
                processing_time_ms,
            )

            return {
                "status": "success",
                "invalidated_entries": total_entries,
                "reason": reason,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as e:
            logger.error("❌ Vollständige Cache-Invalidation fehlgeschlagen: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    def _log_invalidation(
        self,
        invalidation_type: str,
        reason: str,
        affected_entries: int,
        trigger_data: str | None = None,
        performance_impact_ms: int = 0,
    ) -> None:
        """Interne Methode: Invalidation in Log-Database protokollieren."""
        try:
            with sqlite3.connect(self.invalidation_log_path) as conn:
                conn.execute(
                    """
                    INSERT INTO invalidation_log
                    (invalidation_type, reason, affected_entries, trigger_data, performance_impact_ms)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        invalidation_type,
                        reason,
                        affected_entries,
                        trigger_data,
                        performance_impact_ms,
                    ),
                )
        except Exception as e:
            logger.error("❌ Invalidation-Logging fehlgeschlagen: %s", e)

    def get_invalidation_statistics(self) -> dict[str, Any]:
        """
        📊 Invalidation-Statistiken für Monitoring und Debugging.

        Returns:
            Umfassende Invalidation-Metriken und Trends
        """
        try:
            with sqlite3.connect(self.invalidation_log_path) as conn:
                # Gesamt-Statistiken
                cursor = conn.execute(
                    """
                    SELECT invalidation_type, COUNT(*), SUM(affected_entries)
                    FROM invalidation_log
                    GROUP BY invalidation_type
                """
                )
                type_stats = {
                    row[0]: {"count": row[1], "total_entries": row[2]} for row in cursor
                }

                # Letzte Invalidations
                cursor = conn.execute(
                    """
                    SELECT timestamp, invalidation_type, reason, affected_entries
                    FROM invalidation_log
                    ORDER BY timestamp DESC LIMIT 10
                """
                )
                recent_invalidations = [
                    {
                        "timestamp": row[0],
                        "type": row[1],
                        "reason": row[2],
                        "affected_entries": row[3],
                    }
                    for row in cursor
                ]

                # Performance-Impact
                cursor = conn.execute(
                    """
                    SELECT AVG(performance_impact_ms), MAX(performance_impact_ms)
                    FROM invalidation_log
                """
                )
                perf_stats = cursor.fetchone()
                avg_impact_ms = perf_stats[0] if perf_stats[0] else 0
                max_impact_ms = perf_stats[1] if perf_stats[1] else 0

            return {
                "tracking_metrics": self.tracking_metrics,
                "invalidation_by_type": type_stats,
                "recent_invalidations": recent_invalidations,
                "performance_impact": {
                    "avg_impact_ms": round(avg_impact_ms, 2),
                    "max_impact_ms": max_impact_ms,
                },
                "configuration": self.invalidation_rules,
            }

        except Exception as e:
            logger.error("❌ Invalidation-Statistiken-Abruf fehlgeschlagen: %s", e)
            return {"status": "error", "message": str(e)}

    def get_cache_health_report(self) -> dict[str, Any]:
        """
        📊 Erstellt umfassenden Cache-Health-Report.

        Returns:
            Dictionary mit Cache-Health-Informationen
        """
        try:
            if not Path(self.cache_db_path).exists():
                return {"status": "no_cache"}

            with sqlite3.connect(self.cache_db_path) as conn:
                # Cache-Größe ermitteln
                cache_size_bytes = Path(self.cache_db_path).stat().st_size
                cache_size_mb = cache_size_bytes / (1024 * 1024)

                # Anzahl Einträge
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_cache")
                total_entries = cursor.fetchone()[0]

                if total_entries == 0:
                    return {
                        "status": "empty",
                        "cache_size_mb": cache_size_mb,
                        "total_entries": 0,
                    }

                # Ältester Eintrag
                cursor = conn.execute("SELECT MIN(processed_at) FROM pdf_cache")
                oldest_entry = cursor.fetchone()[0]

                # Durchschnittliches Alter
                cursor = conn.execute("""
                    SELECT AVG(julianday('now') - julianday(processed_at))
                    FROM pdf_cache
                """)
                avg_days_since_access = cursor.fetchone()[0] or 0

                # Cache-Auslastung bestimmen
                max_size_mb = 1000  # Default max cache size
                usage_percent = (cache_size_mb / max_size_mb) * 100

                # Empfehlungen generieren
                recommendations = []
                if usage_percent > 80:
                    recommendations.append(
                        "🚨 Cache-Größe überschreitet 80% - Cleanup empfohlen"
                    )
                if avg_days_since_access > 30:
                    recommendations.append(
                        "⏰ Viele alte Einträge - Age-based Cleanup empfohlen"
                    )
                if total_entries > 10000:
                    recommendations.append(
                        "📦 Hohe Anzahl Einträge - Maintenance empfohlen"
                    )

                return {
                    "status": "healthy" if usage_percent < 80 else "warning",
                    "cache_size_mb": cache_size_mb,
                    "max_size_mb": max_size_mb,
                    "usage_percent": usage_percent,
                    "total_entries": total_entries,
                    "oldest_entry": oldest_entry,
                    "avg_days_since_access": avg_days_since_access,
                    "recommendations": recommendations,
                }

        except Exception as e:
            logger.error("❌ Cache-Health-Report fehlgeschlagen: %s", e)
            return {"status": "error", "message": str(e)}

    def invalidate_by_age(self, max_age_days: int) -> dict[str, Any]:
        """
        Alias für invalidate_cache_by_age für CLI-Kompatibilität.

        Args:
            max_age_days: Maximales Alter in Tagen

        Returns:
            Dictionary mit Invalidation-Statistiken
        """
        result = self.invalidate_cache_by_age(max_age_days)
        return {
            "invalidated_entries": result.get("invalidated_entries", 0),
            "space_freed_mb": 0,  # Placeholder
            "cutoff_date": result.get("cutoff_date", ""),
        }

    def invalidate_by_schema_version(
        self, target_version: str | None = None
    ) -> dict[str, Any]:
        """
        🔄 Schema-basierte Cache-Invalidation bei Modell-Updates.

        Args:
            target_version: Schema-Version zum Invalidieren

        Returns:
            Dictionary mit Invalidation-Statistiken
        """
        start_time = time.time()

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Zähle betroffene Einträge
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM pdf_cache WHERE schema_version != ? OR schema_version IS NULL",
                    (target_version or "current",),
                )
                affected_count = cursor.fetchone()[0]

                if affected_count > 0:
                    # Lösche inkompatible Einträge
                    conn.execute(
                        "DELETE FROM pdf_cache WHERE schema_version != ? OR schema_version IS NULL",
                        (target_version or "current",),
                    )
                    conn.execute("VACUUM")

            invalidation_time_ms = int((time.time() - start_time) * 1000)

            if affected_count > 0:
                self._log_invalidation(
                    invalidation_type="schema_based",
                    reason=f"schema_version_mismatch_{target_version}",
                    affected_entries=affected_count,
                    trigger_data=json.dumps({"target_version": target_version}),
                    performance_impact_ms=invalidation_time_ms,
                )

                logger.info(
                    "🔄 Schema-basierte Invalidation: %d Einträge gelöscht in %dms",
                    affected_count,
                    invalidation_time_ms,
                )

            return {
                "status": "success",
                "invalidated_entries": affected_count,
                "target_version": target_version,
                "processing_time_ms": invalidation_time_ms,
            }

        except Exception as e:
            logger.error("❌ Schema-basierte Invalidation fehlgeschlagen: %s", e)
            return {"status": "error", "message": str(e)}

    def emergency_cleanup(self, force: bool = False) -> dict[str, Any]:
        """
        🚨 Notfall-Cache-Cleanup bei kritischen Speicherproblemen.

        Args:
            force: Erzwinge aggressive Bereinigung

        Returns:
            Dictionary mit Cleanup-Statistiken
        """
        start_time = time.time()

        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Gesamtgröße vor Cleanup
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_cache")
                initial_count = cursor.fetchone()[0]

                if force:
                    # Aggressive Bereinigung: Lösche alles älter als 7 Tage
                    cutoff_date = datetime.now() - timedelta(days=7)
                    conn.execute(
                        "DELETE FROM pdf_cache WHERE processed_at < ?", (cutoff_date,)
                    )
                else:
                    # Standard-Notfall: Lösche älteste 50%
                    cursor = conn.execute(
                        "SELECT processed_at FROM pdf_cache ORDER BY processed_at LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM pdf_cache)"
                    )
                    result = cursor.fetchone()
                    if result:
                        median_date = result[0]
                        conn.execute(
                            "DELETE FROM pdf_cache WHERE processed_at <= ?",
                            (median_date,),
                        )

                # Neue Anzahl ermitteln
                cursor = conn.execute("SELECT COUNT(*) FROM pdf_cache")
                final_count = cursor.fetchone()[0]

                cleaned_entries = initial_count - final_count

                if cleaned_entries > 0:
                    conn.execute("VACUUM")

            cleanup_time_ms = int((time.time() - start_time) * 1000)

            if cleaned_entries > 0:
                self._log_invalidation(
                    invalidation_type="emergency",
                    reason="emergency_cleanup_triggered",
                    affected_entries=cleaned_entries,
                    trigger_data=json.dumps({"force": force}),
                    performance_impact_ms=cleanup_time_ms,
                )

                logger.warning(
                    "🚨 Emergency Cleanup: %d Einträge gelöscht in %dms",
                    cleaned_entries,
                    cleanup_time_ms,
                )

            return {
                "status": "success",
                "initial_entries": initial_count,
                "final_entries": final_count,
                "cleaned_entries": cleaned_entries,
                "force_mode": force,
                "processing_time_ms": cleanup_time_ms,
            }

        except Exception as e:
            logger.error("❌ Emergency Cleanup fehlgeschlagen: %s", e)
            return {"status": "error", "message": str(e)}

    def run_scheduled_maintenance(self) -> dict[str, Any]:
        """
        🔧 Führt geplante Cache-Wartung durch.

        Returns:
            Dictionary mit Wartungs-Statistiken
        """
        start_time = time.time()
        maintenance_actions = []

        try:
            # 1. Health Check
            health_report = self.get_cache_health_report()
            maintenance_actions.append(
                f"Health Check: {health_report.get('status', 'unknown')}"
            )

            # 2. Age-based Cleanup (> 30 Tage)
            age_result = self.invalidate_cache_by_age(30)
            if age_result.get("invalidated_entries", 0) > 0:
                maintenance_actions.append(
                    f"Age Cleanup: {age_result['invalidated_entries']} Einträge gelöscht"
                )

            # 3. Quality-based Cleanup
            quality_result = self.invalidate_cache_by_quality(min_confidence=0.7)
            if quality_result.get("invalidated_entries", 0) > 0:
                maintenance_actions.append(
                    f"Quality Cleanup: {quality_result['invalidated_entries']} Einträge gelöscht"
                )

            # 4. Database Optimization
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                maintenance_actions.append("Database optimiert")

            maintenance_time_ms = int((time.time() - start_time) * 1000)

            # Wartung protokollieren
            self._log_invalidation(
                invalidation_type="maintenance",
                reason="scheduled_maintenance",
                affected_entries=0,
                trigger_data=json.dumps({"actions": maintenance_actions}),
                performance_impact_ms=maintenance_time_ms,
            )

            logger.info(
                "🔧 Scheduled Maintenance abgeschlossen in %dms: %s",
                maintenance_time_ms,
                ", ".join(maintenance_actions),
            )

            return {
                "status": "success",
                "actions_performed": maintenance_actions,
                "processing_time_ms": maintenance_time_ms,
                "health_after_maintenance": self.get_cache_health_report(),
            }

        except Exception as e:
            logger.error("❌ Scheduled Maintenance fehlgeschlagen: %s", e)
            return {"status": "error", "message": str(e)}

    async def run_automated_invalidation_check(self) -> dict[str, Any]:
        """
        🤖 Automatisierte Invalidation-Prüfung: Alle Strategien durchlaufen.

        Returns:
            Zusammenfassung aller durchgeführten Checks und Invalidations
        """
        start_time = time.time()

        logger.info("🤖 Automatisierte Cache-Invalidation-Prüfung startet...")

        results: dict[str, Any] = {
            "skr03_check": self.check_skr03_updates(),
            "age_invalidation": self.invalidate_cache_by_age(),
            "quality_invalidation": self.invalidate_cache_by_quality(),
            "total_processing_time_ms": 0,
        }

        # SKR03-basierte Invalidation bei Bedarf
        skr03_check = results["skr03_check"]
        if isinstance(skr03_check, dict) and skr03_check.get(
            "invalidation_recommended"
        ):
            logger.warning(
                "🚨 SKR03-Update detektiert - Vollständige Cache-Invalidation wird ausgeführt"
            )
            results["skr03_full_invalidation"] = self.force_full_invalidation(
                "skr03_update_detected"
            )
            self.tracking_metrics["skr03_triggered_invalidations"] += 1

        results["total_processing_time_ms"] = int((time.time() - start_time) * 1000)

        # Zusammenfassung
        def safe_get_int(result: Any, key: str, default: int = 0) -> int:
            """Type-sichere Extraktion von Integer-Werten aus Dictionaries."""
            if isinstance(result, dict) and key in result:
                value: Any = result[key]
                return int(value) if isinstance(value, int | float | str) else default
            return default

        age_invalidated = safe_get_int(
            results["age_invalidation"], "invalidated_entries"
        )
        quality_invalidated = safe_get_int(
            results["quality_invalidation"], "invalidated_entries"
        )
        skr03_invalidated = safe_get_int(
            results.get("skr03_full_invalidation", {}), "invalidated_entries"
        )

        total_invalidated = age_invalidated + quality_invalidated + skr03_invalidated

        logger.info(
            "✅ Automatisierte Cache-Invalidation abgeschlossen: %d Einträge invalidiert in %dms",
            total_invalidated,
            results["total_processing_time_ms"],
        )

        return results


# CLI-Integration und Testing
if __name__ == "__main__":
    import asyncio

    async def test_invalidation_manager() -> None:
        """Test der Cache-Invalidation-Features."""
        manager = CacheInvalidationManager()

        print("🔄 Cache-Invalidation-Manager Test startet...")

        # 1. SKR03-Update-Check
        print("\n1. SKR03-Update-Check:")
        skr03_result = manager.check_skr03_updates()
        print(f"   Status: {skr03_result['status']}")

        # 2. Quality-based Invalidation
        print("\n2. Quality-based Invalidation:")
        quality_result = manager.invalidate_cache_by_quality()
        print(
            f"   Invalidierte Einträge: {quality_result.get('invalidated_entries', 0)}"
        )

        # 3. Age-based Invalidation
        print("\n3. Age-based Invalidation:")
        age_result = manager.invalidate_cache_by_age(30)  # 30 Tage
        print(f"   Invalidierte Einträge: {age_result.get('invalidated_entries', 0)}")

        # 4. Statistiken
        print("\n4. Invalidation-Statistiken:")
        stats = manager.get_invalidation_statistics()
        print(
            f"   Gesamte Invalidations: {stats['tracking_metrics']['total_invalidations']}"
        )

        # 5. Automatisierte Prüfung
        print("\n5. Automatisierte Invalidation-Prüfung:")
        auto_result = await manager.run_automated_invalidation_check()
        print(f"   Gesamtzeit: {auto_result['total_processing_time_ms']}ms")

        print("\n✅ Cache-Invalidation-Manager Test abgeschlossen!")

    # Test ausführen
    asyncio.run(test_invalidation_manager())
