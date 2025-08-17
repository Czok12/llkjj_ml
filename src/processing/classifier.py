#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Classification Module
==========================================

This module handles all classification functionality:
- SKR03 classification with RAG system
- Rule-based classification with German keywords
- Vector similarity matching
- Classification result combination
- Account validation

Extracted from unified processor following KISS modularization principles.

Author: LLKJJ ML Pipeline
Version: 1.0.0 (Post-Modularization)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DataClassifier:
    """
    Handles all classification functionality for German electrical contractor invoices.

    Responsibilities:
    - SKR03 classification with RAG system integration
    - Rule-based classification with keyword matching
    - Vector similarity matching using ChromaDB
    - Intelligent combination of classification results
    - Account validation against SKR03 chart
    """

    def __init__(
        self,
        skr03_manager: Any = None,
        vector_store: Any = None,
        skr03_regeln: dict[str, Any] | None = None,
    ) -> None:
        """Initialize data classifier with dependencies"""
        self.skr03_manager = skr03_manager
        self.vector_store = vector_store
        self.skr03_regeln = skr03_regeln or {}
        logger.info("DataClassifier initialized with German SKR03 optimization")
        logger.debug(
            "DataClassifier vector_store parameter: %s (type: %s)",
            self.vector_store,
            type(self.vector_store),
        )

    def process_classifications(
        self, line_items: list[dict[str, Any]], structured_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Hauptmethode für die Klassifizierung von Rechnungspositionen.

        Args:
            line_items: Liste der extrahierten Rechnungspositionen
            structured_data: Strukturierte Rechnungsdaten

        Returns:
            Liste der klassifizierten Positionen mit SKR03-Konten
        """
        try:
            logger.info("🔍 Starte Klassifizierung von %d Positionen", len(line_items))

            # Führe SKR03-Klassifizierung durch
            classifications = self.classify_skr03(structured_data)

            logger.info(
                "✅ Klassifizierung abgeschlossen: %d Positionen klassifiziert",
                len(classifications),
            )

            return classifications

        except Exception as e:
            logger.error("❌ Fehler bei Klassifizierung: %s", e)
            raise

    def classify_skr03(self, structured_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Erweiterte SKR03-Klassifizierung mit RAG-System für intelligente Artikel-Klassifizierung.

        Kombiniert:
        1. Regelbasierte Klassifizierung (bestehende Keywords)
        2. RAG-basierte Klassifizierung (ähnliche validierte Buchungsbeispiele)
        3. Intelligente Konfidenz-Bewertung
        """
        classifications = []
        line_items = structured_data.get("line_items", [])

        logger.info(f"🏷️ Klassifiziere {len(line_items)} Positionen mit RAG-System")

        for i, item in enumerate(line_items):
            description = item.get("description", "")

            # Schritt 1: Regelbasierte Klassifizierung (bestehend)
            rule_match = self.find_best_skr03_match(description)

            # Schritt 2: RAG-basierte Klassifizierung (NEU)
            rag_match = self.classify_with_rag_system(description)

            # Schritt 3: Intelligente Kombination der Ergebnisse
            final_classification = self.combine_classification_results(
                description, rule_match, rag_match
            )

            classification = {
                "position": i + 1,
                "description": description,
                "skr03_konto": final_classification["konto"],
                "category": final_classification["category"],
                "confidence": final_classification["confidence"],
                "reasoning": final_classification["reasoning"],
                "amount": item.get("total_price", "0"),
                "quantity": item.get("quantity", "1"),
                # Erweiterte RAG-Informationen
                "rule_based_confidence": rule_match["confidence"],
                "rag_based_confidence": rag_match["confidence"],
                "similar_articles": rag_match.get("similar_articles", []),
                "classification_method": final_classification["method"],
                "supplier_detected": rag_match.get("supplier", "Unknown"),
            }

            logger.info(
                f"📊 Position {i+1}: '{description[:50]}...' -> "
                f"SKR03 {final_classification['konto']} "
                f"({final_classification['category']}, "
                f"{final_classification['confidence']:.1%} via {final_classification['method']})"
            )

            classifications.append(classification)

        return classifications

    def classify_with_rag_system(self, description: str) -> dict[str, Any]:
        """
        RAG-basierte Klassifizierung: Nutzt ähnliche validierte Buchungsbeispiele
        für intelligente Klassifizierung neuer oder unbekannter Artikel.
        """
        # Suche ähnliche validierte Buchungsbeispiele
        similar_bookings = self.find_similar_bookings(
            description=description,
            n_results=5,
            similarity_threshold=0.3,  # Lower threshold for better recall
        )

        if not similar_bookings:
            return {
                "konto": "3400",  # Fallback
                "category": "Unbekannt",
                "confidence": 0.1,
                "reasoning": "Keine ähnlichen Buchungsbeispiele gefunden",
                "method": "fallback",
                "similar_articles": [],
            }

        # Gewichtete Bewertung basierend auf Ähnlichkeit und Konfidenz
        weighted_votes = {}
        total_weight = 0.0

        for booking in similar_bookings:
            skr03_account = booking["skr03_account"]
            weight = booking["similarity"] * booking["confidence"]

            if skr03_account not in weighted_votes:
                weighted_votes[skr03_account] = {
                    "weight": 0.0,
                    "category": booking["category"],
                    "examples": [],
                }

            weighted_votes[skr03_account]["weight"] += weight
            weighted_votes[skr03_account]["examples"].append(
                {
                    "description": booking["description"],
                    "similarity": booking["similarity"],
                    "supplier": booking["supplier"],
                }
            )
            total_weight += weight

        # Finde bestes Ergebnis
        if not weighted_votes:
            return {
                "konto": "3400",
                "category": "Unbekannt",
                "confidence": 0.1,
                "reasoning": "Keine gewichteten Stimmen gefunden",
                "method": "fallback",
                "similar_articles": [],
            }

        best_account = max(
            weighted_votes.keys(), key=lambda k: weighted_votes[k]["weight"]
        )
        best_info = weighted_votes[best_account]

        # Berechne RAG-Konfidenz
        rag_confidence = min(0.95, best_info["weight"] / total_weight)

        # Erkenne Lieferant aus ähnlichen Beispielen
        suppliers = [ex["supplier"] for ex in best_info["examples"]]
        most_common_supplier = (
            max(set(suppliers), key=suppliers.count) if suppliers else "Unknown"
        )

        return {
            "konto": best_account,
            "category": best_info["category"],
            "confidence": rag_confidence,
            "reasoning": f"RAG: {len(similar_bookings)} ähnliche Artikel, stärkste Übereinstimmung mit {best_info['category']}",
            "method": "rag_similarity",
            "similar_articles": similar_bookings[:3],  # Top 3 für Debugging
            "supplier": most_common_supplier,
        }

    def combine_classification_results(
        self, description: str, rule_match: dict[str, Any], rag_match: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Intelligente Kombination von regelbasierter und RAG-basierter Klassifizierung.

        Strategie:
        - Bei hoher Regel-Konfidenz: Bevorzuge Regeln (bewährte Keywords)
        - Bei niedriger Regel-Konfidenz: Bevorzuge RAG (ähnliche Beispiele)
        - Bei Konflikt: Wähle höhere Konfidenz, aber protokolliere beide
        """
        rule_confidence = rule_match["confidence"]
        rag_confidence = rag_match["confidence"]

        # Schwellenwerte für Entscheidungslogik
        HIGH_CONFIDENCE_THRESHOLD = 0.7
        MEDIUM_CONFIDENCE_THRESHOLD = 0.4

        # Szenario 1: Beide niedrige Konfidenz -> Regelbasiert (konservativer Ansatz)
        if (
            rule_confidence < MEDIUM_CONFIDENCE_THRESHOLD
            and rag_confidence < MEDIUM_CONFIDENCE_THRESHOLD
        ):
            return {
                "konto": rule_match.get("account", rule_match.get("konto", "3400")),
                "category": rule_match["category"],
                "confidence": max(rule_confidence, rag_confidence),
                "reasoning": f"Niedrige Konfidenz beider Methoden. Regel: {rule_confidence:.2f}, RAG: {rag_confidence:.2f}. Standardklassifizierung gewählt.",
                "method": "rule_fallback",
            }

        # Szenario 2: Regelbasiert hat hohe Konfidenz -> Bevorzuge Regeln
        if rule_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return {
                "konto": rule_match.get("account", rule_match.get("konto", "3400")),
                "category": rule_match["category"],
                "confidence": rule_confidence,
                "reasoning": f"Starke Keywords gefunden: {rule_match['reasoning']}",
                "method": "rule_dominant",
            }

        # Szenario 3: RAG hat deutlich höhere Konfidenz -> Bevorzuge RAG
        if rag_confidence > rule_confidence + 0.2:
            return {
                "konto": rag_match.get("account", rag_match.get("konto", "3400")),
                "category": rag_match["category"],
                "confidence": rag_confidence,
                "reasoning": f"RAG-System überzeugt: {rag_match['reasoning']}",
                "method": "rag_dominant",
            }

        # Szenario 4: Beide haben ähnliche mittlere Konfidenz -> Hybrid
        if abs(rule_confidence - rag_confidence) <= 0.2:
            # Prüfe ob beide zur gleichen Klassifizierung kommen
            regel_konto = rule_match.get("account", rule_match.get("konto", "3400"))
            rag_konto = rag_match.get("account", rag_match.get("konto", "3400"))
            if regel_konto == rag_konto:
                hybrid_confidence = (rule_confidence + rag_confidence) / 2
                return {
                    "konto": regel_konto,
                    "category": rule_match["category"],
                    "confidence": hybrid_confidence,
                    "reasoning": f"Hybrid: Beide Methoden stimmen überein (Regel: {rule_confidence:.2f}, RAG: {rag_confidence:.2f})",
                    "method": "hybrid_consensus",
                }
            else:
                # Konflikt -> Wähle höhere Konfidenz, aber reduziere sie leicht
                if rule_confidence >= rag_confidence:
                    return {
                        "konto": rule_match["konto"],
                        "category": rule_match["category"],
                        "confidence": rule_confidence
                        * 0.9,  # Leichte Reduktion wegen Konflikt
                        "reasoning": f"Konflikt gelöst: Regel ({rule_confidence:.2f}) vs RAG ({rag_confidence:.2f})",
                        "method": "rule_conflict_resolution",
                    }
                else:
                    return {
                        "konto": rag_match["konto"],
                        "category": rag_match["category"],
                        "confidence": rag_confidence * 0.9,
                        "reasoning": f"Konflikt gelöst: RAG ({rag_confidence:.2f}) vs Regel ({rule_confidence:.2f})",
                        "method": "rag_conflict_resolution",
                    }

        # Fallback: Standardregeln
        return {
            "konto": rule_match.get("konto", "3400"),  # Fallback-Konto
            "category": rule_match.get("category", "Elektromaterial"),
            "confidence": rule_confidence,
            "reasoning": f"Fallback zu Regeln: {rule_match.get('reasoning', 'Standard-Klassifizierung')}",
            "method": "rule_fallback",
        }

    def find_best_skr03_match(self, description: str) -> dict[str, Any]:
        """
        SKR03-Klassifizierung über den neuen SKR03Manager.

        Verwendet saubere Trennung von Regeln (YAML) und Konten (CSV).
        """
        # Verwende den neuen Manager falls verfügbar
        if not self.skr03_manager or not self.skr03_manager.ist_bereit():
            logger.error(
                "SKR03Manager nicht verfügbar. Regelbasierte Klassifizierung stark eingeschränkt."
            )
            # Fallback zu alter Logik, aber mit klarer Fehlermeldung
            return self.find_best_skr03_match_fallback(description)

        kategorie, konto, konfidenz, keywords = (
            self.skr03_manager.klassifiziere_artikel(description)
        )

        # Konvertiere Manager-Ergebnis zu erwartetem Format für Kompatibilität
        return {
            "category": kategorie,
            "konto": konto,  # Verwende "konto" statt "account" für Kompatibilität
            "confidence": konfidenz,
            "matched_keywords": keywords,
            "method": "skr03_manager_v2",
            "reasoning": f"SKR03Manager Klassifizierung basierend auf Keywords: {keywords}",
        }

    def find_best_skr03_match_fallback(self, description: str) -> dict[str, Any]:
        """
        Fallback-Klassifizierung für den Fall dass Manager nicht verfügbar ist.
        """
        best_score = 0.0
        best_category = "elektromaterial"  # Default
        matched_keywords = []

        # Normalisiere Beschreibung für besseres Matching
        description_lower = description.lower()

        for kategorie, regeln in self.skr03_regeln.items():
            score = 0.0
            gefundene_schluesselwoerter = []

            for schluesselwort in regeln["schlüsselwörter"]:
                if schluesselwort.lower() in description_lower:
                    # Gewichte längere Schlüsselwörter höher
                    gewicht = len(schluesselwort) / 10.0 + 1.0
                    score += gewicht
                    gefundene_schluesselwoerter.append(schluesselwort)

            if score > best_score:
                best_score = score
                best_category = kategorie
                matched_keywords = gefundene_schluesselwoerter

        category_info = self.skr03_regeln.get(best_category, {})

        # Erweiterte Konfidenz-Berechnung
        base_confidence = min(0.9, 0.3 + (best_score * 0.1))

        # Bonus für mehrere Keyword-Matches
        keyword_bonus = min(0.2, len(matched_keywords) * 0.05)

        # Bonus für bekannte Elektrotechnik-Marken
        brand_bonus = 0.0
        elektro_marken = [
            "gira",
            "hager",
            "siemens",
            "abb",
            "schneider",
            "wago",
            "phoenix",
        ]
        for marke in elektro_marken:
            if marke in description_lower:
                brand_bonus = 0.15
                break

        final_confidence = min(0.95, base_confidence + keyword_bonus + brand_bonus)

        # Hole Konto aus Kategorie-Info
        konto = category_info.get("konto", "3400") if category_info else "3400"

        # Optional: Validiere Konto gegen Kontenplan
        konto_validierung = self.validiere_konto_zuordnung(konto)

        result = {
            "category": best_category,
            "konto": konto,  # Verwende "konto" für Konsistenz
            "confidence": final_confidence,
            "reasoning": (
                f"Matched keywords: {', '.join(matched_keywords)}"
                if matched_keywords
                else "Default classification"
            ),
            "method": "rule_fallback",
            "validation_info": {
                "konto_gueltig": (
                    bool(konto_validierung) if konto_validierung else False
                ),
                "konto_info": konto_validierung,
            },
        }

        return result

    def validiere_konto_zuordnung(self, kontonummer: str) -> dict[str, Any] | None:
        """
        Validiert eine Kontenzuordnung gegen den vollständigen Kontenplan.

        Returns:
            Validierungsergebnis oder None falls Kontenplan nicht verfügbar
        """
        if not self.skr03_manager or not self.skr03_manager.ist_bereit():
            logger.warning("SKR03Manager nicht verfügbar für Kontovalidierung")
            return None

        ist_gueltig = self.skr03_manager.validiere_konto(kontonummer)
        if ist_gueltig and self.skr03_manager.kontenplan_parser:
            konto_info = self.skr03_manager.kontenplan_parser.get_konto_info(
                kontonummer
            )
            return {"ist_gueltig": True, "konto_info": konto_info}
        else:
            return {"ist_gueltig": False, "konto_info": None}

    def find_similar_bookings(
        self, description: str, n_results: int = 5, similarity_threshold: float = 0.6
    ) -> list[dict[str, Any]]:
        """
        Sucht ähnliche validierte Buchungsbeispiele in der Vektordatenbank.

        Args:
            description: Artikelbeschreibung für die Suche
            n_results: Maximale Anzahl Ergebnisse
            similarity_threshold: Mindest-Ähnlichkeitsschwelle

        Returns:
            Liste ähnlicher Buchungsbeispiele mit Metadaten
        """
        if not self.vector_store:
            logger.warning("Vector store not available for RAG classification")
            return []

        try:
            # Suche ähnliche Dokumente in ChromaDB ohne confidence filter
            # (since some old items don't have confidence field)
            results = self.vector_store.query(
                query_texts=[description],
                n_results=n_results,
                # Removed where clause due to mixed metadata formats
            )

            # Konvertiere ChromaDB-Ergebnisse zu einheitlichem Format
            similar_bookings = []
            if results["documents"] and results["distances"] and results["metadatas"]:
                for _i, (doc, distance, metadata) in enumerate(
                    zip(
                        results["documents"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                        strict=False,
                    )
                ):
                    # Fix similarity calculation for different distance metrics
                    if distance < 2.0:  # Euclidean distance
                        similarity = max(0.0, 1.0 - (distance / 2.0))
                    else:  # Cosine distance
                        similarity = max(0.0, 1.0 - distance)

                    # Check confidence threshold manually for filtering
                    item_confidence = metadata.get("confidence", 0.0)
                    if isinstance(item_confidence, str):
                        try:
                            item_confidence = float(item_confidence)
                        except ValueError:
                            item_confidence = 0.0

                    # Apply similarity threshold and basic confidence check
                    if similarity >= similarity_threshold and item_confidence >= 0.3:
                        # Handle both old and new metadata formats
                        skr03_account = metadata.get("skr03_account") or metadata.get(
                            "skr03_konto", "3400"
                        )

                        similar_bookings.append(
                            {
                                "description": doc,
                                "similarity": similarity,
                                "skr03_account": skr03_account,
                                "category": metadata.get("category", "Unbekannt"),
                                "supplier": metadata.get("supplier", "Unknown"),
                                "confidence": item_confidence,
                                "amount": metadata.get("amount", 0),
                            }
                        )

            logger.debug(
                f"Found {len(similar_bookings)} similar bookings for '{description[:50]}...'"
            )
            return similar_bookings

        except Exception as e:
            logger.error(f"Error in RAG similarity search: {e}")
            return []
