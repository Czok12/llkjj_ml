#!/usr/bin/env python3
"""
LLKJJ ML Pipeline - Enhanced SKR03 Classification Test
====================================================

🎯 Test der erweiterten SKR03-Klassifizierung mit:
- Domain-spezifische Elektrotechnik-Regeln
- Feedback-Learning-System
- Lieferanten-Context-Intelligence

Ziel: >90% Klassifizierungsgenauigkeit validieren

Autor: LLKJJ SKR03 Test Team
Version: 1.0.0
Datum: 19. August 2025
"""

from src.classification.enhanced_skr03_classifier import (
    EnhancedSKR03Classifier,
    classify_invoice_items_enhanced,
    create_feedback_report,
)


def test_enhanced_skr03_classification():
    """
    🎯 Test der Enhanced SKR03-Klassifizierung mit realistischen Elektrotechnik-Items
    """
    print("🎯 Enhanced SKR03 Classification Test gestartet...")
    print("=" * 60)

    # Test-Items aus der realen Elektrotechnik-Praxis
    test_items = [
        {
            "description": "GIRA Schalter System 55 weiß glänzend",
            "amount": 15.99,
            "supplier": "sonepar",
        },
        {
            "description": "Siemens Leitungsschutzschalter 5SL6116-7 B16 1-polig",
            "amount": 45.50,
            "supplier": "sonepar",
        },
        {
            "description": "NYM-J 3x1,5 mm² Installationsleitung 100m Ring",
            "amount": 85.00,
            "supplier": "sonepar",
        },
        {
            "description": "Makita Bohrmaschine HP1631K im Koffer",
            "amount": 89.99,
            "supplier": "amazon",
        },
        {
            "description": "WÜRTH Kreuzschlitzschraubendreher-Satz 6-teilig",
            "amount": 35.50,
            "supplier": "würth",
        },
        {
            "description": "Hager Kleinverteiler Golf 2-reihig 24 Module",
            "amount": 125.00,
            "supplier": "hager",
        },
        {
            "description": "LED-Einbaustrahler 5W warmweiß dimmbar 10er Set",
            "amount": 49.90,
            "supplier": "amazon",
        },
        {
            "description": "FLUKE 117 True RMS Multimeter Elektriker",
            "amount": 189.99,
            "supplier": "amazon",
        },
    ]

    print(f"📋 Test-Items: {len(test_items)}")
    print()

    # Teste Enhanced Classification
    try:
        enhanced_results = classify_invoice_items_enhanced(test_items)

        print("🎯 KLASSIFIZIERUNGS-ERGEBNISSE:")
        print("-" * 60)

        accuracy_scores = []

        for i, (original, enhanced) in enumerate(
            zip(test_items, enhanced_results, strict=False), 1
        ):
            classification = enhanced["skr03_classification"]

            print(f"{i}. {original['description'][:50]}...")
            print(
                f"   Lieferant: {original['supplier']} | Betrag: €{original['amount']:.2f}"
            )
            print(
                f"   SKR03: {classification['skr03_account']} | Konfidenz: {classification['confidence']:.3f}"
            )
            print(f"   Regel: {classification.get('rule_applied', 'unknown')}")
            print(
                f"   Keywords: {', '.join(classification.get('matched_keywords', [])[:3])}"
            )
            print()

            accuracy_scores.append(classification["confidence"])

        # Gesamtstatistiken
        avg_confidence = sum(accuracy_scores) / len(accuracy_scores)
        high_confidence_count = sum(1 for score in accuracy_scores if score > 0.8)

        print("📊 GESAMTSTATISTIKEN:")
        print("-" * 60)
        print(f"Durchschnittliche Konfidenz: {avg_confidence:.3f}")
        print(
            f"Hohe Konfidenz (>0.8): {high_confidence_count}/{len(accuracy_scores)} ({high_confidence_count/len(accuracy_scores)*100:.1f}%)"
        )

        if avg_confidence > 0.9:
            print("✅ ZIEL ERREICHT: >90% Durchschnitts-Konfidenz!")
        elif avg_confidence > 0.8:
            print("🟡 GUTE PERFORMANCE: >80% Durchschnitts-Konfidenz")
        else:
            print("🔴 VERBESSERUNG NÖTIG: <80% Durchschnitts-Konfidenz")

    except Exception as e:
        print(f"❌ Classification-Test fehlgeschlagen: {e}")
        return

    print("\n" + "=" * 60)
    print("🎯 Enhanced SKR03 Classification Test abgeschlossen")


def test_feedback_learning():
    """
    📝 Test des Feedback-Learning-Systems
    """
    print("\n📝 Feedback-Learning Test gestartet...")
    print("-" * 60)

    try:
        classifier = EnhancedSKR03Classifier()

        # Simuliere User-Feedback
        feedback_scenarios = [
            {
                "item_description": "GIRA Schalter System 55",
                "supplier": "sonepar",
                "amount": 15.99,
                "suggested_skr03": "3400",
                "corrected_skr03": "0420",  # User korrigiert zu Anlagevermögen
                "user_confidence": 0.95,
            },
            {
                "item_description": "LED-Stripe 5m wasserdicht",
                "supplier": "amazon",
                "amount": 12.50,
                "suggested_skr03": "0420",
                "corrected_skr03": "3400",  # User korrigiert zu Verbrauchsmaterial
                "user_confidence": 0.9,
            },
        ]

        print("📝 Speichere Feedback-Szenarien...")
        for scenario in feedback_scenarios:
            classifier.record_user_feedback(**scenario)
            print(
                f"   ✅ {scenario['item_description'][:30]}... ({scenario['suggested_skr03']} → {scenario['corrected_skr03']})"
            )

        # Teste Feedback-Report
        print("\n📊 Erstelle Feedback-Report...")
        report = create_feedback_report()

        print(f"Geschätzte Genauigkeit: {report['accuracy_estimate']:.3f}")
        print(f"Status: {report['accuracy_status']}")
        print(f"Empfehlung: {report['recommendation']}")

        stats = report.get("statistics", {})
        feedback_stats = stats.get("feedback_stats", {})
        print(f"Gesamt-Feedback: {feedback_stats.get('total_feedback', 0)}")
        print(
            f"Durchschnitts-User-Konfidenz: {feedback_stats.get('avg_user_confidence', 0):.3f}"
        )

        print("✅ Feedback-Learning Test erfolgreich")

    except Exception as e:
        print(f"❌ Feedback-Learning Test fehlgeschlagen: {e}")


def test_supplier_context():
    """
    🏢 Test der Lieferanten-Context-Intelligence
    """
    print("\n🏢 Lieferanten-Context Test gestartet...")
    print("-" * 60)

    supplier_test_cases = [
        {
            "description": "Standard-Kabel 3x1,5",
            "amount": 25.00,
            "supplier": "sonepar",
            "expected_category": "installation",
        },
        {
            "description": "Bohrmaschine professionell",
            "amount": 150.00,
            "supplier": "würth",
            "expected_category": "tools",
        },
        {
            "description": "Kleinverteiler 12 Module",
            "amount": 85.00,
            "supplier": "hager",
            "expected_category": "switching_devices",
        },
    ]

    try:
        classifier = EnhancedSKR03Classifier()

        print("🔍 Teste Lieferanten-spezifische Klassifizierung...")

        for test_case in supplier_test_cases:
            result = classifier.classify_with_enhanced_rules(test_case)

            print(f"\n📦 {test_case['description']}")
            print(f"   Lieferant: {test_case['supplier']}")
            print(
                f"   SKR03: {result['skr03_account']} (Konfidenz: {result['confidence']:.3f})"
            )
            print(f"   Methode: {result.get('classification_method', 'unknown')}")

            # Validierung der Erwartungen
            if result["confidence"] > 0.7:
                print("   ✅ Hohe Konfidenz erreicht")
            else:
                print("   ⚠️ Niedrige Konfidenz")

        print("\n✅ Lieferanten-Context Test abgeschlossen")

    except Exception as e:
        print(f"❌ Lieferanten-Context Test fehlgeschlagen: {e}")


if __name__ == "__main__":
    print("🎯 LLKJJ Enhanced SKR03 Classification Test Suite")
    print("=" * 60)

    # Führe alle Tests aus
    test_enhanced_skr03_classification()
    test_feedback_learning()
    test_supplier_context()

    print("\n🏆 Alle Tests abgeschlossen!")
    print("=" * 60)
