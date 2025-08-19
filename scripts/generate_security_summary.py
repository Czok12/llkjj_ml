#!/usr/bin/env python3
"""
LLKJJ ML Security Summary Generator
==================================

Generiert umfassende Security-Reports aus verschiedenen Scanner-Outputs.
Konsolidiert Bandit, Safety, und Custom-Audit-Ergebnisse in einen übersichtlichen Bericht.

Usage:
    python scripts/generate_security_summary.py <security_reports_directory>
"""

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class SecuritySummaryGenerator:
    """Generiert umfassende Security-Zusammenfassungen aus verschiedenen Quellen."""

    def __init__(self, reports_dir: str):
        self.reports_dir = Path(reports_dir)
        self.summary = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "reports_directory": str(self.reports_dir),
                "summary_version": "1.0.0",
            },
            "bandit_findings": [],
            "safety_findings": [],
            "custom_audit_findings": [],
            "ruff_findings": [],
            "mypy_findings": [],
            "overall_risk_assessment": "UNKNOWN",
            "recommendations": [],
            "statistics": {
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0,
                "info_issues": 0,
            },
        }

    def process_bandit_reports(self) -> None:
        """Verarbeitet alle Bandit-Reports im Directory."""
        bandit_files = glob.glob(str(self.reports_dir / "bandit_report_*.json"))

        for file_path in bandit_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    bandit_data = json.load(f)

                if "results" in bandit_data:
                    for result in bandit_data["results"]:
                        finding = {
                            "source": "bandit",
                            "file": file_path,
                            "test_id": result.get("test_id", "unknown"),
                            "test_name": result.get("test_name", "unknown"),
                            "filename": result.get("filename", "unknown"),
                            "line_number": result.get("line_number", 0),
                            "issue_severity": result.get("issue_severity", "unknown"),
                            "issue_confidence": result.get(
                                "issue_confidence", "unknown"
                            ),
                            "issue_text": result.get("issue_text", ""),
                            "code": result.get("code", ""),
                        }
                        self.summary["bandit_findings"].append(finding)
                        self._update_statistics(result.get("issue_severity", "unknown"))

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(
                    f"⚠️ Warnung: Kann Bandit-Report nicht verarbeiten: {file_path} - {e}"
                )

    def process_safety_reports(self) -> None:
        """Verarbeitet alle Safety-Reports im Directory."""
        safety_files = glob.glob(str(self.reports_dir / "safety_report_*.json"))

        for file_path in safety_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    safety_data = json.load(f)

                # Safety kann verschiedene Formate haben
                if isinstance(safety_data, list):
                    vulnerabilities = safety_data
                elif "vulnerabilities" in safety_data:
                    vulnerabilities = safety_data["vulnerabilities"]
                else:
                    continue

                for vuln in vulnerabilities:
                    finding = {
                        "source": "safety",
                        "file": file_path,
                        "package": vuln.get("package_name", "unknown"),
                        "installed_version": vuln.get("installed_version", "unknown"),
                        "affected_versions": vuln.get("affected_versions", "unknown"),
                        "vulnerability_id": vuln.get("vulnerability_id", "unknown"),
                        "advisory": vuln.get("advisory", ""),
                        "cve": vuln.get("cve", ""),
                        "severity": vuln.get("severity", "unknown"),
                    }
                    self.summary["safety_findings"].append(finding)
                    self._update_statistics(vuln.get("severity", "unknown"))

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(
                    f"⚠️ Warnung: Kann Safety-Report nicht verarbeiten: {file_path} - {e}"
                )

    def process_custom_audit_reports(self) -> None:
        """Verarbeitet alle Custom-Audit-Reports im Directory."""
        custom_files = glob.glob(
            str(self.reports_dir / "custom_security_report_*.json")
        )

        for file_path in custom_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    custom_data = json.load(f)

                if "findings" in custom_data:
                    for finding in custom_data["findings"]:
                        audit_finding = {
                            "source": "custom_audit",
                            "file": file_path,
                            "category": finding.get("category", "unknown"),
                            "description": finding.get("description", ""),
                            "severity": finding.get("severity", "unknown"),
                            "confidence": finding.get("confidence", "unknown"),
                            "location": finding.get("location", ""),
                            "recommendation": finding.get("recommendation", ""),
                        }
                        self.summary["custom_audit_findings"].append(audit_finding)
                        self._update_statistics(finding.get("severity", "unknown"))

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(
                    f"⚠️ Warnung: Kann Custom-Audit-Report nicht verarbeiten: {file_path} - {e}"
                )

    def process_ruff_reports(self) -> None:
        """Verarbeitet alle Ruff-Reports im Directory."""
        ruff_files = glob.glob(str(self.reports_dir / "ruff_security_*.json"))

        for file_path in ruff_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    ruff_data = json.load(f)

                for finding in ruff_data:
                    ruff_finding = {
                        "source": "ruff",
                        "file": file_path,
                        "filename": finding.get("filename", "unknown"),
                        "row": finding.get("row", 0),
                        "column": finding.get("column", 0),
                        "code": finding.get("code", "unknown"),
                        "message": finding.get("message", ""),
                        "severity": self._ruff_to_severity(finding.get("code", "")),
                    }
                    self.summary["ruff_findings"].append(ruff_finding)
                    self._update_statistics(ruff_finding["severity"])

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(
                    f"⚠️ Warnung: Kann Ruff-Report nicht verarbeiten: {file_path} - {e}"
                )

    def _ruff_to_severity(self, code: str) -> str:
        """Mappt Ruff-Codes zu Severity-Levels."""
        security_codes = {
            "S": "HIGH",  # Security
            "B": "MEDIUM",  # Bandit-ähnlich
            "E": "LOW",  # Pycodestyle errors
            "W": "LOW",  # Pycodestyle warnings
            "F": "MEDIUM",  # Pyflakes
            "C": "LOW",  # Complexity
            "N": "INFO",  # Naming
        }

        if code and len(code) > 0:
            return security_codes.get(code[0], "INFO")
        return "INFO"

    def _update_statistics(self, severity: str) -> None:
        """Aktualisiert die Statistiken basierend auf Severity."""
        severity_upper = severity.upper()
        self.summary["statistics"]["total_issues"] += 1

        if severity_upper in ["CRITICAL", "FATAL"]:
            self.summary["statistics"]["critical_issues"] += 1
        elif severity_upper == "HIGH":
            self.summary["statistics"]["high_issues"] += 1
        elif severity_upper == "MEDIUM":
            self.summary["statistics"]["medium_issues"] += 1
        elif severity_upper == "LOW":
            self.summary["statistics"]["low_issues"] += 1
        else:
            self.summary["statistics"]["info_issues"] += 1

    def assess_overall_risk(self) -> None:
        """Bewertet das Gesamtrisiko basierend auf gefundenen Issues."""
        stats = self.summary["statistics"]

        if stats["critical_issues"] > 0:
            self.summary["overall_risk_assessment"] = "CRITICAL"
        elif stats["high_issues"] > 10:
            self.summary["overall_risk_assessment"] = "HIGH"
        elif stats["high_issues"] > 5 or stats["medium_issues"] > 20:
            self.summary["overall_risk_assessment"] = "MEDIUM"
        elif stats["medium_issues"] > 0 or stats["low_issues"] > 10:
            self.summary["overall_risk_assessment"] = "LOW"
        else:
            self.summary["overall_risk_assessment"] = "MINIMAL"

    def generate_recommendations(self) -> None:
        """Generiert Empfehlungen basierend auf gefundenen Issues."""
        recommendations = []
        stats = self.summary["statistics"]

        if stats["critical_issues"] > 0:
            recommendations.append(
                {
                    "priority": "IMMEDIATE",
                    "action": "Address all critical security vulnerabilities immediately",
                    "details": f"{stats['critical_issues']} critical issues found that require immediate attention",
                }
            )

        if stats["high_issues"] > 5:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "action": "Review and fix high-severity security issues",
                    "details": f"{stats['high_issues']} high-severity issues found",
                }
            )

        if len(self.summary["safety_findings"]) > 0:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "action": "Update vulnerable dependencies",
                    "details": f"{len(self.summary['safety_findings'])} vulnerable dependencies detected",
                }
            )

        if stats["medium_issues"] > 15:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "action": "Code review for medium-severity issues",
                    "details": f"{stats['medium_issues']} medium-severity issues need review",
                }
            )

        # Standard-Empfehlungen
        recommendations.extend(
            [
                {
                    "priority": "ONGOING",
                    "action": "Implement automated security scanning in CI/CD",
                    "details": "Ensure security scans run on every commit and pull request",
                },
                {
                    "priority": "ONGOING",
                    "action": "Regular security dependency updates",
                    "details": "Schedule weekly dependency vulnerability checks",
                },
                {
                    "priority": "ONGOING",
                    "action": "Security training for development team",
                    "details": "Ensure team understands secure coding practices",
                },
            ]
        )

        self.summary["recommendations"] = recommendations

    def generate_comprehensive_report(self) -> dict[str, Any]:
        """Generiert den umfassenden Security-Report."""
        print("🔍 Verarbeite Bandit-Reports...")
        self.process_bandit_reports()

        print("🔍 Verarbeite Safety-Reports...")
        self.process_safety_reports()

        print("🔍 Verarbeite Custom-Audit-Reports...")
        self.process_custom_audit_reports()

        print("🔍 Verarbeite Ruff-Reports...")
        self.process_ruff_reports()

        print("📊 Bewerte Gesamtrisiko...")
        self.assess_overall_risk()

        print("💡 Generiere Empfehlungen...")
        self.generate_recommendations()

        return self.summary

    def print_summary_stats(self) -> None:
        """Druckt Zusammenfassungsstatistiken in die Konsole."""
        stats = self.summary["statistics"]
        print("\n" + "=" * 60)
        print("🛡️  LLKJJ ML SECURITY SUMMARY")
        print("=" * 60)
        print(f"📊 Total Issues Found: {stats['total_issues']}")
        print(f"🚨 Critical Issues: {stats['critical_issues']}")
        print(f"🔴 High Issues: {stats['high_issues']}")
        print(f"🟡 Medium Issues: {stats['medium_issues']}")
        print(f"🟢 Low Issues: {stats['low_issues']}")
        print(f"ℹ️  Info Issues: {stats['info_issues']}")
        print(f"⚠️  Overall Risk: {self.summary['overall_risk_assessment']}")
        print(f"📝 Bandit Findings: {len(self.summary['bandit_findings'])}")
        print(f"🛡️  Safety Findings: {len(self.summary['safety_findings'])}")
        print(f"🔐 Custom Audit Findings: {len(self.summary['custom_audit_findings'])}")
        print(f"📋 Ruff Findings: {len(self.summary['ruff_findings'])}")
        print("=" * 60)

        if self.summary["recommendations"]:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(self.summary["recommendations"][:5], 1):
                print(f"{i}. [{rec['priority']}] {rec['action']}")
                print(f"   {rec['details']}")

        print("\n" + "=" * 60)


def main():
    """Main-Funktion für Command-Line-Ausführung."""
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/generate_security_summary.py <security_reports_directory>"
        )
        sys.exit(1)

    reports_dir = sys.argv[1]

    if not os.path.exists(reports_dir):
        print(f"❌ Fehler: Directory '{reports_dir}' existiert nicht!")
        sys.exit(1)

    print(f"🔍 Analysiere Security-Reports in: {reports_dir}")

    generator = SecuritySummaryGenerator(reports_dir)
    summary = generator.generate_comprehensive_report()

    # Ausgabe des JSON-Reports
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Zusätzliche Konsolen-Statistiken
    generator.print_summary_stats()


if __name__ == "__main__":
    main()
