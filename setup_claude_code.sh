#!/bin/bash

# LLKJJ ML Claude Code Setup Script
# Installiert und konfiguriert Claude Code Best Practices für die ML-Pipeline

set -e

echo "🚀 LLKJJ ML Claude Code Setup gestartet..."

# 1. Claude Code Installation überprüfen
echo "📦 Überprüfe Claude Code Installation..."
if ! command -v claude &> /dev/null; then
    echo "⚠️  Claude Code nicht gefunden. Installiere..."
    npm install -g @anthropic-ai/claude-code
else
    echo "✅ Claude Code bereits installiert: $(claude --version)"
fi

# 2. GitHub CLI überprüfen (für GitHub Integration)
echo "🔧 Überprüfe GitHub CLI..."
if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI nicht gefunden. Installation empfohlen für GitHub-Integration."
    echo "   💡 Installation: https://github.com/cli/cli#installation"
else
    echo "✅ GitHub CLI verfügbar: $(gh --version | head -1)"
fi

# 3. Node.js für MCP Servers überprüfen
echo "📊 Überprüfe Node.js für MCP Servers..."
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js nicht gefunden. Benötigt für MCP Servers."
    echo "   💡 Installation: https://nodejs.org/"
else
    echo "✅ Node.js verfügbar: $(node --version)"
fi

# 4. Verzeichnisstruktur erstellen
echo "📁 Erstelle Claude Code Verzeichnisstruktur..."
mkdir -p .claude/{commands,agents,hooks,logs}

# 5. Basis-Konfiguration erstellen
echo "⚙️  Erstelle Basis-Konfiguration..."
cat > .claude/settings.local.json << EOF
{
  "allowedTools": [
    "Edit",
    "Read",
    "Write",
    "Bash(poetry:*)",
    "Bash(git:*)",
    "Bash(pytest:*)",
    "Bash(mypy:*)",
    "Bash(ruff:*)",
    "Bash(black:*)",
    "mcp__*"
  ]
}
EOF

# 6. Hook für Tool-Usage Logging
echo "🪝 Erstelle Tool-Usage Logging Hook..."
cat > .claude/hooks/log-tool-usage.sh << 'EOF'
#!/bin/bash
# Tool Usage Logging Hook für LLKJJ ML Pipeline

LOG_DIR=".claude/logs"
LOG_FILE="$LOG_DIR/tool-usage.log"

# Erstelle Log-Verzeichnis falls nicht vorhanden
mkdir -p "$LOG_DIR"

# Log-Eintrag erstellen
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] Tool used: $CLAUDE_TOOL_NAME with args: $CLAUDE_TOOL_ARGS" >> "$LOG_FILE"

# Bei kritischen Tools zusätzliche Logging
case "$CLAUDE_TOOL_NAME" in
    "Bash")
        echo "[$TIMESTAMP] BASH COMMAND: $CLAUDE_TOOL_ARGS" >> "$LOG_FILE"
        ;;
    "Edit"|"Write")
        echo "[$TIMESTAMP] FILE OPERATION: $CLAUDE_TOOL_NAME -> $CLAUDE_TOOL_ARGS" >> "$LOG_FILE"
        ;;
esac
EOF

chmod +x .claude/hooks/log-tool-usage.sh

# 7. MCP Server Installation (optional, nur wenn Node.js verfügbar)
if command -v node &> /dev/null; then
    echo "🔌 Installiere empfohlene MCP Servers..."

    # Context7 MCP Server für up-to-date Dokumentation
    echo "   📚 Context7 MCP Server..."
    # Note: Diese Installation wird über Claude Code direkt gemacht

    # Filesystem MCP Server für sichere Dateisystem-Operationen
    echo "   📁 Filesystem MCP Server vorbereiten..."
    # Note: Diese Installation wird über Claude Code direkt gemacht

    echo "   💡 MCP Server werden über Claude Code konfiguriert:"
    echo "      claude mcp add context7 -- npx -y @context7/mcp-server"
    echo "      claude mcp add filesystem -- npx -y @anthropic-ai/mcp-server-filesystem $(pwd)"
else
    echo "⚠️  Node.js nicht verfügbar - MCP Server Installation übersprungen"
fi

# 8. Performance Monitoring Script
echo "📊 Erstelle Performance Monitoring Script..."
cat > scripts/monitor_performance.py << 'EOF'
#!/usr/bin/env python3
"""
LLKJJ ML Performance Monitor
Überwacht Pipeline-Performance und erstellt Reports
"""

import time
import psutil
import json
from pathlib import Path
from datetime import datetime

def monitor_memory():
    """Memory Usage Monitoring"""
    process = psutil.Process()
    return {
        'rss': process.memory_info().rss / 1024 / 1024,  # MB
        'vms': process.memory_info().vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def monitor_cpu():
    """CPU Usage Monitoring"""
    return {
        'percent': psutil.cpu_percent(interval=1),
        'cores': psutil.cpu_count()
    }

def create_performance_report():
    """Erstelle Performance Report"""
    timestamp = datetime.now().isoformat()

    report = {
        'timestamp': timestamp,
        'memory': monitor_memory(),
        'cpu': monitor_cpu(),
        'disk_usage': psutil.disk_usage('.').percent
    }

    # Report speichern
    report_file = Path('.claude/logs/performance_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Performance Report erstellt: {report_file}")
    return report

if __name__ == '__main__':
    create_performance_report()
EOF

chmod +x scripts/monitor_performance.py

# 9. Quick Start Guide erstellen
echo "📖 Erstelle Quick Start Guide..."
cat > CLAUDE_CODE_QUICKSTART.md << 'EOF'
# Claude Code Quick Start für LLKJJ ML

## Sofort starten:

1. **Claude Code starten:**
   ```bash
   claude
   ```

2. **Repository analysieren:**
   ```
   /primer
   ```

3. **PDF-Pipeline testen:**
   ```
   /test-pdf-pipeline test_pdfs/Sonepar_test3.pdf
   ```

4. **SKR03-Validation:**
   ```
   /validate-skr03 test_pdfs/Sonepar_test3.pdf
   ```

## Wichtige Commands:

- `/permissions` - Tool-Berechtigunen verwalten
- `/clear` - Context zwischen Tasks leeren
- `/agents` - Subagents verwalten
- `ESC` - Claude unterbrechen
- `ESC ESC` - In History zurückspringen

## MCP Servers installieren:

```bash
claude mcp add context7 -- npx -y @context7/mcp-server
claude mcp add filesystem -- npx -y @anthropic-ai/mcp-server-filesystem $(pwd)
```

## Performance Monitoring:

```bash
python scripts/monitor_performance.py
```

## Logs prüfen:

```bash
tail -f .claude/logs/tool-usage.log
```
EOF

# 10. Abschluss
echo ""
echo "🎉 Claude Code Setup für LLKJJ ML erfolgreich abgeschlossen!"
echo ""
echo "📋 Nächste Schritte:"
echo "   1. claude                    # Claude Code starten"
echo "   2. /primer                   # Repository analysieren"
echo "   3. /test-pdf-pipeline        # Pipeline testen"
echo ""
echo "📚 Dokumentation:"
echo "   • CLAUDE_CODE_BEST_PRACTICES.md  # Vollständige Best Practices"
echo "   • CLAUDE_CODE_QUICKSTART.md      # Quick Start Guide"
echo ""
echo "🔧 Konfiguration:"
echo "   • .claude/settings.local.json    # Tool-Berechtigunen"
echo "   • .claude/commands/              # Custom Commands"
echo "   • .claude/agents/                # Specialized Subagents"
echo ""
echo "🚀 Ready for AI-powered ML development!"
