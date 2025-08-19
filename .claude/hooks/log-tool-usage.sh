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
