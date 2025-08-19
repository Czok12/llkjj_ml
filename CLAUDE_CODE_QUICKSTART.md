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
