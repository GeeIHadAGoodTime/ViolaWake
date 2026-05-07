@echo off
REM Wrapper for backup_postgres.sh, suitable for Windows Task Scheduler.
REM Avoids the nested-quoting headache of calling bash directly from schtasks.
"C:\Program Files\Git\usr\bin\bash.exe" -c "cd /j/CLAUDE/PROJECTS/Wakeword && bash scripts/backup_postgres.sh > /tmp/violawake_backup.log 2>&1"
