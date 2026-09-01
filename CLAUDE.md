@AGENTS.md

<!-- Claude-specific additions only below this line -->

- Scoped rules in `.claude/rules/` load automatically when you touch the files
  they cover (ML pipeline, HTTP layer).
- `.claude/settings.json` pre-approves the test/typecheck loop and denies
  `sudo`, `systemctl`, force-push, and `stop_workers.sh`. A denied command is a
  deliberate boundary, not an obstacle to work around.
- The `restart-workers` project skill documents the only safe way to restart the
  RQ workers without cold-killing an in-flight transcode.
