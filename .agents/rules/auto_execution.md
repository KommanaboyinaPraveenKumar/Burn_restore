# Auto-Execution & Command Formatting Rules

- **Unwrapped Commands:** Never wrap single terminal commands in extra outer quotes (e.g. avoid `powershell -Command "..."` when a flat command works), which triggers Antigravity's security tokenizer.
- **Autonomous Continuation:** Proceed through code edits, diagnostic evaluations, and model benchmark loops autonomously without stopping for manual approvals unless explicitly required.
