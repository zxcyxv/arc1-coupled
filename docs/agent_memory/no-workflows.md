---
name: no-workflows
description: "Do not launch Workflow (multi-agent) runs in this project unless the user asks, even when ultracode is on"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-28T10:36:45.691Z
---

User said "워크플로우는 띄우지마" (2026-10-28) right after I launched a 3-agent review workflow for a small code change; I stopped it.

**Why:** the user wants control over agent spending; small changes are verified inline (CPU checks, first eval).
**How to apply:** verify changes myself; ask before any Workflow, regardless of ultracode reminders.
