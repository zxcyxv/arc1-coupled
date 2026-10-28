---
name: external-doc-style
description: "Documents the user hands to other researchers must not carry internal provenance notes (e.g. \"no repo record\", \"ran on another instance\")"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ae142cb7-3355-47e0-b126-2adf442a38bb
  modified: 2026-10-26T09:42:20.006Z
---

When writing a document meant for outside readers (e.g. the WRL model explainer for a colleague), state results plainly; do not add bookkeeping caveats such as "저장소 기록은 없다" or "별도 인스턴스에서 돌렸다".

**Why:** the user said on 2026-10-26: "딴 사람한테 문서만 던져줄건데 그런걸 왜 적어". Such notes are for our own records (handoffs, decisions.md), not for readers.
**How to apply:** keep provenance caveats in internal docs only; in external docs keep only what a reader needs (numbers, conditions like single seed).
