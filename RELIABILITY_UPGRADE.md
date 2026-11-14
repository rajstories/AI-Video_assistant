# AI Video Assistant — Reliability upgrade (draft)

This branch is a small, reviewable first step toward an evidence-backed meeting intelligence platform. It does **not** implement the complete roadmap yet.

## Changes in this PR
- **Per-analysis Chroma collection:** each run builds a fresh non-persistent index rather than reusing one shared on-disk `meeting_transcript` collection. This prevents accidental retrieval from another analysis in the single-session demo. It intentionally sacrifices cross-restart persistence until authenticated, date-scoped meeting memory is implemented.
- **Read-only RAG prompt hardening:** retrieved transcript excerpts are marked as untrusted data. The global `load_rag_chain()` entry point is intentionally disabled because it cannot safely select a meeting. This is not a complete prompt-injection solution.
- **HTML escaping in Streamlit:** transcript, generated summaries, extracted actions, decisions, and chat are escaped before insertion into `unsafe_allow_html=True` templates.
- **Whisper segment filtering:** disable previous-text conditioning and drop likely silence hallucinations or repetitive outputs. This is only a conservative first filter; we still need real VAD and quiet-speech regression tests.
- **Initial tests:** scoped collection names, empty-index handling, explicit-store requirements, context marking, and blank questions.

## How to review locally
```bash
pip install -r Requirements.txt
pip install pytest
pytest -q tests/test_meeting_scope.py
streamlit run app.py
```

This test suite uses mocks but imports the project stack; model/API calls are not required for unit tests. No end-to-end inference or cost benchmark has been run as part of these GitHub edits.

## Remaining priorities
1. Speech-aware chunking with Silero VAD and realistic quiet-speech/WER measurements.
2. Authentication and tenant/date filtered persistent memory with source timestamps.
3. Evidence-verified Pydantic schemas for action owner and deadline.
4. Retry-safe background jobs, deduplication, load testing, and AWS demonstration.
5. Unit tests for prompt injection, privacy, and malformed media.

## Known limitations
- New per-session vector indexes are ephemeral; persistent cross-meeting query is not implemented.
- The Streamlit app is not suitable for public multi-tenant deployment.
- No file-level limits, SSRF protection, PII redaction, or cleanup lifecycle have been implemented in this PR.
- Whisper thresholds are heuristic; evaluate possible loss of quiet speech before claiming improvement.
