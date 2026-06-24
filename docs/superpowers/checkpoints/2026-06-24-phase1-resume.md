# ken-rag — Resume Checkpoint (2026-06-24)

## Where we are
Planning done + approved. Building Phase 1 via subagents. Git repo at project root.

**Plan:** `docs/superpowers/plans/2026-06-24-ken-rag-v1.md` (6 phases, STOP checkpoint per phase).
**Architecture + decisions:** `docs/superpowers/specs/2026-06-24-ken-rag-architecture-draft.md`.
**Reviews passed:** ecc architect, gstack eng review (2 fixes: nomic prefixes, tree-sitter-language-pack; hybrid + distribution into v1), gstack design review (TUI 5→9/10; inline citations, single-stream layout, cyan/teal accent — specced in plan under "TUI / CLI Design Spec").

## Build mode (locked by user)
- Phase-by-phase, STOP for review after each phase.
- Subagents + parallelism. Worktree isolation NOT available here (harness doesn't see the repo) → use **same-dir parallel agents on disjoint dirs, agents do NOT git/reinstall, controller commits per lane**. venv at `.venv` (editable install done; new files auto-picked-up).
- Run tests: `source .venv/Scripts/activate && python -m pytest -q`.
- Commits: conventional, NO Co-Authored-By (attribution disabled).

## DONE (committed) — Phase 1, 207 tests green, ruff clean
- Foundation: scaffold, domain models/enums, 10 protocols, error hierarchy, config (defaults/paths/settings/loader). Reviewed (python-reviewer) + 3 HIGH fixes applied.
- Parallel wave (5 lanes): llm+embedding (nomic prefixes ✓, dim guard, batching), parsing (text/md + detect), prose chunking (+fallback+registry seam for CODE), LanceDB store (schema+FTS+meta+migrations + InMemoryStore fake), generation (prompt+citation dedup+OllamaGenerator + FakeGenerator).
- Hybrid retrieval (DenseStage+KeywordStage+RRF fusion, RetrievalPipeline).
- Fakes ready: fake_embedder (768-dim deterministic), in_memory_store, fake_generator.

Last commit: `7116749 feat: hybrid retrieval`.

## NEXT STEP — finish Phase 1 capstone (Tasks 1.11 + 1.12)
Build (was about to dispatch, interrupted):
- `tracking/hasher.py` (streaming sha256), `tracking/walker.py` (walk path, skip .ken/ .git/) — minimal; full FileTracker/ignore/git = Phase 3.
- `pipeline/ingest.py` (walk→detect→hash→parse→chunk→embed→upsert; first ingest writes `embedder_name`+dim to meta; re-validate every op → EmbedderMismatchError; re-index changed-by-hash files) + `pipeline/query.py` (guard→embed_query→RetrievalPipeline.retrieve→PromptBuilder→generator.stream(num_ctx=settings.num_ctx)→citation.build). Combined <200 lines.
- `cli/context.py` build_context(settings, overrides=) DI root (arch §Composition) — overrides inject fakes.
- `cli/render.py` — DESIGN SPEC: Rich Theme (cyan/teal accent, dim meta, red errors), `format_citation(chunk)`, `ken ask` one-shot layout (answer → blank → dim `① path:lines symbol` block), NO_COLOR + non-TTY plain.
- `cli/errors.py` (KenError→friendly+hint+Exit(1), no traceback), `cli/app.py` (Typer root, --verbose/--db, register add+ask, additive), `commands/add.py`, `commands/ask.py`.
- Tests: integration test_pipeline.py (first-ingest meta write; mismatch error; unchanged re-ingest stable; ask returns Answer+citations); e2e test_cli_add_ask.py (CliRunner + fakes via build_context override; OllamaUnavailableError → exit≠0 + hint, no traceback).

Then: full suite green + ruff → commit per task → **CHECKPOINT 1: STOP for user review** (demo `ken add ./docs && ken ask`).

Remaining phases: 2 code-aware chunking (moat), 3 tracking/incremental/git, 4 setup/model, 5 Textual TUI, 6 search+CI/PyPI+README/demo.

## Resume command
Dispatch one sonnet implementer for Tasks 1.11+1.12 (tightly coupled integration capstone) with the detailed brief; then run full suite, commit, present Checkpoint 1.
