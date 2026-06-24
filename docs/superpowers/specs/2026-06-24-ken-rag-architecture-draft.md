# ken-rag — Architecture & Build Plan

> Status: REVIEWED (ecc architect + gstack eng review, 2026-06-24). Source of truth for goal/scope is `ken-rag-spec.md`. This doc adds the module architecture and phased build plan.

## Eng-review decisions (locked 2026-06-24)
1. **nomic-embed-text task prefixes (CRITICAL).** OllamaEmbedder prepends `search_document: ` to documents and `search_query: ` to queries. Omitting silently wrecks retrieval. nomic via Ollama does NOT auto-prefix. The prefix mode is part of the Embedder contract.
2. **tree-sitter-language-pack, NOT tree-sitter-languages.** The latter is unmaintained (breaks on tree-sitter ≥0.22 / Python 3.13). Use `tree-sitter-language-pack` (maintained, 306 grammars, py≥3.10, lazy grammar load).
3. **Hybrid search IN v1.** Dense + keyword fused via LanceDB built-in FTS (RRF). DenseStage + KeywordStage both ship v1 in the RetrievalPipeline. Reason: code RAG needs exact-identifier match; marginal cost low with LanceDB native FTS.
4. **Distribution IN v1.** GitHub Actions: test matrix (3.10–3.13 × OS) → build → publish to PyPI on tag + GitHub Release. The `uv tool install ken-rag` promise must work at launch.
5. **Embed batching.** `embed_texts` batches (≤64/call) so large-repo ingest isn't one-HTTP-call-per-chunk.

## Goal
Local, private, terminal-native RAG CLI. 5k+ GitHub stars via polish + code-aware-chunking moat + git-awareness + Claude-Code-quality TUI. Tech stack LOCKED (see spec §3). v1 scope LOCKED to 8 commands: `setup, add, ask, chat, list, status, search, model`.

## Module layout (src/ken_rag/)
- `cli/` — Typer command surface (thin). `app.py` root, `context.py` DI root (`build_context`), `render.py` Rich output, `errors.py` exception→message mapping, `commands/{setup,add,ask,chat,list_cmd,status,search,model}.py`.
- `config/` — `settings.py` (frozen Settings), `paths.py` (platformdirs), `defaults.py` (DEFAULT_K=5, NUM_CTX=8192, EMBED_DIM=768), `loader.py` (file+env+CLI merge, boundary validation).
- `domain/` — pure, zero-IO. `models.py` (Chunk, EmbeddedChunk, RetrievedChunk, Citation, Answer, FileRecord, FileChange — all frozen), `enums.py`, `protocols.py` (Parser, Chunker, Embedder, VectorStore, MetadataStore, Retriever, Generator, FileTracker, IgnoreFilter, GitClient), `errors.py` (KenError hierarchy).
- `parsing/` — `registry.py` (FileType→Parser, detect_file_type), `text_parser.py`, `pdf_parser.py` (PyMuPDF), `code_parser.py`.
- `chunking/` — THE MOAT. `registry.py`, `base.py`, `prose_chunker.py` (recursive, header/para-aware, 300-800 tok, 10-15% overlap), `code/{code_chunker,treesitter_backend,python_ast_backend,language_map,subsplit}.py`, `fallback_chunker.py`.
- `embedding/` — `ollama_embedder.py` (batched /api/embed), `dimension.py` (768 guard).
- `store/` — repository pattern. `schema.py` (PyArrow/Lance), `lancedb_store.py`, `metadata_store.py` (embedder_name KV), `migrations.py` (schema_version stamp, open_or_create).
- `tracking/` — `hasher.py` (streaming sha256), `file_tracker.py` (diff→added/modified/deleted/unchanged), `walker.py`, `ignore_filter.py` (pathspec: .gitignore+.kenignore), `git/{git_client,null_git}.py` (shell-out + graceful fallback).
- `retrieval/` — `retriever.py` (VectorRetriever), `pipeline.py` (ordered Stages + RRF fusion), `stages.py` (DenseStage + KeywordStage via LanceDB FTS, both v1; RerankStage v2 extension point), `fusion.py` (reciprocal-rank fusion).
- `generation/` — `ollama_generator.py` (stream, num_ctx≥8192), `prompt.py` (PromptBuilder: system+numbered sources+question), `citation.py` (dedup by file+lines).
- `pipeline/` — hand-written orchestration (anti-LangChain). `ingest.py` (walk→ignore→parse→chunk→embed→upsert lifecycle), `query.py` (retrieve→prompt→stream→cite). Keep combined <200 lines.
- `setup/` — `hardware.py` (detect_ram via psutil), `model_picker.py` (RAM→tier→model), `ollama_health.py` (running/installed/install_hint/pull), `flow.py` (SetupFlow, IO injected).
- `tui/` — Textual. `chat_app.py`, `widgets/{history_pane,input_box,sources_panel,status_bar}.py`, `messages.py`, `stream_worker.py` (QueryPipeline.ask_stream→Textual worker→TokenReceived).
- `llm/ollama_client.py` — low-level shared HTTP client (timeouts, typed errors).

## Key contracts (domain/protocols.py)
Structural Protocols; concrete impls depend on protocols never each other. DI assembled in `cli/context.py:build_context`. Tests override the assembly map to inject fakes.

## Data flow
- `add`: Walker → IgnoreFilter → detect type → content_hash → FileTracker.diff → ADDED:parse+chunk+embed+upsert / MODIFIED:delete_by_file then re-upsert / DELETED:delete_by_file / UNCHANGED:skip. Embedder name written once into ken_meta, re-validated every ingest+query.
- `ask`: guard embedder → embed_query → RetrievalPipeline (DenseStage→ANN top-k=5) → PromptBuilder (num_ctx≥8192) → Generator.stream tokens → citations after stream.

## LanceDB schema
Table `chunks`: id (`file_path::chunk_index` PK), vector (fixed_size_list<float32>[768]), chunk_text, file_path (indexed), file_type, chunk_index, content_hash, line_start, line_end, symbol_name(nullable), chunk_kind, git_commit(nullable), indexed_at. Table `ken_meta`: key/value KV (embedder_name, embed_dim, schema_version, created_at). FileRecords derived by grouping chunks on file_path (no second source of truth in v1).

## Critical-path decisions
- Code chunking: `.py`→stdlib `ast`; others→tree-sitter (`tree-sitter-languages`). Uniform SymbolSpan→Chunk. Module-level code → own chunk (no orphans). Huge function → subsplit windows w/ signature+docstring header. Unknown/missing grammar → fallback line-window (never crash).
- Incremental lifecycle: delete-by-file-then-upsert keyed on file_path; PK file_path::chunk_index → no drift. Contract test: chunk count after edit == fresh-index count.
- Retrieval future-proof: ordered Stage list; v1=[DenseStage]; v2 inserts KeywordStage(BM25/FTS+RRF) + RerankStage. QueryPipeline only knows Retriever.retrieve.
- Streaming: one iterator, two consumers (CLI print / Textual worker). Pipeline UI-agnostic.
- Graceful degradation: NullGitClient when no git/.git; typed OllamaUnavailable/ModelNotPulled → friendly messages.
- Model pick: detect_ram → ≥8GB qwen2.5:3b / ≥4GB 1.5b / else llama3.2:1b. Pure function.

## Testing seams (target 80%+)
Fakes: FakeEmbedder (deterministic 768-dim), FakeGenerator (scripted tokens), InMemoryStore (numpy cosine), FakeGitClient, HTTP mock for OllamaClient. Pure islands tested directly (all chunking via golden fixtures, prompt, citation, model_picker, file_tracker.diff). VectorStore contract test runs against both InMemoryStore and real LanceVectorStore. CLI e2e via Typer CliRunner. TUI via App.run_test() pilot.

## Phased build plan (per spec §9; phase-by-phase w/ checkpoints)
1. **Core pipeline, no UI.** domain + config + parsing(text/md) + prose_chunker + embedding + store(schema/lancedb/meta/migrations) + retrieval(dense) + generation + pipeline(ingest/query) + cli(add/ask). Prove RAG e2e w/ citations + embedder guard. Tests w/ fakes.
2. **Code-aware chunking (moat).** chunking/code/* (ast + tree-sitter), subsplit, fallback, golden fixtures. Validate retrieval on a real codebase.
3. **File tracking + incremental + git.** hasher, file_tracker, walker, ignore_filter(pathspec), git_client/null_git. `ken list`, `ken status`. .gitignore/.kenignore respect + repo detection.
4. **setup onboarding.** hardware, model_picker, ollama_health, flow. `ken setup`, `ken model`.
5. **chat TUI.** Textual app + widgets + stream_worker. Streaming typing feel, markdown, /sources /clear /exit.
6. **search + distribution + README/demo + launch.** `ken search`. GitHub Actions: test matrix (py3.10-3.13 × linux/mac/win) → build → publish-to-PyPI-on-tag + GitHub Release. README headline pitch, asciinema/GIF demo, copy-paste `uv tool install ken-rag`. Show HN / r/LocalLLaMA.

Note: hybrid retrieval (DenseStage + KeywordStage + RRF) lands in Phase 1's retrieval layer, not deferred.

## Top risks → mitigations
1. Embedder mismatch → ken_meta guard + dim guard, re-validate every op.
2. Index drift → delete-then-upsert lifecycle, contract test.
3. Chunking regressions → golden-file tests in CI.
4. tree-sitter platform fragility → ast for .py, fallback chunker, pinned grammars.
5. Coupling creep → protocol DI, named v2 extension points.
6. Ollama/git absence → typed errors + friendly mapping + NullGitClient.
7. num_ctx too small → Settings constant 8192 threaded explicitly, no path bypasses.
8. Schema lock-in → schema_version stamp + open_or_create migration.
