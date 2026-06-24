# ken-rag — Project Spec & Build Context

> A local, private, terminal-native RAG tool. Point it at your files, ask questions, chat with your data — all offline. Think "git for your documents" meets "Claude Code for RAG."

This document is the source of truth for the project. Keep it in the Claude Code session. It contains the goal, the locked tech stack, the architecture, and — most importantly — the **hard parts that must be done carefully**. Read the "Critical Features" section before writing any retrieval code.

---

## 1. Goal

Build a RAG CLI that a developer can install in one command, point at a folder (docs *or* a codebase), and immediately ask questions about — with answers grounded in their actual files, citing sources, running 100% locally. No API keys, no cloud, no data leaving the machine.

**Why this exists / the gap it fills:** The local-RAG space is full of two kinds of projects — heavy web UIs (AnythingLLM, Open WebUI, kotaemon) and rough tutorial scripts. Almost nobody has built a *polished, terminal-native* RAG tool with git-like ergonomics and a Claude-Code-quality chat experience. That is the wedge. The audience that stars GitHub repos (r/LocalLLaMA, Hacker News, terminal-loving devs) is underserved by pretty-but-heavy Electron apps.

**Star target:** 5k–10k. This is achievable ONLY through polish + a strong README + a launch moment (Show HN / r/LocalLLaMA post). It will NOT come from feature count. A narrow tool that nails its core beats a sprawling one that half-works. Discipline on scope is part of the strategy.

**The one-line pitch to lead the README with:**
> "RAG that actually understands your code — local, private, and lives in your terminal."

---

## 2. What makes ken-rag different (the moat)

These are the things competitors don't do. Protect them. They are the reason anyone stars this instead of the 50 other ollama-rag repos.

1. **Code-aware chunking (HEADLINE FEATURE).** Most RAG tools split text by character count, which butchers source code — a function gets cut in half, an import block gets orphaned, retrieval quality on codebases is terrible. ken-rag chunks code by *semantic unit* (function, class, method) using AST/tree-sitter parsing. This makes it genuinely good at "ask questions about my codebase," which almost no local tool does well. This is the differentiator. Treat it as a first-class feature, not an edge case.
2. **git-like ergonomics.** `ken add`, `ken status`, `ken list`, `ken tree`, file tracking so re-indexing only re-embeds *changed* files (hash-based, like git tracking working-tree changes). Nobody else commits to this mental model. It makes the tool feel familiar and fast.
3. **Claude-Code-quality TUI.** A real full-screen chat experience (Textual), streaming tokens, rendered markdown, source citations, scrollback — not a raw `input()` loop. The existing CLI tools have none of this. The "typing feel" is a feature.
4. **Zero-config defaults with hardware detection.** Detect available RAM at setup and auto-pick the right model so a non-expert never has to choose. Most tools dump model selection on the user and fail.
5. **Correct-by-default RAG.** Set `num_ctx` properly, pick a good embedder, sane chunk sizes. Most toy projects ship broken defaults (tiny context window) that silently produce bad answers.

---

## 2b. Git integration goals (a second moat — lean into it)

There are two separate "git" ideas. Don't conflate them.

- **Git-like ergonomics** (the *feel*): `add`, `status`, `list`, `tree`, hash-based change tracking. Covered in the moat above — it's about borrowing git's mental model so the tool feels instantly familiar.
- **Real git integration** (this section): ken-rag actually *reads and understands the repo's git state*. Since the headline use case is "RAG for your codebase," and codebases live in git, this is a natural, underexplored edge almost no competitor has. It turns ken-rag from "a RAG tool you can point at code" into "a RAG tool built for version-controlled code."

Treat git-awareness as the codebase counterpart to the prose pipeline. Prioritize the cheap, high-value pieces (respect `.gitignore`, track by git state) for v1; defer the fancier ones to v2.

### Git goals — prioritized

**v1 (do these — cheap and high-impact):**
1. **Respect `.gitignore`.** When indexing a repo, skip anything git ignores (`node_modules`, `.venv`, `dist`, build artifacts, secrets). This is the single biggest quality+speed win for codebase RAG — indexing junk pollutes retrieval and wastes embeddings. Also honor a `.kenignore` for tool-specific excludes.
2. **Detect "is this a git repo?"** and adapt. Inside a repo, `ken add .` should behave like "index the tracked source tree," not "blindly walk every byte on disk."
3. **Track by content, align with git.** The existing hash-based change tracking should map cleanly onto git's notion of modified/untracked/deleted files so `ken status` reads like `git status` ("3 modified, 1 untracked, 2 deleted since last index").

**v2 (announce as updates — these drive a second wave of stars):**
4. **Index only tracked files by default.** Use `git ls-files` as the source of truth for what to embed (with a flag to include untracked).
5. **Commit-aware citations.** Store the commit hash at index time so a citation can say "as of commit `a1b2c3d`," and `ken status` can warn "index is 12 commits behind HEAD."
6. **Re-index on git events.** A `ken sync` that diffs current `HEAD` against the indexed commit and re-embeds only the files that changed between them — incremental updates powered by `git diff` instead of re-hashing the whole tree. This is the killer feature: keeping a codebase index fresh becomes nearly free.
7. **Blame/history-aware answers (stretch).** Optionally pull in `git blame`/recent-commit context so the model can answer "who last touched this and why" or weight recently-changed code. Stretch goal — only if retrieval quality clearly benefits.

### Why this is a moat
The competitors are either generic document tools (treat code as plain text, index `node_modules`, no git awareness) or web UIs aimed at PDFs. A terminal-native RAG tool that *natively understands git* speaks directly to the exact audience that stars repos: developers working in version-controlled codebases. `.gitignore` respect alone visibly outclasses tools that drown in vendored dependencies.

### Pitfalls
- Don't make git **required** — many users will point ken-rag at a plain folder of docs with no `.git`. Git-awareness must be an *enhancement that activates when a repo is detected*, never a hard dependency. Degrade gracefully to plain filesystem walking.
- Shelling out to the `git` binary is fine and simplest; if you want to avoid that dependency, `pygit2`/`dulwich` exist, but the `git` CLI is universal on dev machines. Prefer shelling out, handle "git not installed" by falling back to filesystem mode.
- Keep `.gitignore` parsing correct (nested ignores, negation patterns) — use a tested library (e.g. `pathspec`) rather than hand-rolling glob matching.

---

## 3. Locked tech stack (decided — do not re-litigate)

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.10+** | Best RAG/ML ecosystem |
| LLM runtime | **Ollama** | One-command cross-platform install; it wraps llama.cpp so low-end perf is fine without the UX cost of raw llama.cpp |
| Embedding model | **nomic-embed-text** (via Ollama) | Better retrieval quality than all-MiniLM, still tiny, keeps everything in one runtime (no torch dependency) |
| Default LLM | **qwen2.5:3b**, with RAM-based fallback to **qwen2.5:1.5b** / **llama3.2:1b** on low-end | Best instruction-following per parameter for RAG on weak hardware |
| Vector store | **LanceDB** | Embedded (no server), fast, lightweight, persists to disk — fits the "clone/install and run" + low-end promise |
| CLI framework | **Typer** | Clean command definitions, great help output, the `ken add/ask/chat/list` structure maps naturally |
| TUI / chat UI | **Textual** (full-screen chat) + **Rich** (styled `ask`/`list`/`tree` output, spinners, markdown) | Best-in-class Python TUI; this is how the Claude-Code feel is achieved |
| PDF parsing | **PyMuPDF** | Fast, robust text extraction |
| Install method | **`uv tool install ken-rag`** / **`pipx install ken-rag`** | Isolated, single-command, no venv friction for the user |
| Orchestration framework | **None — write the pipeline by hand (~200 lines)** | LangChain is heavy and hides the chunking control we specifically need for the code-aware feature. Stay framework-free. |

**Embedder note:** earlier the plan considered all-MiniLM-L6-v2. That was reconsidered and replaced by nomic-embed-text. Reason to remember: MiniLM would add a heavy `torch` dependency; nomic runs inside Ollama so the tool has a single runtime to depend on. Also clears up an earlier confusion — the embedder's job is *retrieval* (turning text into vectors), NOT generating answers. There is exactly one generative LLM. Embedder + generator = the "two models," not two answerers. No smart routing in v1.

---

## 4. The RAG pipeline (how it works end-to-end)

```
ken add <path>
  └─> walk path → per-filetype parser → raw text
        └─> CHUNK (code-aware for code, recursive-overlap for prose)
              └─> embed each chunk via nomic-embed-text (Ollama)
                    └─> store (vector, chunk_text, metadata) in LanceDB
                          metadata = {file_path, file_type, chunk_index, content_hash, line_start, line_end, symbol_name}

ken ask "question"
  └─> embed question (same nomic model — MUST be the same model used at ingest)
        └─> LanceDB nearest-neighbor search → top-k chunks (k=4–6)
              └─> (optional v2) re-rank with cross-encoder
                    └─> build prompt: system + retrieved chunks (with sources) + question
                          └─> stream from qwen2.5 (Ollama) with num_ctx raised (>=8192)
                                └─> render answer + cite source files/lines
```

Key invariants:
- **Embed query and documents with the SAME model.** Mixing embedders silently destroys retrieval. Store the embedder name in the DB and refuse mismatched queries.
- **Vectors are 768-dim for nomic-embed-text.** Configure the LanceDB schema accordingly (don't hardcode 384, that was MiniLM).
- **Always raise `num_ctx`** on the Ollama call (default 2048/4096 is too small to hold retrieved chunks + question + answer; use >=8192). This is a top cause of "the answers are bad" in competing tools.
- **Always return citations** (file path + line range). Users trust grounded answers and it's a visible quality signal.

---

## 5. CRITICAL FEATURES — build these carefully

These are the parts where sloppiness ruins the project. Spend the care budget here.

### 5.1 Code-aware chunking (the moat — get this right)
- For source files (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, ...), chunk by **semantic unit**: function, method, class. Use **tree-sitter** (language-agnostic, supports dozens of grammars) or Python's `ast` for `.py` as a start.
- Each code chunk's metadata should carry `symbol_name`, `line_start`, `line_end` so citations point to "the `parse_config` function, lines 40–72."
- If a function is huge, sub-split it but keep its signature/docstring as a header on each sub-chunk so context isn't lost.
- For prose (`txt`, `md`, `pdf`): recursive splitting ~300–800 tokens with ~10–15% overlap. Respect markdown headers / paragraph boundaries — don't cut mid-sentence.
- **Failure mode to avoid:** naive fixed-character splitting on code. It's the #1 reason existing tools are bad at codebases. This feature only matters if it's actually done right.

### 5.2 File tracking / incremental re-index (the git feel)
- Store a `content_hash` per file. On `ken add` of an already-indexed path, only re-embed files whose hash changed; delete+re-add their chunks. Skip unchanged files.
- This makes the tool feel instant on large repos and is the foundation for `ken status` ("3 files changed, 1 new, run `ken add` to update").
- Get the add/update/delete chunk lifecycle correct so the index never drifts from disk (no orphaned chunks, no stale duplicates).

### 5.3 Retrieval quality
- Tune `k` (start 5). Too low → missing context; too high → noise + blows the context window.
- **Hybrid search matters for code:** pure vector similarity misses exact identifiers (a user searching for `getUserToken` wants the literal match). Combine vector search with keyword/BM25 where possible. Even a simple keyword pre-filter helps a lot on codebases. (v2 if needed, but design the retrieval layer so it can be added.)
- Show the user *which* chunks were retrieved (at least in a `--verbose` / `ken search` raw mode) — it's the best debugging tool and a trust signal.

### 5.4 The TUI "typing feel" (`ken chat`)
- Full-screen Textual app: scrollable history pane + input box, multi-line input, command history (up-arrow), `/exit`, `/clear`, `/sources`.
- **Stream tokens as they generate** from Ollama — this is what makes it feel alive vs. a frozen "thinking..." then a wall of text.
- Render markdown and code blocks with syntax highlighting (Rich).
- Show a subtle spinner/status while retrieving + first token latency.
- This polish is a feature, not decoration — it's a primary reason people will choose this over rough CLI competitors.

### 5.5 Setup / onboarding flow (first impression = stars)
1. User installs (`uv tool install ken-rag`) and runs `ken` or `ken setup`.
2. Friendly intro screen (what it is, that it's 100% local).
3. **Check for Ollama.** If missing → show the exact install command for their OS and the `ollama pull` commands; don't crash, guide them.
4. **Detect RAM**, recommend + pull a default model automatically (qwen2.5:3b, or 1.5b/1b on low-end). Let them override but never *require* a choice.
5. Pull `nomic-embed-text`.
6. Confirm "Setup done — try `ken add ./docs` then `ken ask "..."`".
- A broken/confusing first run kills adoption. This flow deserves real care.

### 5.6 Correct Ollama defaults
- Set `num_ctx` >= 8192 on generation calls.
- Handle Ollama-not-running gracefully (clear message, offer to start it).
- Stream responses. Reasonable timeouts. Clear errors when a model isn't pulled.

---

## 6. Commands (v1 surface)

| Command | Purpose |
|---|---|
| `ken setup` | Onboarding: check Ollama, detect RAM, pick + pull model, pull embedder |
| `ken add <path>` | Ingest a file or folder (incremental — only changed files; respects `.gitignore`/`.kenignore` and detects git repos) |
| `ken ask "<question>"` | One-shot grounded answer with citations |
| `ken chat` | Full-screen Textual chat UI (multi-turn) |
| `ken list` | Show indexed files (count, type, last indexed) |
| `ken status` | Index summary + changed/new/deleted files vs disk (git-status feel) |
| `ken search "<term>"` | Raw retrieval, no LLM — shows top chunks (fast, debuggable) |
| `ken model` | Show / switch / pull the active LLM |

**Deferred to v2 (don't build for launch):** `ken tree` (pretty file tree of the index), `ken reindex` (force full re-embed, e.g. after changing chunk settings), `ken sync` (git-diff-based incremental re-index against HEAD), `ken remove <file>` / `ken clear`, `ken config` (chunk size, k, model), re-ranking, hybrid search, commit-aware citations, docx/xlsx support, PDF-with-images OCR.

---

## 7. File type support

**v1 (launch with these, done well):** `.txt`, `.md`, `.pdf` (text), and code files (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`, `.rb`, `.php`, etc.).

**v2+:** `.docx` (python-docx), `.xlsx` (openpyxl), PDF with images (PyMuPDF render + OCR via pytesseract), more.

Rationale: nailing 4 categories beats half-handling 12. Each v2 file type is an "update" that keeps the repo active and gives you something to post about — which itself drives stars.

---

## 8. Scope discipline (read before adding anything)

The biggest risk to this project is **building too much before launch**. Ship a small thing that is *perfect*. Every feature outside the v1 list above is a v2 announcement, not a v1 blocker. When tempted to add something, ask: "does the headline pitch (great local RAG that understands code, in a beautiful terminal) need this to be true on day one?" If no → v2.

---

## 9. Suggested build order

1. Core pipeline, no UI: `add` (prose chunking only) → embed → LanceDB → `ask` with citations. Prove RAG works end-to-end.
2. Add **code-aware chunking** (the moat). Test retrieval quality on a real codebase.
3. File tracking / incremental `add` + `ken status` + `ken list`. Add `.gitignore`/`.kenignore` respect and git-repo detection here — it's cheap and dramatically improves codebase indexing quality.
4. `ken setup` onboarding + RAM detection + Ollama checks.
5. `ken chat` Textual UI with streaming. Polish the typing feel.
6. `ken search` raw retrieval. README with the headline pitch, a GIF/asciinema demo, and copy-paste install. Then launch (Show HN / r/LocalLLaMA).

A great asciinema/GIF demo in the README is worth more than any single feature for stars. Budget time for it.

---

## 10. Known pitfalls (learn from competitors)
- Tiny `num_ctx` → bad answers. Fix in defaults.
- Mismatched embedder between ingest and query → broken retrieval. Pin the model.
- Naive character-splitting on code → poor codebase RAG. Use semantic chunking.
- Requiring the user to choose a model with no guidance → drop-off. Auto-pick.
- Crashing when Ollama isn't installed/running → first-run failure. Guide instead.
- Orphaned/duplicate chunks after edits → index drift. Get the chunk lifecycle right.
- Indexing `node_modules`/`.venv`/build junk → polluted retrieval + wasted embeddings. Respect `.gitignore` from the start.
- Making git a hard requirement → breaks the plain-folder-of-docs use case. Git-awareness must activate only when a repo is detected, and degrade gracefully otherwise.
- Shipping a web UI "to be safe" → you become one of 50 indistinguishable repos. Stay terminal-native; that's the whole point.
