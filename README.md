# ken-rag

> RAG that actually understands your code — local, private, and lives in your terminal.

Point `ken` at a folder of docs or a codebase, ask questions, get answers grounded in your actual files with source citations. 100% local. No API keys, no cloud, nothing leaves your machine.

## Why

Local-RAG today is either heavy Electron web UIs or rough tutorial scripts. `ken` is a polished, terminal-native tool with git-like ergonomics: code-aware chunking, hybrid search, and citations you can trust.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally

```bash
ollama pull nomic-embed-text   # embeddings (retrieval)
ollama pull qwen2.5:3b         # answers (use qwen2.5:1.5b / llama3.2:1b on low RAM)
```

## Install

```bash
git clone https://github.com/<you>/ken-rag && cd ken-rag
uv tool install .        # or: pipx install .
```

## Quickstart

```bash
ken add .                       # index the current folder (recursive, skips .venv/node_modules/etc.)
ken add ./docs                  # or a specific folder
ken add ./notes/spec.md         # or a single file
ken list                        # show what's indexed
ken ask "how does auth work?"   # grounded answer + source citations
```

Re-running `ken add` only re-embeds files that changed (hash-tracked), so it stays fast on big repos.

## How it works

```
ken add → walk files → chunk → embed (nomic-embed-text) → store (LanceDB)
ken ask → embed query → hybrid search (vector + keyword, RRF) → qwen2.5 → answer + citations
```

Everything runs through Ollama. One embedding model, one LLM, an embedded vector DB. No framework, no server.

## File types

`.txt` `.md` `.pdf` and source code (`.py .js .ts .go .rs .java .c .cpp .h .rb .php` …).

## Status

Early. Working today: `add`, `ask`, `list`. Coming: code-aware chunking, `ken chat` (full-screen streaming TUI), `ken search`, `ken status`, git-awareness, one-command `uv tool install ken-rag` from PyPI.

## License

MIT
