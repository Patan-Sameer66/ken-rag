# ken-rag

Ask questions about your own files from the terminal. Your docs, your codebase, whatever you point it at. The answers come back with the exact files and line ranges they came from, and nothing ever leaves your machine.

No API keys. No cloud. No web UI to babysit. Just `ken`.

```text
$ ken ask "how does the auth middleware work?"

The require_auth decorator wraps each route and checks the bearer
token before the handler runs. Expired tokens get a 401.

  ① auth/middleware.py:14-37  require_auth
  ② auth/tokens.py:5-22       verify_token
```

## Why this exists

Most local RAG tools are one of two things: a heavy web app you have to run and log into, or a tutorial script that splits your code every 500 characters and then wonders why it can't answer anything about it.

I wanted the thing in between. Something that lives in the terminal, feels like `git`, and is actually good at code, not just PDFs. So citations are first class, the defaults are sane, and it runs entirely on your own hardware through Ollama.

It is early, but the core loop works today: point it at files, ask questions, get grounded answers.

## What you can do right now

- `ken add` indexes a folder or a single file. Re-running only re-embeds what changed, so it stays quick on big trees.
- `ken ask` retrieves the relevant chunks and answers from them, with citations.
- `ken list` shows what is indexed.

Retrieval is hybrid out of the box: it combines vector similarity with keyword matching, so searching for a literal identifier like `getUserToken` actually finds it instead of something that merely sounds related.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running

Pull the two models once:

```bash
ollama pull nomic-embed-text    # turns text into vectors (retrieval)
ollama pull qwen2.5:3b          # writes the answers
```

On a lighter machine, swap the answer model for `qwen2.5:1.5b` or `llama3.2:1b`.

## Install

```bash
git clone https://github.com/Patan-Sameer66/ken-rag
cd ken-rag
uv tool install .          # or: pipx install .
```

That puts `ken` on your PATH.

## Quickstart

```bash
ken add .                          # index the current project
ken list                           # see what got indexed
ken ask "what does this project do?"
```

A few more ways to point it:

```bash
ken add ./docs                     # a specific folder, recursively
ken add ./notes/spec.md            # a single file
ken --db ./.ken add .              # choose where the index lives (defaults to ./.ken)
```

When you add a folder, it walks every subdirectory and skips the usual noise: `.venv`, `node_modules`, `__pycache__`, `.git`, build and cache directories. No point embedding your dependencies.

## How it works

```text
ken add   →  walk files  →  chunk  →  embed (nomic-embed-text)  →  store in LanceDB
ken ask   →  embed your question  →  hybrid search (vector + keyword)  →  qwen2.5  →  answer + citations
```

Two models, both running inside Ollama. One turns text into vectors for search, the other writes the answer from the chunks that came back. The vector store is LanceDB, which is just a file on disk, so there is no server to start and nothing to connect to.

The query and the documents are always embedded with the same model, and that model's name is written into the index. If you ever try to query an index with a different embedder, it stops and tells you instead of quietly returning garbage. The context window is set wide enough to actually hold the retrieved chunks, which is the thing most toy setups get wrong.

## What it handles

Plain text, Markdown, PDF, and source code across the common languages (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.rb`, `.php`, and friends).

## Where it's going

The roadmap, roughly in order:

- Code-aware chunking that splits source by function and class instead of by character count, so a citation can point at "the `parse_config` function" rather than "characters 4000 to 4500."
- `ken chat`, a full-screen chat with streaming answers and scrollback.
- `ken search` to see the raw retrieved chunks with no model in the loop, which is the fastest way to debug retrieval.
- `ken status`, git awareness, and a one-line install straight from PyPI.

## Privacy

Everything runs locally. The files you index, the questions you ask, and the answers you get all stay on your machine. There is no telemetry and no network call except to your own Ollama instance.

## Contributing

Issues and pull requests are welcome. If you hit a file type that parses badly or a retrieval result that feels off, open an issue with the input. That kind of report is worth more than it sounds.

## License

MIT
