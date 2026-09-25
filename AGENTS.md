# AGENTS.md — hello_llm_translate

## What this is
Example of translating text using Hugging Face Large Language / Marian MT models (e.g. Helsinki-NLP/opus-mt-en-de).

## Stack
- Python 3.13
- Hugging Face `transformers`, `datasets`, `torch`
- Poetry / pip virtual env
- Ollama (local CPU LLM server, optional)

## Setup
```bash
huggingface-cli login   # get a token at https://huggingface.co
python3.13 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# From inside the model dir:
cd Helsinki-NLP/opus-mt-en-de/
./main.py
```
The script loads the Marian model/tokenizer, translates a sample English string to German and prints it. CUDA is used when available.

## Local LLM server (Ollama + qwen3:8b)
CPU-only, OpenAI-compatible. Managed through the `./ask.sh` menu (see README). Status of the setup:

- **Host** runs Ollama as a systemd service (`ollama.service`, `ollama serve`, port 11434). On the host, `ask.sh` detects systemd (`systemctl is-active ollama`) and routes Start/Stop/Status through `systemctl`; otherwise it falls back to manual `nohup` mode (PID/log: `./.ollama.pid`, `./.ollama.log`).
- **Bind vs client URL** are separate: the server listens on `BIND` (default `0.0.0.0:11434` from `OLLAMA_HOST`, so containers/other hosts can reach it); `HOST` (default `127.0.0.1:11434`) is what status/test/chat probing uses.
- **opencode** in this repo uses the local model via `opencode.json` → `ollama/qwen3:8b` with `baseURL http://192.168.43.2:11434/v1` (the host IP reachable from opencode's container). opencode config is not hot-reloaded — restart opencode after changing it.
- Model `qwen3:8b` (~5.2 GB) emits reasoning tokens first; give `max_tokens >= 256` or replies come back empty. `ask.sh` uses 512 for tests, 1024 for chat.
- qwen3 replies in the language you use; a bare "hello" may come back in Chinese — say "reply in English" to force it.

Verify: `curl -s http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Say one word: hello in Thai"}],"max_tokens":512}'`

## Structure
- `main.py` — translation script
- `requirements.txt` — Python dependencies
- `ask.sh` — Ollama server menu (tasks: 1 start, 2 status, 3 stop, 4 pull, 5 test chat, 6 install, 7 interactive chat)
- `opencode.json` — project opencode config (local model)
- `Helsinki-NLP/` — model dir (opus-mt-en-de)
- `deepseek-ai/` — additional model dir

## Conventions
- Do not commit Hugging Face access tokens or real credentials.
- No comments in code unless asked.
- Verify: `python -m py_compile main.py`; for `ask.sh`: `bash -n ask.sh`.