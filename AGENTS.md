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
CPU-only server on 127.0.0.1:11434 (OpenAI-compatible). opencode in this repo uses it via `opencode.json` -> model `ollama/qwen3:8b`.

```bash
./ask.sh          # menu: start/stop server, pull model, test chat
ollama serve      # manual start (PID/log: ./.ollama.pid, ./.ollama.log)
./ask.sh          # pick "2 Status" to check the model is up
```
Verify chat: `curl -s http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Say one word: hello in Thai"}],"max_tokens":512}'`
Note: qwen3 emits reasoning tokens first; give `max_tokens >= 256` or replies come back empty.

## Structure
- `main.py` — translation script
- `requirements.txt` — Python dependencies
- `ask.sh` — Ollama server menu (start/status/stop/pull/test)
- `opencode.json` — project opencode config (local model)
- `Helsinki-NLP/` — model dir (opus-mt-en-de)
- `deepseek-ai/` — additional model dir

## Conventions
- Do not commit Hugging Face access tokens or real credentials.
- No comments in code unless asked.
- Verify: `python -m py_compile main.py`
