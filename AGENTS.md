# AGENTS.md — hello_llm_translate

## What this is
Example of translating text using Hugging Face Large Language / Marian MT models (e.g. Helsinki-NLP/opus-mt-en-de).

## Stack
- Python 3.13
- Hugging Face `transformers`, `datasets`, `torch`
- Poetry / pip virtual env

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

## Structure
- `main.py` — translation script
- `requirements.txt` — Python dependencies
- `Helsinki-NLP/` — model dir (opus-mt-en-de)
- `deepseek-ai/` — additional model dir

## Conventions
- Do not commit Hugging Face access tokens or real credentials.
- No comments in code unless asked.
- Verify: `python -m py_compile main.py`
