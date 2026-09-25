<img src="hello_llm_translate.svg" alt="hello_llm_translate" width="120">

# hello_llm_translate

Example to translate with Large Language Models 

## Setup an Account at https://huggingface.co
* Get and Access Token, so the script can download all the models databases

## Setup the Environment 
```
poetry install
```
## Login to huggingface
```
poetry run hf auth login
```

## Example from English to German
```
cd Helsinki-NLP/opus-mt-en-de/
./main.py 
```

## Local LLM server (Ollama + qwen3:8b)
Run a CPU-only LLM API server on localhost:11434, usable directly by opencode
in this repo (see `opencode.json`).
```
./ask.sh        # menu: start/stop server, pull model, test chat
```


## Extra Repository ##
```
 git remote add codeberg ssh://git@codeberg.org/veto/hello_llm_translate
 git push codeberg

```



