#!/bin/bash
# hello_llm_translate local LLM server menu (Ollama + Qwen3 8B, CPU)
# Port 11434, OpenAI-compatible API for opencode.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_BIN="${OLLAMA_BIN:-}"
if [ -z "$OLLAMA_BIN" ]; then
  for c in "$(command -v ollama)" /usr/local/bin/ollama /usr/bin/ollama /home/veto/.local/bin/ollama; do
    [ -n "$c" ] && [ -x "$c" ] && OLLAMA_BIN="$c" && break
  done
fi
if [ -z "$OLLAMA_BIN" ]; then
  echo "ollama binary not found in PATH (/usr/local/bin, /usr/bin, ~/.local/bin). Install or set OLLAMA_BIN."
  exit 1
fi
MODEL="${MODEL:-qwen3:8b}"
HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
PORT="${PORT:-11434}"
PID_FILE="$DIR/.ollama.pid"
LOG_FILE="$DIR/.ollama.log"

server_up() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    return 0
  fi
  curl -s --max-time 2 "http://$HOST/api/tags" >/dev/null 2>&1
}

start_server() {
  if server_up; then
    echo "Ollama already running on $HOST"
    return 0
  fi
  echo "Starting Ollama on $HOST (log: $LOG_FILE)"
  nohup "$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  for i in $(seq 1 30); do
    sleep 1
    if server_up; then
      echo "Ready."
      return 0
    fi
  done
  echo "Failed to start; see $LOG_FILE"
  return 1
}

stop_server() {
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null
    rm -f "$PID_FILE"
    echo "Stopped."
  else
    pkill -f "ollama serve" 2>/dev/null && echo "Stopped." || echo "Not running."
  fi
}

status_server() {
  if server_up; then
    echo "Running on $HOST — models:"
    curl -s "http://$HOST/api/tags" | python3 -c "import json,sys; [print('  -', m['name'], f\"{m['size']/1e9:.1f} GB\") for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || echo "  (none)"
  else
    echo "Not running."
  fi
}

pull_model() {
  "$OLLAMA_BIN" pull "$MODEL"
}

test_chat() {
  curl -s "http://$HOST/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Translate 'Hello' to Thai in one word.\"}],\"max_tokens\":512}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

while true; do
  echo ""
  echo "hello_llm_translate — local LLM ($MODEL)"
  echo "  1  Start Ollama server"
  echo "  2  Status"
  echo "  3  Stop server"
  echo "  4  Pull model $MODEL"
  echo "  5  Test chat"
  echo "  0  Exit"
  if ! read -rp "Task: " task; then
    break
  fi
  case "$task" in
    1) start_server ;;
    2) status_server ;;
    3) stop_server ;;
    4) pull_model ;;
    5) test_chat ;;
    0) break ;;
    *) echo "Unknown task" ;;
  esac
done
exit 0