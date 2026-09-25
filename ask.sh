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
NEED_INSTALL=0
if [ -z "$OLLAMA_BIN" ]; then
  NEED_INSTALL=1
  echo "Warning: ollama binary not found (checked PATH, /usr/local/bin, /usr/bin, ~/.local/bin)."
  echo "Use task 6 to install it, or set OLLAMA_BIN to an existing binary."
fi
MODEL="${MODEL:-qwen3:8b}"
BIND="${OLLAMA_HOST:-0.0.0.0:11434}"
HOST="${HOST:-127.0.0.1:11434}"
PORT="${PORT:-11434}"
PID_FILE="$DIR/.ollama.pid"
LOG_FILE="$DIR/.ollama.log"

systemd_managed() {
  command -v systemctl >/dev/null 2>&1 && systemctl is-active ollama >/dev/null 2>&1
}

install_ollama() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
    OLLAMA_BIN="$(command -v ollama)"
    if [ -n "$OLLAMA_BIN" ]; then
      NEED_INSTALL=0
      echo "Ollama installed: $OLLAMA_BIN"
    else
      echo "Install finished but ollama not found — add /usr/local/bin to PATH or set OLLAMA_BIN."
    fi
  else
    echo "curl not found — install Ollama manually: https://ollama.com/download"
  fi
}

server_up() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    return 0
  fi
  curl -s --max-time 2 "http://$HOST/api/tags" >/dev/null 2>&1
}

start_server() {
  if systemd_managed; then
    echo "Ollama is a systemd service (ollama.service) — using systemctl"
    systemctl start ollama
    echo "Started."
    return 0
  fi
  if server_up; then
    echo "Ollama already running on $HOST"
    return 0
  fi
  echo "Starting Ollama (listen $BIND, log: $LOG_FILE)"
  nohup env OLLAMA_HOST="$BIND" "$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &
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
  if systemd_managed; then
    echo "Ollama is a systemd service (ollama.service) — using systemctl"
    systemctl stop ollama
    echo "Stopped."
    return 0
  fi
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null
    rm -f "$PID_FILE"
    echo "Stopped."
  else
    pkill -f "ollama serve" 2>/dev/null && echo "Stopped." || echo "Not running."
  fi
}

status_server() {
  if systemd_managed; then
    echo "Running via systemd (ollama.service) — models:"
    curl -s "http://$HOST/api/tags" | python3 -c "import json,sys; [print('  -', m['name'], f\"{m['size']/1e9:.1f} GB\") for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null || echo "  (none)"
    return 0
  fi
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

interactive_chat() {
  if ! server_up; then
    echo "Server not running — start it with task 1 first."
    return 1
  fi
  echo "Interactive chat with $MODEL (empty line or /quit to exit, /clear to reset, /models to list)."
  local history="[]"
  local tmpf
  tmpf="$(mktemp)"
  while true; do
    printf "\n> "
    if ! IFS= read -r line; then
      break
    fi
    case "$line" in
      "") break ;;
      "/quit") break ;;
      "/clear") history="[]"; echo "(history cleared)"; continue ;;
      "/models") "$OLLAMA_BIN" list 2>/dev/null; continue ;;
      *) ;;
    esac
    history="$(python3 - "$history" "$line" <<'PYEOF'
import json,sys
hist=json.loads(sys.argv[1])
hist.append({"role":"user","content":sys.argv[2]})
print(json.dumps(hist))
PYEOF
)"
    payload="$(python3 - "$history" "$MODEL" <<'PYEOF'
import json,sys
hist=json.loads(sys.argv[1])
print(json.dumps({"model":sys.argv[2],"messages":hist,"stream":True,"max_tokens":1024}))
PYEOF
)"
    echo ""
    curl -sN "http://$HOST/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      | python3 -c "
import json,sys
reply=''
for line in sys.stdin:
    line=line.strip()
    if not line.startswith('data:'):
        continue
    data=line[5:].strip()
    if data=='[DONE]':
        break
    try:
        chunk=json.loads(data)
    except Exception:
        continue
    c=chunk.get('choices',[{}])[0].get('delta',{}).get('content') or ''
    if c:
        sys.stdout.write(c); sys.stdout.flush(); reply+=c
import os
with open(os.environ['TMPF'],'w') as f:
    f.write(reply)
" 2>/dev/null
    TMPF="$tmpf"
    history="$(python3 - "$history" "$tmpf" <<'PYEOF'
import json,sys
hist=json.loads(sys.argv[1])
try:
    with open(sys.argv[2]) as f:
        reply=f.read()
    hist.append({"role":"assistant","content":reply})
except Exception:
    pass
print(json.dumps(hist))
PYEOF
)"
  done
  rm -f "$tmpf"
  echo ""
}

while true; do
  echo ""
echo "hello_llm_translate — local LLM ($MODEL)"
[ "$NEED_INSTALL" = "1" ] && echo "  !  Ollama not installed yet — run task 6"
echo "  1  Start Ollama server"
echo "  2  Status"
echo "  3  Stop server"
echo "  4  Pull model $MODEL"
echo "  5  Test chat"
echo "  6  Install Ollama (only if not found)"
echo "  7  Chat (interactive)"
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
    6) install_ollama ;;
    7) interactive_chat ;;
    0) break ;;
    *) echo "Unknown task" ;;
  esac
done
exit 0