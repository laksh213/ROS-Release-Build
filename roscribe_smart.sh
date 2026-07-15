#!/usr/bin/env bash
#
# ROScribe — Control Script & Process Manager (On/Off Switch)
#
# Usage:
#   ./roscribe.sh start [--ui nicegui|streamlit] [--public]
#   ./roscribe.sh stop
#   ./roscribe.sh status
#   ./roscribe.sh build
#

set -uo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0;50m' # No Color
BOLD='\033[1m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE_APP="/tmp/roscribe_smart_app.pid"
PID_FILE_TUNNEL="/tmp/roscribe_smart_tunnel.pid"
LOG_APP="/tmp/roscribe_smart_app.log"
LOG_TUNNEL="/tmp/roscribe_smart_tunnel.log"

cd "$REPO_ROOT"

# Ensure virtualenv exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment (.venv) not found.${NC}"
    echo -e "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Load .env_smart or fall back to .env variables
ENV_FILE=".env"
[ -f ".env_smart" ] && ENV_FILE=".env_smart"
line=""
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "${line:-}" ]; do
        # Ignore comments and empty lines
        if [[ ! "${line:-}" =~ ^# ]] && [[ -n "${line:-}" ]]; then
            key=$(echo "$line" | cut -d= -f1 | xargs)
            value=$(echo "$line" | cut -d= -f2- | xargs)
            value="${value#\"}"
            value="${value%\"}"
            value="${value#\'}"
            value="${value%\'}"
            export "$key=$value" 2>/dev/null || true
        fi
    done < "$ENV_FILE"
fi

# Helper to activate virtualenv
activate_venv() {
    source .venv/bin/activate
}

status() {
    local app_running=0
    local tunnel_running=0
    local app_pid=""
    local tunnel_pid=""

    # Check App
    if [ -f "$PID_FILE_APP" ]; then
        app_pid=$(cat "$PID_FILE_APP")
        if ps -p "$app_pid" > /dev/null 2>&1; then
            app_running=1
        fi
    fi
    # Fallback to ps check if pid file matches
    if [ $app_running -eq 0 ]; then
        app_pid=$(ps aux | grep -Ei "python (app/workspace_smart.py)" | grep -v grep | awk '{print $2}' | head -n 1)
        [ -n "$app_pid" ] && app_running=1
    fi

    # Check Tunnel
    if [ -f "$PID_FILE_TUNNEL" ]; then
        tunnel_pid=$(cat "$PID_FILE_TUNNEL")
        if ps -p "$tunnel_pid" > /dev/null 2>&1; then
            tunnel_running=1
        fi
    fi
    if [ $tunnel_running -eq 0 ]; then
        tunnel_pid=$(ps aux | grep -E "cloudflared tunnel" | grep -v grep | awk '{print $2}' | head -n 1)
        [ -n "$tunnel_pid" ] && tunnel_running=1
    fi

    echo -e "${BOLD}=== ROScribe Smart Status ===${NC}"
    if [ $app_running -eq 1 ]; then
        # Check if streamlit or nicegui
        local proc_cmd=$(ps -p "$app_pid" -o args= 2>/dev/null || echo "")
        local ui_type="NiceGUI Workspace (Smart)"
        echo -e "App Server : ${GREEN}RUNNING${NC} (PID: $app_pid, UI: $ui_type)"
        echo -e "Local URL  : ${BLUE}http://127.0.0.1:8081${NC}"
    else
        echo -e "App Server : ${RED}STOPPED${NC}"
    fi

    if [ $tunnel_running -eq 1 ]; then
        local cf_url=""
        if [ -f "$LOG_TUNNEL" ]; then
            cf_url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_TUNNEL" | head -n 1)
        fi
        echo -e "Public Tunnel: ${GREEN}RUNNING${NC} (PID: $tunnel_pid)"
        if [ -n "$cf_url" ]; then
            echo -e "Public URL : ${BLUE}${cf_url}${NC}"
        else
            echo -e "Public URL : ${YELLOW}Obtaining... check status again in a moment.${NC}"
        fi
    else
        # Check Tailscale Funnel as a fallback
        local ts_url=""
        if command -v tailscale >/dev/null 2>&1; then
            ts_url=$(export PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/Tailscale.app/Contents/MacOS:$PATH"; tailscale funnel status 2>/dev/null | grep -oE 'https://[a-zA-Z0-9.-]+\.ts\.net' | head -n 1 || true)
        fi
        if [ -n "$ts_url" ]; then
            echo -e "Public Tunnel: ${GREEN}RUNNING${NC} (Tailscale Funnel)"
            echo -e "Public URL : ${BLUE}${ts_url}/${NC}"
        else
            echo -e "Public Tunnel: ${RED}STOPPED${NC}"
        fi
    fi
    echo -e "========================"

    if [ $app_running -eq 1 ]; then
        return 0
    else
        return 3
    fi
}

stop() {
    echo -e "${YELLOW}Stopping all ROScribe processes...${NC}"

    # Kill App
    if [ -f "$PID_FILE_APP" ]; then
        local app_pid=$(cat "$PID_FILE_APP")
        if ps -p "$app_pid" > /dev/null 2>&1; then
            kill "$app_pid" 2>/dev/null || kill -9 "$app_pid" 2>/dev/null
            echo -e "Stopped app server (PID $app_pid)."
        fi
        rm -f "$PID_FILE_APP"
    fi
    # Extra safety pkill
    pkill -f "app/workspace_smart.py" 2>/dev/null || true

    # Kill Tunnel
    if [ -f "$PID_FILE_TUNNEL" ]; then
        local tunnel_pid=$(cat "$PID_FILE_TUNNEL")
        if ps -p "$tunnel_pid" > /dev/null 2>&1; then
            kill "$tunnel_pid" 2>/dev/null || kill -9 "$tunnel_pid" 2>/dev/null
            echo -e "Stopped Cloudflare tunnel (PID $tunnel_pid)."
        fi
        rm -f "$PID_FILE_TUNNEL"
    fi
    pkill -f "cloudflared tunnel" 2>/dev/null || true

    echo -e "${GREEN}All processes stopped successfully.${NC}"
}

start() {
    # Check if already running
    if status >/dev/null 2>&1; then
        echo -e "${YELLOW}ROScribe is already running. Run './roscribe.sh stop' first to restart.${NC}"
        exit 0
    fi
    # Parse arguments
    local public=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --public)
                public=1
                shift
                ;;
            *)
                echo -e "${RED}Unknown argument: $1${NC}"
                echo "Usage: ./roscribe_smart.sh start [--public]"
                exit 1
                ;;
        esac
    done

    activate_venv

    echo -e "${GREEN}Starting ROScribe Smart server (NiceGUI)...${NC}"
    # Ensure credentials exist in the chosen env file
    if ! grep -q '^ROSCRIBE_USERS=' "$ENV_FILE" 2>/dev/null; then
        local pw=$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 10)
        echo "ROSCRIBE_USERS=laksh:${pw}" >> "$ENV_FILE"
        echo "ROSCRIBE_STORAGE_SECRET=$(LC_ALL=C tr -dc 'a-f0-9' </dev/urandom | head -c 48)" >> "$ENV_FILE"
        echo -e "${YELLOW}Generated temporary NiceGUI login credentials in $ENV_FILE:${NC}"
        echo -e "  User: laksh"
        echo -e "  Pass: ${pw}"
    fi

    .venv/bin/python -u app/workspace_smart.py > "$LOG_APP" 2>&1 &
    echo $! > "$PID_FILE_APP"
    
    # Wait for app to boot
    printf "Waiting for app to start"
    for _ in $(seq 1 30); do
        if curl -s -o /dev/null http://127.0.0.1:8081/; then
            echo -e " ${GREEN}Online!${NC}"
            break
        fi
        printf "."
        sleep 1
    done
    echo -e "Local NiceGUI URL: ${BLUE}http://127.0.0.1:8081${NC}"

    # Start Cloudflare quick tunnel if requested
    if [ $public -eq 1 ]; then
        echo -e "${GREEN}Starting public Cloudflare tunnel...${NC}"
        local port=8081

        if ! command -v cloudflared >/dev/null 2>&1; then
            echo -e "${YELLOW}Warning: 'cloudflared' command not found. Cannot start public tunnel.${NC}"
            echo -e "Install it with: brew install cloudflared"
        else
            cloudflared tunnel --url "http://127.0.0.1:${port}" > "$LOG_TUNNEL" 2>&1 &
            echo $! > "$PID_FILE_TUNNEL"

            printf "Obtaining public URL"
            local cf_url=""
            for _ in $(seq 1 30); do
                cf_url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_TUNNEL" | head -n 1 || true)
                if [ -n "$cf_url" ]; then
                    echo -e " ${GREEN}Ready!${NC}"
                    break
                fi
                printf "."
                sleep 1
            done

            if [ -n "$cf_url" ]; then
                echo -e "Public URL: ${BLUE}${cf_url}${NC}"
                local creds=$(grep '^ROSCRIBE_USERS=' "$ENV_FILE" | cut -d= -f2-)
                echo -e "Credentials: ${YELLOW}${creds}${NC}"
            else
                echo -e " ${RED}Failed to fetch public URL automatically.${NC} Check ${LOG_TUNNEL} for details."
            fi
        fi
    fi
}

build_pipeline() {
    echo -e "${GREEN}Running build & index pipeline...${NC}"
    bash scripts/build_all.sh
}

case "${1:-}" in
    start)
        shift
        start "$@"
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    build)
        build_pipeline
        ;;
    *)
        echo -e "${BOLD}ROScribe Smart Control Script${NC}"
        echo "Usage: ./roscribe_smart.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start [--public]   Start the smart app on port 8081 (local-only by default)"
        echo "  stop               Stop running ROScribe Smart processes"
        echo "  status             Check if smart processes are active"
        echo ""
        exit 1
        ;;
esac
