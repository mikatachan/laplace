#!/usr/bin/env bash
# Idempotent installer for the com.axis.laplaced launchd service (plan-0017 P2).
#
# - creates ~/.laplace/logs
# - writes the ratified default laplaced.toml IF ABSENT (never clobbers a hand-edited config)
# - installs the package in editable mode with the serve extra into the repo venv
# - refuses to start if port 4242 is already taken by something else
# - bootout-wait-then-bootstrap: bootout is async, so poll until the service is
#   gone before bootstrapping (avoids the "service already loaded" race)
# - waits for /healthz to come green before declaring success
set -euo pipefail

REPO="/Users/axis/github/laplace"
VENV_PY="${REPO}/.venv/bin/python"
VENV_PIP="${REPO}/.venv/bin/pip"
LAPLACE_HOME="${HOME}/.laplace"
LOG_DIR="${LAPLACE_HOME}/logs"
CONFIG="${LAPLACE_HOME}/laplaced.toml"
LABEL="com.axis.laplaced"
PLIST_SRC="${REPO}/deploy/${LABEL}.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
PORT=4242
DOMAIN="gui/$(id -u)"

echo "==> laplaced install starting"

# --- dirs -------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
mkdir -p "${HOME}/Library/LaunchAgents"

# --- default config (only if absent) ----------------------------------------
if [[ -f "${CONFIG}" ]]; then
    echo "==> config already present, leaving it untouched: ${CONFIG}"
else
    echo "==> writing default config: ${CONFIG}"
    cat >"${CONFIG}" <<'TOML'
bind = "127.0.0.1"
port = 4242
upstream = "http://127.0.0.1:1234"
budget_mb = 0                 # 0 = auto (sysctl, 90112 fallback)
max_concurrent = 2
admit_timeout_s = 120
fail_open = true
reaper_sweep = true
sweep_interval_s = 60
load_timeout_s = 300
keep_loaded = ["nomic-embed-text"]
default_context_length = 64000
log_rotate_mb = 10
log_backups = 5

[origins]
hermes = "interactive"   # ratified: user waits in Discord threads
openclaw = "interactive"
default = "scheduled"

[model_context]                # operating ctx, not max window (D12)
"qwen3-next-80b-a3b-thinking" = 131072
"qwen/qwen3-coder-30b" = 131072
"qwen3.6-27b-mlx" = 131072     # max 262144; 131k operating — ratify
"google/gemma-4-26b-a4b-qat" = 64000
"google/gemma-4-e4b" = 32768
"txgsync/gpt-oss-120b-derestricted" = 64000
"glm-4.5-air-106b" = 64000
"nomic-embed-text" = 2048

[model_footprint_mb]           # provisional, unvalidated — P3 gate finalizes (D13)
"qwen3-next-80b-a3b-thinking" = 48000
"qwen3.6-27b-mlx" = 31500
TOML
fi

# --- install package --------------------------------------------------------
if [[ ! -x "${VENV_PIP}" ]]; then
    echo "ERROR: venv pip not found at ${VENV_PIP}" >&2
    exit 1
fi
echo "==> pip install -e .[serve]"
( cd "${REPO}" && "${VENV_PIP}" install -e ".[serve]" )

# --- validate the config actually parses under the daemon loader ------------
echo "==> validating config parses"
"${VENV_PY}" -c "from laplace_serve.config import LaplacedConfig; c=LaplacedConfig.from_toml_file('${CONFIG}'); print('config ok: bind=%s port=%s upstream=%s sweep=%s' % (c.bind, c.port, c.upstream, c.reaper_sweep))"

# --- port check -------------------------------------------------------------
# If 4242 is held by our own already-running service, that's fine (we bootout below).
# If it is held by something else, refuse rather than fight over the port.
if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    HOLDER_CMD="$(lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | head -1)"
    if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
        echo "==> port ${PORT} held by our own ${LABEL} (will bootout/rebootstrap)"
    else
        echo "ERROR: port ${PORT} already in use by '${HOLDER_CMD:-unknown}' and it is not ${LABEL}" >&2
        exit 1
    fi
fi

# --- install plist ----------------------------------------------------------
echo "==> installing plist -> ${PLIST_DST}"
plutil -lint "${PLIST_SRC}"
cp "${PLIST_SRC}" "${PLIST_DST}"

# --- bootout (async) then wait until gone -----------------------------------
if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
    echo "==> booting out existing ${LABEL} (async)"
    launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
    for _ in $(seq 1 50); do
        if ! launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.2
    done
    if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
        echo "ERROR: ${LABEL} still loaded after bootout wait" >&2
        exit 1
    fi
    echo "==> old service gone"
fi

# --- bootstrap --------------------------------------------------------------
echo "==> bootstrapping ${LABEL}"
launchctl bootstrap "${DOMAIN}" "${PLIST_DST}"

# --- wait for healthz -------------------------------------------------------
echo "==> waiting for /healthz"
for _ in $(seq 1 50); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/healthz" 2>/dev/null || true)"
    if [[ "${code}" == "200" ]]; then
        echo "==> healthz 200 — laplaced is up"
        launchctl print "${DOMAIN}/${LABEL}" 2>/dev/null | grep -E 'state =|pid =' || true
        exit 0
    fi
    sleep 0.3
done

echo "ERROR: /healthz never returned 200; check ${LOG_DIR}/laplaced.log and ${LOG_DIR}/launchd.err" >&2
exit 1
