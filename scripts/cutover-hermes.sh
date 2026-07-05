#!/usr/bin/env bash
# Live Hermes cutover onto the com.axis.laplaced daemon (plan-0017 P3).
#
# Idempotent: repointing an already-cutover config re-writes the same values and
# re-kickstarts cleanly. Safe to re-run. Aborts BEFORE touching anything if the
# daemon is unhealthy or the target modelKey is not present in LM Studio.
#
# What it changes (and ONLY this):
#   ~/.hermes/config.yaml  model.base_url        -> http://localhost:4242/hermes/v1
#                          model.model           -> qwen3-next-80b-a3b-thinking
#                          model.context_length  -> 131072
# then kickstarts ai.hermes.gateway and verifies discord reconnects.
#
# A one-time backup ~/.hermes/config.yaml.pre-laplaced is written before the
# first edit (never overwritten). Use rollback-hermes.sh to restore it.
#
# Usage:
#   cutover-hermes.sh                 # full live cutover
#   cutover-hermes.sh --precheck-only # run pre-checks only, change nothing
#
# No secrets are read or needed. This script does NOT touch ~/.hermes/.env.
set -uo pipefail

CONFIG="${HOME}/.hermes/config.yaml"
BACKUP="${HOME}/.hermes/config.yaml.pre-laplaced"
DAEMON="http://127.0.0.1:4242"
TARGET_MODEL="qwen3-next-80b-a3b-thinking"
NEW_BASE_URL="http://localhost:4242/hermes/v1"
NEW_CTX="131072"
LMS="${HOME}/.lmstudio/bin/lms"
GW_LABEL="ai.hermes.gateway"
GW_LOG="${HOME}/.hermes/logs/gateway.log"
DAEMON_LOG="${HOME}/.laplace/logs/laplaced.log"
DOMAIN="gui/$(id -u)"

say()  { printf '\n=== %s ===\n' "$1"; }
info() { printf '  info: %s\n' "$1"; }
ok()   { printf '  OK: %s\n' "$1"; }
die()  { printf '  ABORT: %s\n' "$1" >&2; exit 1; }

# --- pre-check (a): daemon /healthz AND /readyz both good -------------------
precheck_daemon() {
    say "PRE-CHECK (a) — daemon /healthz + /readyz"
    local hc rc
    hc="$(curl -sS -o /dev/null -w '%{http_code}' "${DAEMON}/healthz" 2>/dev/null || true)"
    info "healthz http ${hc}"
    [[ "${hc}" == "200" ]] || die "daemon /healthz not 200 (got ${hc}) — is com.axis.laplaced running?"
    rc="$(curl -sS -o /dev/null -w '%{http_code}' "${DAEMON}/readyz" 2>/dev/null || true)"
    info "readyz http ${rc}"
    [[ "${rc}" == "200" ]] || die "daemon /readyz not 200 (got ${rc}) — upstream LM Studio not ready"
    ok "daemon healthy and ready"
}

# --- pre-check (b): target modelKey present in `lms ls --json` ---------------
precheck_model() {
    say "PRE-CHECK (b) — modelKey '${TARGET_MODEL}' present in lms ls"
    [[ -x "${LMS}" ]] || die "lms binary not found/executable at ${LMS}"
    local ls_json
    ls_json="$("${LMS}" ls --json 2>/dev/null || true)"
    [[ -n "${ls_json}" ]] || die "'lms ls --json' returned nothing"
    # Exact modelKey match; on failure print the actual candidate keys.
    if printf '%s' "${ls_json}" | python3 -c '
import json, sys
target = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception as exc:
    print("  parse error: %s" % exc, file=sys.stderr); sys.exit(2)
keys = [e.get("modelKey") for e in data if isinstance(e.get("modelKey"), str)]
if target in keys:
    sys.exit(0)
print("  candidates present:", file=sys.stderr)
for k in keys:
    print("    - %s" % k, file=sys.stderr)
sys.exit(1)
' "${TARGET_MODEL}"; then
        ok "exact modelKey '${TARGET_MODEL}' found in lms ls"
    else
        die "modelKey '${TARGET_MODEL}' NOT found in lms ls (candidates listed above)"
    fi
}

run_prechecks() {
    [[ -f "${CONFIG}" ]] || die "hermes config not found at ${CONFIG}"
    precheck_daemon
    precheck_model
}

# --- (c) one-time backup ----------------------------------------------------
backup_config() {
    say "BACKUP — ${BACKUP}"
    if [[ -f "${BACKUP}" ]]; then
        info "backup already present, leaving it untouched (first-cutover snapshot)"
    else
        cp -p "${CONFIG}" "${BACKUP}"
        [[ -f "${BACKUP}" ]] || die "backup copy failed"
        ok "backed up config to ${BACKUP}"
    fi
}

# --- (d) in-place edit of ONLY the three model.* keys -----------------------
edit_config() {
    say "EDIT — model.base_url / model.model / model.context_length"
    CONFIG="${CONFIG}" NEW_BASE_URL="${NEW_BASE_URL}" \
    TARGET_MODEL="${TARGET_MODEL}" NEW_CTX="${NEW_CTX}" python3 <<'PY'
import os, re, sys

path = os.environ["CONFIG"]
new = {
    "base_url": os.environ["NEW_BASE_URL"],
    "model": os.environ["TARGET_MODEL"],
    "context_length": os.environ["NEW_CTX"],
}
with open(path, "r") as fh:
    lines = fh.readlines()

in_model = False
changed = {k: False for k in new}
out = []
key_re = re.compile(r"^(\s+)(base_url|model|context_length)(\s*:\s*)(.*?)(\s*)$")
top_re = re.compile(r"^(\S+):")

for line in lines:
    # Track the top-level `model:` block. Any non-indented key ends it.
    m_top = top_re.match(line)
    if m_top:
        in_model = (m_top.group(1) == "model")
        out.append(line)
        continue
    if in_model:
        m = key_re.match(line)
        if m and m.group(2) in new and not changed[m.group(2)]:
            key = m.group(2)
            newline = "%s%s%s%s%s\n" % (m.group(1), key, m.group(3), new[key], "")
            out.append(newline)
            changed[key] = True
            continue
    out.append(line)

missing = [k for k, v in changed.items() if not v]
if missing:
    print("  edit ERROR: keys not found under model: %s" % missing, file=sys.stderr)
    sys.exit(1)

with open(path, "w") as fh:
    fh.writelines(out)
print("  wrote model.base_url=%s model.model=%s model.context_length=%s"
      % (new["base_url"], new["model"], new["context_length"]))
PY
    [[ $? -eq 0 ]] || die "config edit failed (keys not found) — restore from ${BACKUP} if partially written"
    ok "config edited in place (only model.base_url/model/context_length touched)"
    say "POST-EDIT model block"
    sed -n '/^model:/,/^[^[:space:]]/p' "${CONFIG}" | sed 's/^/    /'
}

# --- (e) kickstart the gateway ----------------------------------------------
kickstart_gateway() {
    say "KICKSTART — ${GW_LABEL}"
    launchctl kickstart -k "${DOMAIN}/${GW_LABEL}"
    ok "kickstart -k issued for ${DOMAIN}/${GW_LABEL}"
}

# --- (f) verify reconnect + daemon log clean --------------------------------
# The gateway.log offset MUST be captured by the caller BEFORE launchctl
# kickstart (mirrors rollback-hermes.sh), otherwise a normal fast reconnect
# lands behind the offset and this function false-warns "consider rollback" on a
# genuinely successful cutover.
verify_live() {
    local start_lines="$1"
    say "VERIFY — gateway reconnect + daemon log"
    info "gateway.log line offset (captured before kickstart): ${start_lines}"
    info "waiting up to 30s for '✓ discord connected' after the kickstart..."
    local connected=0 i
    for i in $(seq 1 30); do
        if tail -n "+$((start_lines + 1))" "${GW_LOG}" 2>/dev/null | grep -q 'discord connected'; then
            connected=1; break
        fi
        sleep 1
    done
    if [[ "${connected}" == "1" ]]; then
        ok "gateway reconnected to discord after cutover"
        tail -n "+$((start_lines + 1))" "${GW_LOG}" 2>/dev/null | grep -E 'discord connected|Connected as' | tail -2 | sed 's/^/    /'
    else
        printf '  WARN: no new "discord connected" line within 30s. Recent gateway.log:\n'
        tail -6 "${GW_LOG}" 2>/dev/null | sed 's/^/    /'
        printf '  Investigate the gateway before sending traffic; consider rollback-hermes.sh.\n'
    fi
    info "daemon log tail (checking for errors):"
    tail -8 "${DAEMON_LOG}" 2>/dev/null | sed 's/^/    /'
    if tail -40 "${DAEMON_LOG}" 2>/dev/null | grep -qiE ' ERROR | CRITICAL |Traceback'; then
        printf '  WARN: daemon log tail shows ERROR/Traceback lines above — inspect before trusting the cutover.\n'
    else
        ok "no ERROR/CRITICAL/Traceback in recent daemon log"
    fi
}

next_steps() {
    say "NEXT STEPS"
    cat <<EOF
  1. Send a Hermes task in Discord (a real, long-context one is ideal so the KV
     cache populates on ${TARGET_MODEL}).
  2. Confirm the admission signal:
       grep 'admission origin=hermes' ${DAEMON_LOG} | tail
     expect: origin=hermes tier=interactive model=${TARGET_MODEL} decision=admit
  3. With Hermes on next-80b AND the bot's qwen3.6-27b resident, run the
     measurement gate:
       scripts/measure-coresidency.sh
     It finalizes [model_footprint_mb] from real lms ps sizes (measured + 10%).
  4. If anything is wrong: scripts/rollback-hermes.sh
EOF
}

# --- main -------------------------------------------------------------------
main() {
    if [[ "${1:-}" == "--precheck-only" ]]; then
        run_prechecks
        say "PRE-CHECK ONLY — no changes made"
        ok "all pre-checks passed; config, gateway, and models left untouched"
        exit 0
    fi
    run_prechecks
    backup_config
    edit_config
    # Capture the gateway.log offset BEFORE kickstart so a fast reconnect is not
    # missed (see verify_live). Mirrors rollback-hermes.sh:29-31.
    local gw_offset
    gw_offset="$(wc -l < "${GW_LOG}" 2>/dev/null || echo 0)"
    info "gateway.log line offset before kickstart: ${gw_offset}"
    kickstart_gateway
    # No fixed pre-verify sleep: verify_live polls up to 30s for the reconnect
    # line from the pre-kickstart offset, which subsumes any settle wait.
    verify_live "${gw_offset}"
    next_steps
    say "DONE — Hermes cutover applied"
}

main "$@"
