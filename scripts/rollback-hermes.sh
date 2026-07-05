#!/usr/bin/env bash
# Roll Hermes back off the com.axis.laplaced daemon (plan-0017 P3).
#
# Restores ~/.hermes/config.yaml from the ~/.hermes/config.yaml.pre-laplaced
# snapshot written by cutover-hermes.sh, kickstarts ai.hermes.gateway, and
# verifies discord reconnects. No secrets read; does NOT touch ~/.hermes/.env.
set -uo pipefail

CONFIG="${HOME}/.hermes/config.yaml"
BACKUP="${HOME}/.hermes/config.yaml.pre-laplaced"
GW_LABEL="ai.hermes.gateway"
GW_LOG="${HOME}/.hermes/logs/gateway.log"
DOMAIN="gui/$(id -u)"

say()  { printf '\n=== %s ===\n' "$1"; }
info() { printf '  info: %s\n' "$1"; }
ok()   { printf '  OK: %s\n' "$1"; }
die()  { printf '  ABORT: %s\n' "$1" >&2; exit 1; }

say "RESTORE — ${CONFIG} from ${BACKUP}"
[[ -f "${BACKUP}" ]] || die "no backup at ${BACKUP} — nothing to roll back to (was cutover ever run?)"
cp -p "${BACKUP}" "${CONFIG}"
[[ -f "${CONFIG}" ]] || die "restore copy failed"
ok "config restored from pre-laplaced backup"
say "RESTORED model block"
sed -n '/^model:/,/^[^[:space:]]/p' "${CONFIG}" | sed 's/^/    /'

say "KICKSTART — ${GW_LABEL}"
start_lines="$(wc -l < "${GW_LOG}" 2>/dev/null || echo 0)"
info "gateway.log line offset before kickstart: ${start_lines}"
launchctl kickstart -k "${DOMAIN}/${GW_LABEL}"
ok "kickstart -k issued for ${DOMAIN}/${GW_LABEL}"

say "VERIFY — gateway reconnect"
info "waiting up to 30s for '✓ discord connected'..."
connected=0
for _ in $(seq 1 30); do
    if tail -n "+$((start_lines + 1))" "${GW_LOG}" 2>/dev/null | grep -q 'discord connected'; then
        connected=1; break
    fi
    sleep 1
done
if [[ "${connected}" == "1" ]]; then
    ok "gateway reconnected to discord after rollback"
    tail -n "+$((start_lines + 1))" "${GW_LOG}" 2>/dev/null | grep -E 'discord connected|Connected as' | tail -2 | sed 's/^/    /'
else
    printf '  WARN: no new "discord connected" line within 30s. Recent gateway.log:\n'
    tail -6 "${GW_LOG}" 2>/dev/null | sed 's/^/    /'
fi

say "STATUS"
launchctl print "${DOMAIN}/${GW_LABEL}" 2>/dev/null | grep -E 'state =|pid =' | sed 's/^/    /' || info "launchctl print unavailable"
say "DONE — Hermes rolled back to pre-laplaced config"
