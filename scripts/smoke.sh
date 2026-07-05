#!/usr/bin/env bash
# Live smoke for the com.axis.laplaced daemon (plan-0017 P2).
#
# Runs the ratified P2 checklist in order against the RUNNING launchd service on
# :4242 and the real LM Studio on :1234, using ONLY the small model
# google/gemma-4-e4b. It never loads/unloads any other model itself; if it sees
# the daemon reap a non-gemma external model it records that as a FINDING.
#
# Secret handling: the LM Studio API key is read from $LMSTUDIO_API_KEY if set,
# else pulled from 1Password at runtime. It is captured into a variable, used
# only as a curl Authorization header, and NEVER echoed or written to a file.
# Do NOT add `set -x` and do NOT `curl -v` the authed calls (that would print
# the header).
set -uo pipefail

BASE="http://127.0.0.1:4242"
UPSTREAM="http://127.0.0.1:1234"
MODEL="google/gemma-4-e4b"
LMS="${HOME}/.lmstudio/bin/lms"
LOG="${HOME}/.laplace/logs/laplaced.log"
LABEL="com.axis.laplaced"
DOMAIN="gui/$(id -u)"
TMP="$(mktemp -d /tmp/laplace_smoke.XXXXXX)"
# Expected reap TTL for a ~9GB model: below the reaper's 15 GiB medium tier, so
# it falls to the default delay of 300s, swept at 60s granularity.
TTL_WAIT="${SMOKE_TTL_WAIT:-330}"

pass=0; fail=0; finding=0
declare -a RESULTS=()
say()      { printf '\n=== %s ===\n' "$1"; }
ok()       { printf '  PASS: %s\n' "$1";    RESULTS+=("PASS   $1"); pass=$((pass+1)); }
bad()      { printf '  FAIL: %s\n' "$1";    RESULTS+=("FAIL   $1"); fail=$((fail+1)); }
note()     { printf '  FINDING: %s\n' "$1"; RESULTS+=("FIND   $1"); finding=$((finding+1)); }
info()     { printf '  info: %s\n' "$1"; }

# --- API key (never echoed) -------------------------------------------------
if [[ -n "${LMSTUDIO_API_KEY:-}" ]]; then
    KEY="${LMSTUDIO_API_KEY}"
    info "API key: taken from \$LMSTUDIO_API_KEY"
else
    KEY="$(op read 'op://OpenClaw/LMStudio API Key/axis-fleet' 2>/dev/null || true)"
    info "API key: pulled from 1Password at runtime"
fi
AUTH=()
if [[ -n "${KEY}" ]]; then AUTH=(-H "Authorization: Bearer ${KEY}"); fi

chat_body() { # $1=prompt $2=stream(true/false)
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"stream":%s,"max_tokens":24}' \
        "${MODEL}" "$1" "$2"
}

# ===========================================================================
say "STEP 1 — /healthz 200"
code="$(curl -sS -o "${TMP}/healthz" -w '%{http_code}' "${BASE}/healthz")"
info "http ${code} body=$(cat "${TMP}/healthz")"
[[ "${code}" == "200" ]] && ok "healthz 200" || bad "healthz expected 200 got ${code}"

# ===========================================================================
say "STEP 2 — /readyz 200 (LM Studio up)"
code="$(curl -sS -o "${TMP}/readyz" -w '%{http_code}' "${BASE}/readyz")"
info "http ${code} body=$(cat "${TMP}/readyz")"
[[ "${code}" == "200" ]] && ok "readyz 200 (upstream reachable)" || bad "readyz expected 200 got ${code}"
info "readyz 503 path (documented, NOT exercised on this multi-tenant box): with LM Studio stopped, /readyz returns 503 with reason 'upstream /v1/models not 200 within 5s' and/or 'lms ps not rc 0 within 10s'"

# ===========================================================================
say "STEP 3 — non-streaming chat via /other/v1/chat/completions"
code="$(curl -sS --max-time 320 -o "${TMP}/chat_ns.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' "${AUTH[@]}" \
    -d "$(chat_body 'Say hello in exactly three words.' false)" \
    "${BASE}/other/v1/chat/completions")"
info "http ${code}"
if [[ "${code}" == "200" ]] && grep -q '"choices"' "${TMP}/chat_ns.json"; then
    info "content: $(python3 -c "import json,sys;d=json.load(open('${TMP}/chat_ns.json'));print(d['choices'][0]['message']['content'][:120])" 2>/dev/null)"
    ok "non-streaming chat completion 200 with choices"
else
    bad "non-streaming chat expected 200+choices got ${code}: $(head -c 200 "${TMP}/chat_ns.json")"
fi

# ===========================================================================
say "STEP 4 — streaming chat (stream:true), assert incremental chunks"
curl -sS --max-time 320 -N -o "${TMP}/chat_stream.sse" -w '%{http_code}' \
    -H 'Content-Type: application/json' "${AUTH[@]}" \
    -d "$(chat_body 'Count from one to five.' true)" \
    "${BASE}/other/v1/chat/completions" >"${TMP}/chat_stream.code" 2>/dev/null
code="$(cat "${TMP}/chat_stream.code")"
frames="$(grep -c '^data:' "${TMP}/chat_stream.sse" 2>/dev/null || echo 0)"
done_seen="$(grep -c '\[DONE\]' "${TMP}/chat_stream.sse" 2>/dev/null || echo 0)"
info "http ${code} data-frames=${frames} done-marker=${done_seen}"
if [[ "${code}" == "200" && "${frames}" -ge 2 ]]; then
    ok "streaming chat delivered ${frames} incremental data frames"
else
    bad "streaming chat expected >=2 data frames got ${frames} (http ${code})"
fi

# ===========================================================================
say "STEP 5 — streaming /other/v1/responses (proves D11 upstream support)"
curl -sS --max-time 320 -N -o "${TMP}/responses.sse" -w '%{http_code}' \
    -H 'Content-Type: application/json' "${AUTH[@]}" \
    -d "$(printf '{"model":"%s","input":"Say hi.","stream":true}' "${MODEL}")" \
    "${BASE}/other/v1/responses" >"${TMP}/responses.code" 2>/dev/null
code="$(cat "${TMP}/responses.code")"
rframes="$(grep -c '^data:' "${TMP}/responses.sse" 2>/dev/null || echo 0)"
info "http ${code} data-frames=${rframes} body-head=$(head -c 160 "${TMP}/responses.sse" | tr '\n' ' ')"
if [[ "${code}" == "200" && "${rframes}" -ge 1 ]]; then
    ok "responses passthrough streamed ${rframes} frames — D11 upstream support OBSERVED"
elif [[ "${code}" == "404" ]]; then
    note "responses: LM Studio returned 404 for /v1/responses — D11 upstream support NOT present; P4 rollback (api=openai-completions) would be required. NOT failing the smoke."
else
    note "responses: unexpected http ${code} (not 200, not 404). Recorded as finding, not a hard fail. body-head=$(head -c 160 "${TMP}/responses.sse" | tr '\n' ' ')"
fi

# ===========================================================================
say "STEP 6 — admission log line for gemma (origin=other -> default tier scheduled)"
adm="$(grep 'admission origin=' "${LOG}" 2>/dev/null | grep "model=${MODEL}" | tail -3)"
printf '%s\n' "${adm}" | sed 's/^/    /'
if printf '%s' "${adm}" | grep -q 'origin=other tier=scheduled' && printf '%s' "${adm}" | grep -q 'decision=admit'; then
    ok "admission line present: origin=other tier=scheduled decision=admit (/other/ -> default tier)"
else
    bad "expected admission line 'origin=other tier=scheduled ... decision=admit' for ${MODEL}"
fi

# ===========================================================================
say "STEP 7 — lms ps shows gemma loaded"
"${LMS}" ps 2>&1 | sed 's/^/    /'
if "${LMS}" ps 2>&1 | grep -q 'gemma-4-e4b'; then
    ok "lms ps shows gemma-4-e4b resident"
else
    bad "lms ps does not show gemma-4-e4b"
fi

# ===========================================================================
say "STEP 8 — wait past TTL (~${TTL_WAIT}s + sweep) and verify daemon reap"
info "expected TTL for ~9GB model = 300s (below 15GiB medium tier => default delay 300s), sweep every 60s"
before_freed="$(grep -c 'notify_freed' "${LOG}" 2>/dev/null || echo 0)"
info "notify_freed lines before wait: ${before_freed}"
info "sleeping ${TTL_WAIT}s then polling up to 180s for reap..."
sleep "${TTL_WAIT}"
reaped=0
for _ in $(seq 1 12); do
    if ! "${LMS}" ps 2>&1 | grep -q 'gemma-4-e4b'; then reaped=1; break; fi
    sleep 15
done
after_freed="$(grep -c 'notify_freed' "${LOG}" 2>/dev/null || echo 0)"
info "notify_freed lines after wait: ${after_freed}"
grep 'notify_freed' "${LOG}" 2>/dev/null | tail -2 | sed 's/^/    /'
if [[ "${reaped}" == "1" && "${after_freed}" -gt "${before_freed}" ]]; then
    ok "daemon reaped gemma after TTL (gone from lms ps + notify_freed logged)"
elif [[ "${reaped}" == "1" ]]; then
    ok "gemma gone from lms ps after TTL (no new notify_freed line captured — check log excerpt)"
else
    bad "gemma still resident after ${TTL_WAIT}s+poll — reap not observed"
fi
# guard: did the daemon reap anything that ISN'T gemma? (multi-tenant safety)
others="$(grep -E 'force_unload|notify_freed' "${LOG}" 2>/dev/null | grep -v 'gemma' | tail -3)"
if [[ -n "${others}" ]]; then
    note "daemon log shows non-gemma unload/free activity during smoke (multi-tenant coexistence, D6). Excerpt: ${others}"
fi

# ===========================================================================
say "STEP 9 — kickstart -k then /healthz again (survival)"
launchctl kickstart -k "${DOMAIN}/${LABEL}"
hc=""
for _ in $(seq 1 40); do
    hc="$(curl -sS -o /dev/null -w '%{http_code}' "${BASE}/healthz" 2>/dev/null || true)"
    [[ "${hc}" == "200" ]] && break
    sleep 0.3
done
info "post-kickstart healthz http ${hc}"
[[ "${hc}" == "200" ]] && ok "service survived kickstart -k, healthz 200" || bad "healthz not 200 after kickstart (got ${hc})"

# ===========================================================================
say "STEP 10 — roster check: lms ls ids vs [model_context] keys"
"${LMS}" ls --json >"${TMP}/ls.json" 2>/dev/null || echo '[]' >"${TMP}/ls.json"
python3 - "$@" <<PY
import json, os, tomllib
ls = json.load(open("${TMP}/ls.json"))
ids = set()
for e in ls:
    for k in ("modelKey", "path", "identifier"):
        v = e.get(k)
        if isinstance(v, str) and v:
            ids.add(v); break
cfg = tomllib.load(open(os.path.expanduser("~/.laplace/laplaced.toml"), "rb"))
mapped = set(cfg.get("model_context", {}))
gaps = sorted(i for i in ids if i not in mapped)
extra = sorted(m for m in mapped if m not in ids)
print("    roster ids: %d, [model_context] keys: %d" % (len(ids), len(mapped)))
print("    roster ids MISSING a [model_context] entry (fall back to default 64000): %s" % (gaps or "none"))
print("    [model_context] keys NOT in live roster (config-only): %s" % (extra or "none"))
PY
info "roster gaps above are recorded as findings (they pick default_context_length=64000 by accident)"
note "roster: see gap list above (lms ls ids without a [model_context] entry)"

# ===========================================================================
say "STEP 11 — concurrency probe: 3 simultaneous small chat requests (R5 part 1)"
info "note: gemma was reaped in step 8; the first of these will trigger a reload, which skews its wall time"
probe() { # $1=index
    local t0 t1 code
    t0="$(python3 -c 'import time;print(time.time())')"
    code="$(curl -sS --max-time 320 -o "${TMP}/probe_$1.json" -w '%{http_code}' \
        -H 'Content-Type: application/json' "${AUTH[@]}" \
        -d "$(chat_body 'Reply with a single short sentence.' false)" \
        "${BASE}/other/v1/chat/completions")"
    t1="$(python3 -c 'import time;print(time.time())')"
    printf '%s %s %s\n' "$1" "${code}" "$(python3 -c "print(round(${t1}-${t0},2))")" >"${TMP}/probe_$1.time"
}
gstart="$(python3 -c 'import time;print(time.time())')"
probe 1 & probe 2 & probe 3 &
wait
gend="$(python3 -c 'import time;print(time.time())')"
wall="$(python3 -c "print(round(${gend}-${gstart},2))")"
sum=0; okc=0
for i in 1 2 3; do
    read -r idx c d <"${TMP}/probe_$i.time"
    info "req ${idx}: http ${c} wall ${d}s"
    [[ "${c}" == "200" ]] && okc=$((okc+1))
    sum="$(python3 -c "print(round(${sum}+${d},2))")"
done
info "total wall for all 3 = ${wall}s ; sum of individual = ${sum}s"
verdict="$(python3 -c "print('PARALLEL (overlapped)' if ${wall} < ${sum}*0.8 else 'SERIALIZED (near-sum)')")"
info "concurrency verdict: ${verdict} (wall ${wall}s vs sum ${sum}s; max_concurrent=2 broker cap)"
if [[ "${okc}" == "3" ]]; then
    ok "concurrency probe: 3/3 requests succeeded; behavior=${verdict}"
else
    note "concurrency probe: only ${okc}/3 succeeded (broker cap=2 may 503 the 3rd with Retry-After:30) — verdict ${verdict}"
fi

# ===========================================================================
say "SUMMARY"
for r in "${RESULTS[@]}"; do printf '  %s\n' "${r}"; done
printf '\n  totals: PASS=%d FAIL=%d FINDING=%d\n' "${pass}" "${fail}" "${finding}"
rm -rf "${TMP}"
[[ "${fail}" == "0" ]] && exit 0 || exit 1
