#!/usr/bin/env bash
# P3 co-residency measurement gate (plan-0017 D13). READ-ONLY.
#
# Polls `lms ps --json` N times and records, per snapshot, each loaded model's
# (id, size, status). It NEVER loads or unloads anything. Run it while Hermes is
# live on qwen3-next-80b-a3b-thinking (ideally after a long-context task so the
# KV cache is populated) AND the bot's qwen3.6-27b is resident.
#
# At the end it prints:
#   - max observed size per model
#   - whether the two focus models were EVER simultaneously resident
#   - their summed max size vs the 87040 MiB budget
#   - recommended [model_footprint_mb] = measured max + 10%, as paste-ready TOML
#
# Size unit matches the library/broker: MiB (bytes / 1024 / 1024), budget in MiB.
#
# Usage: measure-coresidency.sh [N] [INTERVAL_S]   (defaults: 12, 10)
set -uo pipefail

N="${1:-12}"
INTERVAL="${2:-10}"
BUDGET_MB="87040"
LMS="${HOME}/.lmstudio/bin/lms"
# Focus models: substring -> canonical [model_footprint_mb] config key.
FOCUS_A_MATCH="qwen3-next-80b"
FOCUS_A_KEY="qwen3-next-80b-a3b-thinking"
FOCUS_B_MATCH="qwen3.6-27b"
FOCUS_B_KEY="qwen3.6-27b-mlx"

[[ -x "${LMS}" ]] || { printf 'ABORT: lms not executable at %s\n' "${LMS}" >&2; exit 1; }

SNAP="$(mktemp /tmp/laplace_coresidency.XXXXXX.jsonl)"
trap 'rm -f "${SNAP}"' EXIT

printf '=== P3 co-residency measurement — N=%s interval=%ss budget=%s MiB ===\n' \
    "${N}" "${INTERVAL}" "${BUDGET_MB}"
printf 'focus: %s (-> %s) + %s (-> %s)\n\n' \
    "${FOCUS_A_MATCH}" "${FOCUS_A_KEY}" "${FOCUS_B_MATCH}" "${FOCUS_B_KEY}"

for i in $(seq 1 "${N}"); do
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    raw="$("${LMS}" ps --json 2>/dev/null || echo '[]')"
    # Parse one snapshot: print a table row per loaded model, append a JSONL
    # summary line for the final aggregation. Field precedence mirrors
    # laplace/adapters/lmstudio.py (_entry_identifier/_entry_size_bytes/_entry_status).
    # NOTE: the JSON must ride an env var, not a pipe: python3 <<heredoc takes
    # its PROGRAM from stdin, so piped data would be silently discarded.
    RAW="${raw}" TS="${ts}" IDX="${i}" NTOTAL="${N}" \
        FA="${FOCUS_A_MATCH}" FB="${FOCUS_B_MATCH}" SNAP="${SNAP}" python3 <<'PY'
import json, os, sys

ts = os.environ["TS"]; idx = os.environ["IDX"]; ntotal = os.environ["NTOTAL"]
fa = os.environ["FA"]; fb = os.environ["FB"]; snap = os.environ["SNAP"]

def ident(e):
    for k in ("identifier", "modelKey", "path"):
        v = e.get(k)
        if isinstance(v, str) and v:
            return v
    return ""

def size_bytes(e):
    for k in ("sizeBytes", "size_bytes", "loadedSizeBytes", "loaded_size_bytes"):
        v = e.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None

def status(e):
    v = e.get("status")
    return v.lower() if isinstance(v, str) and v else "-"

try:
    data = json.loads(os.environ.get("RAW") or "[]")
except Exception as exc:
    print("  [%s/%s] %s  parse error: %s" % (idx, ntotal, ts, exc))
    data = []

print("  [%s/%s] %s" % (idx, ntotal, ts))
rows = []
if not data:
    print("      (no models loaded)")
for e in data:
    mid = ident(e)
    if not mid:
        continue
    sb = size_bytes(e)
    mib = round(sb / 1024 / 1024) if sb else None
    st = status(e)
    focus = ""
    if fa in mid:
        focus = "  <-- FOCUS A"
    elif fb in mid:
        focus = "  <-- FOCUS B"
    print("      %-48s %8s MiB  status=%s%s"
          % (mid[:48], mib if mib is not None else "?", st, focus))
    rows.append({"id": mid, "mib": mib, "status": st})

with open(snap, "a") as fh:
    fh.write(json.dumps({"ts": ts, "models": rows}) + "\n")
PY
    if [[ "${i}" -lt "${N}" ]]; then sleep "${INTERVAL}"; fi
done

printf '\n=== AGGREGATE ===\n'
SNAP="${SNAP}" BUDGET_MB="${BUDGET_MB}" \
FA="${FOCUS_A_MATCH}" FA_KEY="${FOCUS_A_KEY}" \
FB="${FOCUS_B_MATCH}" FB_KEY="${FOCUS_B_KEY}" python3 <<'PY'
import json, math, os

snap = os.environ["SNAP"]; budget = int(os.environ["BUDGET_MB"])
fa = os.environ["FA"]; fa_key = os.environ["FA_KEY"]
fb = os.environ["FB"]; fb_key = os.environ["FB_KEY"]

snapshots = []
with open(snap) as fh:
    for line in fh:
        line = line.strip()
        if line:
            snapshots.append(json.loads(line))

# Max observed size per model id across all snapshots.
max_mib = {}
for s in snapshots:
    for m in s["models"]:
        if m["mib"] is None:
            continue
        cur = max_mib.get(m["id"])
        if cur is None or m["mib"] > cur:
            max_mib[m["id"]] = m["mib"]

print("max observed size per model (MiB):")
if not max_mib:
    print("  (nothing was ever resident during the run)")
for mid in sorted(max_mib):
    print("  %-48s %8d MiB" % (mid[:48], max_mib[mid]))

def find_focus(match):
    hits = {mid: v for mid, v in max_mib.items() if match in mid}
    if not hits:
        return None, None
    mid = max(hits, key=hits.get)
    return mid, hits[mid]

a_id, a_max = find_focus(fa)
b_id, b_max = find_focus(fb)

# Simultaneous residency: any single snapshot with both focus models present.
both_resident = False
for s in snapshots:
    ids = [m["id"] for m in s["models"]]
    if any(fa in x for x in ids) and any(fb in x for x in ids):
        both_resident = True
        break

print("\nfocus A (%s): %s" % (fa_key, ("%d MiB (%s)" % (a_max, a_id)) if a_id else "NEVER OBSERVED"))
print("focus B (%s): %s" % (fb_key, ("%d MiB (%s)" % (b_max, b_id)) if b_id else "NEVER OBSERVED"))
print("both focus models simultaneously resident in a single snapshot: %s"
      % ("YES" if both_resident else "NO"))

if a_max is not None and b_max is not None:
    total = a_max + b_max
    print("\nsum of max sizes: %d + %d = %d MiB  vs budget %d MiB  -> %s"
          % (a_max, b_max, total, budget,
             "FITS" if total <= budget else "OVER BUDGET"))
    if total > budget:
        print("  Co-residency is genuinely infeasible at this ctx (D13): accept eviction")
        print("  churn as CORRECT behavior; do NOT understate footprints to force it.")
else:
    print("\nsum vs budget: cannot compute (one or both focus models never measured)")

def rec(v):
    return int(math.ceil(v * 1.1)) if v is not None else None

ra, rb = rec(a_max), rec(b_max)
print("\nrecommended [model_footprint_mb] (measured max + 10%):")
if ra is not None:
    print("  %-32s measured %d -> recommend %d MiB" % (fa_key, a_max, ra))
if rb is not None:
    print("  %-32s measured %d -> recommend %d MiB" % (fb_key, b_max, rb))
if ra is None and rb is None:
    print("  (no focus model measured — rerun while both are resident)")

print("\npaste-ready TOML for ~/.laplace/laplaced.toml [model_footprint_mb]:")
print("  [model_footprint_mb]")
if ra is not None:
    print('  "%s" = %d' % (fa_key, ra))
if rb is not None:
    print('  "%s" = %d' % (fb_key, rb))
if ra is not None and rb is not None:
    fit = "FITS" if (ra + rb) <= budget else "OVER BUDGET"
    print("  # recommended sum %d MiB vs budget %d MiB -> %s" % (ra + rb, budget, fit))
PY
