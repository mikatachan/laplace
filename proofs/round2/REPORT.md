# Round 2 duplicate-load guard

Verdict: B1/B2 reproduced and patched; ready for independent review, not deployed.
Both findings were correct. No finding was dismissed as an environment failure.
The code and test changes are on top of 6072abf, not an amendment.

## Behavior

- Fresh residency precedes the load lock and failure guards. One matching resident
  with enough context admits during another load (test bound: 200 ms with load held
  on an event until the resident admission finishes).
- Uncertainty is per model. Nonzero exit plus no new instances clears immediately;
  a failed post-load inventory read can reconcile on the next admission.
- Successful exact residency accepts concurrent appearances of other models.
  Known other catalog keys are not fuzzy candidates.
- Verification never unloads newly seen instances: snapshots cannot prove client
  ownership. Fuzzy outcomes cannot retry, including after unrelated catalog changes.
  Verified sufficient residency clears the failure record and uncertainty.
- Context shortfall or unknown context refuses without duplicating or unloading.
  The next request reads context again. Idle unload/reload is intentionally omitted:
  Laplace reservations cannot establish that external clients are idle.

Safety/recovery limit: an absent model after a timeout/cancellation or an unobserved
successful command is not evidence that the server stopped loading. Automatic retry
cannot be both safe and guaranteed recoverable from `ps` alone. The restriction is
per model, and a later verified resident recovers. If that never happens, operator
reconciliation is required. README states this limitation. Concurrent external loads
of the same model remain outside the process-local single-flight guarantee.

## Reproduction and raw evidence

Run from this worktree with its existing `.venv` (no LM Studio invocation):

```sh
.venv/bin/python proofs/round2/run.py old tests/test_load_recovery.py -q
.venv/bin/python proofs/round2/run.py new tests/test_load_recovery.py tests/test_duplicate_load_guard.py tests/test_lmstudio.py -q
.venv/bin/python proofs/round2/run.py mutation tests/test_load_recovery.py tests/test_duplicate_load_guard.py -q
.venv/bin/python -m pytest -q
```

The harness loads the actual adapter source from `git show 6072abf` for the old
run. Mutation replaces only `ensure_loaded` with the actual implementation from
`6072abf^`, retaining the surrounding current module. It never rewrites the source
file or imports a live checkout. All eleven added tests fail against 6072abf.

- [Fail on old: 11 failed](fail-on-old.txt)
- [Pass on new: 42 passed](pass-on-new.txt)
- [Guard reverted: 32 failed, 3 passed](mutation.txt). Includes round-1 fuzzy
  no-retry, single-flight, and failed-inventory protections.
- [Full suite: 139 passed](full-suite.txt)
- [Initial existing-test run: 3 failed, 28 passed](initial-existing.txt).
  The failures were obsolete automatic-cleanup and global-block assertions, plus
  the expected additional pre-lock inventory read. The updated assertions preserve
  fuzzy no-retry and require no unowned unload and no cross-model block.

Raw final full-suite tail:

```text
........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 1.36s
```

These are implementer-run regression results, not independent certification.

## Discord and coordinated live edits

No change to axis-discord 20f896a05d55c19a8a75064fb37314f4f7b01ac8.
Its four fleet-drift failures are real drift against the live OpenClaw allowlist,
not environment failures. The independent review reports 5/5 passing with a copied
openclaw.json containing the renamed ID; that copy experiment was not rerun here.
Deploy 20f896a together with the OpenClaw configuration edit.

Rename `huihui-qwen3.6-27b-abliterated-4-msq` to
`huihui-qwen3.6-27b-abliterated-4.5bit-msq` in these deployment items:

1. `~/.laplace/laplaced.toml`: configured `model_context` and `model_footprint_mb`
   keys. Live contents/line numbers not inspected this round. The default path is
   observed in this worktree's `laplace_serve/__main__.py`.
2. `~/.openclaw/openclaw.json`: local provider model registration, model allowlist
   keys, and agent/default model references using the old ID. Coordinate with the
   Discord commit. Exact live locations not re-enumerated this round.
3. `~/.openclaw/agents/main/agent/models.json:13`.
4. `~/.openclaw/agents/kamille/agent/models.json:13`.
5. Deploy the Discord worktree changes (config.yaml, role defaults, router and
   shadow IDs) as part of 20f896a; do not independently patch the live checkout.

The last two model-cache locations were observed using only the requested grep:

```text
/Users/axis/.openclaw/agents/main/agent/models.json:13:          "id": "huihui-qwen3.6-27b-abliterated-4-msq",
/Users/axis/.openclaw/agents/kamille/agent/models.json:13:          "id": "huihui-qwen3.6-27b-abliterated-4-msq",
```

`nomic-embed-text` in model_context is also a noncatalog key per the review;
its separate configuration correction remains outside this patch.

## UNVERIFIED

- Independent cross-provider validation of this revision is pending.
- Real LM Studio behavior, server-side pending-operation completion, and production
  admission latency were not exercised; the probes use a stateful CLI stub.
- Live Laplace/OpenClaw config reconciliation and deployment were not performed.
- Discord's copied-config 5/5 result is reviewer evidence, not a new run here.
- MC `/api/tasks` returned `{"error":"Unauthorized"}`. Board consultation/card
  update could not be completed without credentials; no secret files were read.

No live files were changed, no services restarted, and nothing was pushed.
