# Model Extraction — Arian's part (Group 9, CSE406 B2)

Black-box model-stealing attack against a prediction-API classifier.
This folder holds the **attacker side** plus the shared protocol and a dev
stand-in for the victim, so it can be built and tested without waiting on
Pritu's real server.

## Files

| File | Owner | What it is |
|------|-------|-----------|
| `protocol.py` | shared | Wire format: 12-byte header `">HBBII"`, REQUEST/RESPONSE/ERROR frames, length-prefixed reassembly. Both sides import this. |
| `mock_victim.py` | Arian (dev aid) | Throwaway TCP server that speaks the protocol so the client can be tested. **Not graded** — swap for Pritu's `victim_server.py` at integration. |
| `attack_client.py` | **Arian** | Online phase: query engine + response parser → writes `transfer_set.npz`. |
| `knockoff_train.py` | **Arian** | Offline phase: soft-label distillation → trains and saves the knockoff `f_hat`. |
| `adv_transfer_eval.py` | **Arian** | Evaluation only (not part of the attack): loads a trained knockoff and the real victim checkpoint, reports accuracy, fidelity, and hand-implemented FGSM adversarial-example transfer. Evaluates on the MNIST *train* split by default, since `--pool mnist` draws queries from the *test* split. |

Pritu owns `victim_server.py`, `defence.py`, and `equation_solve.py`.

## Pipeline

```
build query pool → send REQUEST frames → record RESPONSE probs
      → transfer_set.npz → (offline) distill knockoff → evaluate
```

## How to run

Full setup and every command used to produce the Final Report's results
(server configs, budget sweeps, knockoff training, evaluation) live in the
top-level [README.md](../README.md#setup) — kept in one place rather than
duplicated here, since two copies of the same commands drift out of sync
(see `protocol.py` below for exactly that problem). The short version:
`mock_victim.py` stands in for `victim_server.py` when you just want to
smoke-test `attack_client.py`/`knockoff_train.py` without training a real
victim first; everything else points at [README.md](../README.md#reproducing-the-final-reports-results).

Pools for `attack_client.py --pool`: `mnist` (in-distribution), `fashion`
(out-of-distribution), `uniform` (synthetic OOD / offline fallback).

## Integration with Pritu

The only hard dependency is `protocol.py` — as long as both sides pack/unpack
the same bytes, the client doesn't change when the mock is replaced by the real
server. To test defences, run the client against the real server with each
`defence.py` setting on vs. off; the client reports which flags (rounding /
top-k / noise) the responses advertised via the FLAGS byte.

## Notes

- Seeds fixed to `1337` for reproducibility (report requirement).
- `protocol.py` has a built-in self-test: `python protocol.py`.
- Metrics to report: accuracy, fidelity (label agreement with the victim),
  and adversarial-example transfer, each with defences on vs. off.
