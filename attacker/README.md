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

Pritu owns `victim_server.py`, `defence.py`, and `equation_solve.py`.

## Pipeline

```
build query pool → send REQUEST frames → record RESPONSE probs
      → transfer_set.npz → (offline) distill knockoff → evaluate
```

## How to run

Terminal 1 — start a victim (mock for now; the real one later, same protocol):
```bash
python mock_victim.py --host 127.0.0.1 --port 9009
# or point it at a checkpoint to smoke-test fidelity:
# python mock_victim.py --port 9009 --model victim.pt
```

Terminal 2 — run the attack, sweeping the query budget:
```bash
for q in 1000 5000 10000 30000; do
  python attack_client.py --host 127.0.0.1 --port 9009 \
      --pool mnist --queries $q --out transfer_${q}.npz
done
```
Pools: `mnist` (in-distribution), `fashion` (out-of-distribution), `uniform`
(synthetic OOD / offline fallback).

Train and evaluate the knockoff (offline, no network):
```bash
python knockoff_train.py --transfer transfer_10000.npz \
    --epochs 20 --T 2.0 --out knockoff_10000.pt
```

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
