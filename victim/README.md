# Model Extraction — Pritu's part (Group 9, CSE406 B2)

The **victim side**: the secret classifier, the server that exposes it over
the prediction API, and the two defence-relevant pieces the design report
assigns to this component (the layered defence, and the equation-solving
attack that motivates it).

## Files

| File | Owner | What it is |
|------|-------|-----------|
| `model.py` | **Pritu** | The secret classifier's architecture (`VictimCNN`) and checkpoint loader. Unrelated to any architecture the attacker's side uses. |
| `train_victim.py` | **Pritu** | Offline, one-shot: trains `VictimCNN` on MNIST (seed 1337) and saves `victim.pt` with its test accuracy — report Section 4.1, step 1. |
| `victim_server.py` | **Pritu** | Component A. Loads `victim.pt`, listens on TCP, decodes REQUEST frames, runs inference, routes the output through `defence.py`, returns RESPONSE frames. Threaded (one thread per connection). |
| `defence.py` | **Pritu** | Component F. Output minimisation (rounding / top-k / label-only), prediction perturbation (argmax-preserving noise), and a per-client sliding-window rate limiter — report Section 6. |
| `equation_solve.py` | **Pritu** | Component E ("Attack B"). Functionally-equivalent extraction of a **linear/softmax** victim via one batch of queries + ordinary least squares (no training loop) — report Section 4.3. Also runs (as a diagnostic) against the real CNN to show *why* that assumption doesn't hold there. |

`protocol.py` (one directory up, at the repo root) is the **shared** wire
format both sides import — see the top-level README. These files reach it
by inserting the repo root onto `sys.path`, so nothing here needs to be run
with `-m` or with `PYTHONPATH` set by hand.

## How to run

**1. Train the secret classifier once** (produces `victim.pt`, not committed
— it's gitignored, same as the attacker's `*.pt`/`*.npz` artifacts):
```bash
cd victim
python train_victim.py --epochs 5 --out victim.pt
```
Reference run (this repo, CPU, 3 epochs): **98.4%** MNIST test accuracy.

**2. Start the server**, undefended (a bare smoke test):
```bash
python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt
```
With the Section 6 defences on:
```bash
python victim_server.py --port 9009 --model victim.pt \
    --top-k 3 --noise-std 0.02 --rate-limit 2000 --rate-window 60
# other knobs: --round-decimals 2, --label-only
```
It answers every well-formed query (per the threat model, a single bad
request never gets the attacker blocked) — the only lever it holds is how
much information each answer leaks, which is exactly what the flags above
control.

**3. Point the attacker's client at it** (from a second terminal, in
`attacker/`) — this is the same command from `attacker/README.md`, just
against the real server instead of `mock_victim.py`:
```bash
cd ../attacker
python attack_client.py --host 127.0.0.1 --port 9009 \
    --pool mnist --data-dir ../data --queries 10000 --out transfer_10000.npz
```
The client reports which flags (rounding / top-k / noise) the responses
advertised, straight off the wire — no changes needed on that side to test
each defence on vs. off.

**4. Run the equation-solving attack** (component E), independent of the
server above — it can query a synthetic linear victim in-process, the real
CNN in-process (diagnostic), or the real server over the network:
```bash
# Sanity check: a genuine linear/softmax victim -> ~100% fidelity, no training.
python equation_solve.py --mode offline --target linear

# Diagnostic: the real trained CNN -> fidelity collapses (expected -- see below).
python equation_solve.py --mode offline --target cnn --checkpoint victim.pt

# Online: extract victim_server.py itself over the real wire protocol.
python equation_solve.py --mode network --host 127.0.0.1 --port 9009
```

## Why equation-solving "fails" against the CNN (on purpose)

`equation_solve.py --target cnn` is expected to score close to chance
(~10% label agreement on MNIST) — this is not a bug, it's the report's
point. Attack B's whole method rests on `logit_i(x) = w_i·x + b_i` being
*exactly* linear, which only holds for a true logistic/softmax model.
Querying the CNN, on the other hand, recovers a model that fits perfectly
*inside whatever single ReLU-activation region the random probes happened
to land in*, but that fit does not generalise to the real data manifold —
which is exactly why the script evaluates on real MNIST test images rather
than more probes from the training distribution. This is the documented
reason the report treats Attack B as the special linear-only case, and
leaves general CNN extraction to the attacker's learning-based attack
(`attacker/attack_client.py` + `attacker/knockoff_train.py`).

## Verifying each part works

Every file below has either a self-test (`python <file>.py` with no
network) or was exercised end-to-end against the *unmodified* attacker
code during development:

| File | How it was checked |
|------|--------------------|
| `model.py` | `python model.py` — forward-pass shape/param check. |
| `defence.py` | `python defence.py` — exercises rounding, top-k, label-only, noise (argmax preserved, valid distribution), the combined pipeline, and the rate limiter's sliding window. |
| `train_victim.py` | Run once; produced `victim.pt` at 98.4% MNIST test accuracy. |
| `victim_server.py` | Started standalone, then queried by **`attacker/attack_client.py`, unmodified**, in three configurations: no defence (flags advertised = probs only), `--top-k 3 --noise-std 0.02` (flags advertised = topk + noised, confirmed ≤3 nonzero classes per response), and `--rate-limit` (confirmed the client receives `ERROR code 4` once the quota is exceeded). |
| `equation_solve.py` | Offline vs. a synthetic linear/softmax victim → 100% fidelity, ~0 TV distance (matches the report's Attack-B claim). Offline vs. the real CNN, evaluated on real MNIST images → fidelity collapses to chance level, as expected. Network mode against the real `victim_server.py` reproduces the **exact same numbers** as the offline CNN run, confirming `protocol.py` + `victim_server.py` + this file are correctly wired together end to end. |

## Notes

- Seed fixed to `1337` everywhere (report requirement), matching the
  attacker side's convention.
- `victim.pt` and any `*.npz` you save here are gitignored, same policy as
  the attacker's artifacts.
- Metrics to report per the design report: accuracy, fidelity (label
  agreement with the victim), and adversarial-example transfer, each with
  defences on vs. off — `victim_server.py`'s CLI flags are what toggle
  those defences for the sweep.
