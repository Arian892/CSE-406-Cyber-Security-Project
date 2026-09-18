# Model Extraction: Stealing a Black-Box Classifier

CSE406 (Cyber Security Sessional), Group 9 (Section B2)
— Pritu Dhar Dhip (2105109) · Arian Mahtab Chowdhury (2105106)

A model-extraction ("model-stealing") attack against a machine-learning
classifier exposed only through a network prediction API, plus the layered
defences that raise the cost of stealing it. Full design and threat model
are in `Model_Extraction_Design_Report.pdf`; this README covers the code
layout and how to run it.

## Layout

```
protocol.py          <- SHARED wire format (see below) — canonical copy
victim/               <- Pritu's part (the victim side)
    model.py              secret classifier architecture (VictimCNN)
    train_victim.py       trains it on MNIST, saves victim.pt
    victim_server.py      Component A: serves it over TCP
    defence.py            Component F: rounding / top-k / noise / rate-limit
    equation_solve.py     Component E: Attack B, functionally-equivalent
                           extraction of a linear/softmax victim
    README.md              victim-side details, run instructions, test log
attacker/             <- Arian's part (the attacker side), unmodified
    attack_client.py      Component C: query engine + response parser
    knockoff_train.py     Component D: offline soft-label distillation
    mock_victim.py         dev stand-in for victim_server.py
    protocol.py            vendored copy of the shared protocol (below)
    data/                  torchvision MNIST cache (gitignored) — both
                           sides point here, see the run commands below
    README.md              attacker-side details, run instructions
```

### Why `protocol.py` appears twice

`protocol.py` is the one file both sides must agree on byte-for-byte (the
12-byte header, `REQUEST`/`RESPONSE`/`ERROR` framing — see the design
report, Section 3). The canonical copy lives at the repo root, unchanged
since Arian wrote it. `victim/*.py` reaches it directly (each file inserts
the repo root onto `sys.path` before importing it — no path juggling
needed by you). `attacker/protocol.py` is a **byte-identical vendored
copy**, added only so the attacker's files could move into their own
folder without editing a single line of Arian's code — his scripts still
say plain `import protocol as P` and resolve it from their own directory,
exactly as before the reorganisation. If `protocol.py` ever changes, copy
it into `attacker/protocol.py` again (a one-line `cp`).

## End-to-end run

Every command below is a single line with no shell-specific syntax (no
`for` loops, no `\` line continuation, no `$VAR`), so it runs unchanged in
Windows PowerShell, Windows `cmd.exe`, and any Linux/macOS shell.

Terminal 1 — train and serve the real victim. The MNIST cache lives at
`attacker/data/`, so from `victim/` that's `../attacker/data`:
```
cd victim
python train_victim.py --epochs 5 --out victim.pt --data-dir ../attacker/data
python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt
```
To serve with the Section 6 defences on instead, run this in place of the
last line above:
```
python victim_server.py --port 9009 --model victim.pt --top-k 3 --noise-std 0.02 --rate-limit 2000 --rate-window 60
```

Terminal 2 — run the attack (unmodified from Arian's original workflow),
sweeping the query budget. No `--data-dir` needed here: both
`attack_client.py` and `knockoff_train.py` default to `./data`, which is
now exactly where the cache lives (`attacker/data/`):
```
cd attacker
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 1000 --out transfer_1000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 5000 --out transfer_5000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 10000 --out transfer_10000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 30000 --out transfer_30000.npz
python knockoff_train.py --transfer transfer_10000.npz --epochs 20 --T 2.0 --out knockoff_10000.pt
```

The only thing that changes between "mock victim" and "real victim" is the
`--port` you point the client at — same protocol, same client code.

Independently, the equation-solving attack (component E) can also target
the real server directly (again, the MNIST cache is `../attacker/data`
from here):
```
cd victim
python equation_solve.py --mode network --host 127.0.0.1 --port 9009 --data-dir ../attacker/data
```

See `victim/README.md` and `attacker/README.md` for each side's details,
defence-toggle knobs, and how each file was individually verified.

## Reproducing the Final Report's Results

Every number and console excerpt in `Final Report/Model_Extraction_Final_Report.pdf`
came from the commands below, run in this exact order. All paths are relative
to the repo root; create the output folders once:

```
mkdir -p results/campaigns results/knockoffs
```

**Important gotcha we hit ourselves:** each step below that starts a new
`victim_server.py` with different `--top-k`/`--noise-std`/`--label-only`/
`--rate-limit` flags needs the *previous* server fully stopped first. If you
just background it and re-launch with new flags without confirming the old
process is dead, the new flags never take effect and you silently query the
old (wrong) server — that happened to us mid-way through building this
report. After stopping a server, confirm the port is free before starting
the next one:
```
python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 9009)); print('free'); s.close()"
```

### 1. Victim setup (once)

```
cd victim
python train_victim.py --epochs 5 --out victim.pt --data-dir ../attacker/data
```

### 2. Attack A — undefended budget sweep + out-of-distribution

Terminal 1:
```
cd victim
python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt
```
Terminal 2, one run per budget:
```
cd attacker
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 1000  --data-dir ./data --out ../results/campaigns/b1000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 5000  --data-dir ./data --out ../results/campaigns/b5000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 10000 --data-dir ./data --out ../results/campaigns/b10000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 30000 --data-dir ./data --out ../results/campaigns/b30000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool fashion --queries 10000 --data-dir ./data --out ../results/campaigns/ood10000.npz
```
Attack B, network mode, against the same undefended server:
```
cd victim
python equation_solve.py --mode network --host 127.0.0.1 --port 9009 --data-dir ../attacker/data
```
Then stop the server (Ctrl+C in Terminal 1) and confirm the port is free.

### 3. Attack A — defended (top-3 + noise) budget sweep

Terminal 1:
```
cd victim
python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt --top-k 3 --noise-std 0.02
```
Terminal 2:
```
cd attacker
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 1000  --data-dir ./data --out ../results/campaigns/d1000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 5000  --data-dir ./data --out ../results/campaigns/d5000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 10000 --data-dir ./data --out ../results/campaigns/d10000.npz
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 30000 --data-dir ./data --out ../results/campaigns/d30000.npz
```
Attack B against the defended server:
```
cd victim
python equation_solve.py --mode network --host 127.0.0.1 --port 9009 --data-dir ../attacker/data
```
Stop the server, confirm the port is free.

### 4. Attack A — label-only extreme

```
cd victim
python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt --label-only
```
```
cd attacker
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 10000 --data-dir ./data --out ../results/campaigns/labelonly10000.npz
```
Stop the server, confirm the port is free.

### 5. Rate-limit demonstration

This one is *supposed* to stop the client partway through — that's the
defence working. With the fixed `attack_client.py`, it now exits 0 and saves
whatever it collected before the quota (see `attacker/attack_client.py`'s
`RateLimitedError` handling).
```
cd victim
python victim_server.py --host 127.0.0.1 --port 9009 --model victim.pt --rate-limit 50 --rate-window 60
```
```
cd attacker
python attack_client.py --host 127.0.0.1 --port 9009 --pool mnist --queries 200 --data-dir ./data --out ../results/campaigns/ratelimit_fixed.npz
```
Expect: `quota exceeded after 50 queries ... stopping, keeping what was
already collected`, exit code 0, 50 pairs saved. Stop the server.

### 6. Attack B — offline diagnostics (no server needed)

```
cd victim
python equation_solve.py --mode offline --target linear
python equation_solve.py --mode offline --target cnn --checkpoint victim.pt --data-dir ../attacker/data
```

### 7. Train a knockoff for every campaign above

```
cd attacker
python knockoff_train.py --transfer ../results/campaigns/b1000.npz         --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_b1000.pt
python knockoff_train.py --transfer ../results/campaigns/b5000.npz         --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_b5000.pt
python knockoff_train.py --transfer ../results/campaigns/b10000.npz        --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_b10000.pt
python knockoff_train.py --transfer ../results/campaigns/b30000.npz        --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_b30000.pt
python knockoff_train.py --transfer ../results/campaigns/ood10000.npz      --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_ood10000.pt
python knockoff_train.py --transfer ../results/campaigns/d1000.npz         --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_d1000.pt
python knockoff_train.py --transfer ../results/campaigns/d5000.npz         --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_d5000.pt
python knockoff_train.py --transfer ../results/campaigns/d10000.npz        --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_d10000.pt
python knockoff_train.py --transfer ../results/campaigns/d30000.npz        --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_d30000.pt
python knockoff_train.py --transfer ../results/campaigns/labelonly10000.npz --epochs 20 --T 2.0 --data-dir ./data --out ../results/knockoffs/k_labelonly10000.pt
```

### 8. Evaluate fidelity, accuracy, and adversarial transfer for every knockoff

Uses `attacker/adv_transfer_eval.py` (Section 2 of the design/final report
explains why `--split train` is required, not `test`: the query pool for
`--pool mnist` above **is** the MNIST test split, so evaluating fidelity on
the test split would score the knockoff on images it already saw labelled).
```
cd attacker
mkdir -p ../results/eval
for f in ../results/knockoffs/*.pt; do
  name=$(basename "$f" .pt)
  python adv_transfer_eval.py --knockoff "$f" --victim ../victim/victim.pt \
      --data-dir ./data --split train --n-samples 3000 --eps 0.05 0.1 0.2 \
      --out "../results/eval/${name}.json"
done
```
(The `for` loop above is bash; on Windows, run the same command once per
file in `results/knockoffs/`.)

## Notes

- Seed fixed to `1337` everywhere (report requirement).
- `attacker/data/`, `*.pt`, `*.npz`, and `results/*.log` are gitignored —
  re-run the commands above to regenerate them locally. `results/eval/*.json`
  (from `adv_transfer_eval.py --out`) are **not** gitignored, so `git add`
  them explicitly if you want the numbers committed.
- Metrics to report per the design report: accuracy, fidelity (label
  agreement with the victim), and adversarial-example transfer, each with
  defences on vs. off.
