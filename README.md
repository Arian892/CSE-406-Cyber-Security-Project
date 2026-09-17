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

## Notes

- Seed fixed to `1337` everywhere (report requirement).
- `attacker/data/`, `*.pt`, and `*.npz` are gitignored — re-run
  `train_victim.py` and `attack_client.py` to regenerate them locally.
- Metrics to report per the design report: accuracy, fidelity (label
  agreement with the victim), and adversarial-example transfer, each with
  defences on vs. off.
