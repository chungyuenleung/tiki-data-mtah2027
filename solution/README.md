# tiki-data — More than a Hack 2027, Phase 1

Our entry for the ball-possession challenge. This repository is the organisers' starter
repository (MIT licence, see `LICENSE`) plus our solution in `solution/`.

- `solution/possession.py` — the whole pipeline
- `solution/tiki-data.md` — the technical document
- `solution/validation/validate_on_metrica_sample.py` — threshold validation on two other
  matches (Metrica Sports public sample data)

## Reproduce the submission

Python 3.11 (developed on 3.11.9), numpy 2.4.6, pandas 3.1.1, pinned in `requirements.txt`.
The match data is not in this repository.

```
pip install -r requirements.txt
set "MTAH_DATA_DIR=<folder containing tactical\>"        (Windows)
export MTAH_DATA_DIR=<folder containing tactical/>       (macOS / Linux)
python solution/possession.py --name tiki-data --no-cache
python evaluation/validate_submission.py evaluation/results/tiki-data.csv
```

Output: `evaluation/results/tiki-data.csv`. Needs `BAR-ATH_tactical_atd.txt`,
`BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml` under `tactical/`.
