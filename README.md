# tiki-data — More than a Hack 2027, Phase 1: Ball Possession

Team **tiki-data**'s entry for the frame-level ball-possession challenge
(FC Barcelona vs Athletic Club, organised by Metrica Sports, FC Barcelona and Mobile World
Capital Barcelona).

| | |
|---|---|
| The pipeline | [`solution/possession.py`](solution/possession.py) |
| Technical document | [`solution/tiki-data.md`](solution/tiki-data.md) |
| Validation on other matches | [`solution/validation/validate_on_metrica_sample.py`](solution/validation/validate_on_metrica_sample.py) |

## Reproduce the submission

Python 3.11 (developed on 3.11.9), numpy 2.4.6, pandas 3.1.1, pinned in `requirements.txt`.
The match data is **not** in this repository; it is released by the organisers to enrolled
participants.

```
pip install -r requirements.txt
set "MTAH_DATA_DIR=<folder containing tactical\>"        (Windows)
export MTAH_DATA_DIR=<folder containing tactical/>       (macOS / Linux)
python solution/possession.py --name tiki-data --no-cache
python evaluation/validate_submission.py evaluation/results/tiki-data.csv
```

The submission is written to `evaluation/results/tiki-data.csv`. Required data files, all under
`tactical/`: `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`,
`BAR-ATH_tactical_smart_tagging.xml`.

## What is ours and what is not

Everything under `solution/` and this README are ours. The rest (`evaluation/`, `starter/`,
`games.py`, the documentation files) is the organisers' starter repository,
<https://github.com/metrica-sports/more-than-a-hack-2027>, unchanged apart from pinned versions in
`requirements.txt`. Its original README is kept as [`README_STARTER.md`](README_STARTER.md). The
starter code is under the MIT licence (see `LICENSE`); the match data is licensed separately and
is not redistributed here.

Threshold validation uses the Metrica Sports public sample data,
<https://github.com/metrica-sports/sample-data>.
