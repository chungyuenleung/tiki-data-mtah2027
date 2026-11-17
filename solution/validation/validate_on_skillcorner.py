"""Validation on OTHER matches with AUTOMATIC tracking: SkillCorner open data.

    git lfs install
    git clone https://github.com/SkillCorner/opendata
    set SKILLCORNER_DIR=<path>\\opendata\\data\\matches          (Windows)
    export SKILLCORNER_DIR=<path>/opendata/data/matches         (macOS / Linux)
    python solution/validation/validate_on_skillcorner.py

Nothing here touches the competition match. It uses eight matches of the 2024/25 Australian
A-League published by SkillCorner (https://github.com/SkillCorner/opendata): broadcast tracking at
10 fps and a "phases of play" file. SkillCorner produced both; we labelled nothing.

Why this set: it is automatic tracking, like the competition's. The ball is detected on about
92% of live frames and about 19% of dead-ball frames, 53-63% overall (59% on the competition
match), and it is often still visible in the first and last seconds of a stoppage. The Metrica
sample matches have corrected tracking, where the ball disappears exactly when play is dead.

Labels: SkillCorner defines a phase of play only while the ball is in play, and every phase has a
team in possession. So a frame inside a phase is H or A, and a frame in no phase is D.

The tracking is repeated onto a 25 fps grid so that the pipeline runs unchanged. A ball that is
extrapolated rather than detected is treated as unseen. There is no Smart Tagging equivalent, so
the pipeline runs on tracking alone.
"""
import csv
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import possession_metrics as pm  # noqa: E402
import possession as sol  # noqa: E402

warnings.simplefilter("ignore")

BASE = os.environ.get("SKILLCORNER_DIR") or os.path.join(ROOT, "opendata", "data", "matches")
MATCHES = (1886347, 1899585, 1925299, 1953632,      # used while developing
           1996435, 2006229, 2011166, 2017461)      # held out until the end
SOURCE_FPS = 10.0
H, A, D, X = 0, 1, 2, 3


def path(match_id, suffix):
    return os.path.join(BASE, str(match_id), "{}_{}".format(match_id, suffix))


def load(match_id):
    """(ball, nearest, pace, blocks, labels, pitch) on a 25 fps grid."""
    meta = json.load(open(path(match_id, "match.json")))
    home_id = meta["home_team"]["id"]
    team_of = {player["id"]: (H if player["team_id"] == home_id else A)
               for player in meta["players"]}
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    with open(path(match_id, "tracking_extrapolated.jsonl")) as handle:
        frames = [json.loads(line) for line in handle]

    n = len(frames)
    ids = sorted(team_of)
    column = {player_id: index for index, player_id in enumerate(ids)}
    team = np.array([team_of[player_id] for player_id in ids])
    players = np.full((n, len(ids), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan)
    period = np.zeros(n, dtype=np.int8)
    for index, frame in enumerate(frames):
        period[index] = frame["period"] or 0
        seen = frame["ball_data"]
        if seen["x"] is not None and seen["is_detected"]:
            ball[index] = (seen["x"] + length / 2, seen["y"] + width / 2)
        for player in frame["player_data"]:
            slot = column.get(player["player_id"])
            if slot is not None and player["x"] is not None:
                players[index, slot] = (player["x"] + length / 2, player["y"] + width / 2)

    labels = np.full(n, X, dtype=np.int8)
    labels[period > 0] = D
    with open(path(match_id, "phases_of_play.csv")) as handle:
        for row in csv.DictReader(handle):
            side = H if int(row["team_in_possession_id"]) == home_id else A
            labels[int(row["frame_start"]):int(row["frame_end"])] = side
    labels[period == 0] = X

    step = sol.FPS / SOURCE_FPS
    grid = np.minimum((np.arange(int(n * step)) / step).astype(int), n - 1)
    ball, players, labels = ball[grid], players[grid], labels[grid]

    def nearest_of(mask):
        distance = np.hypot(players[:, mask, 0] - ball[:, None, 0],
                            players[:, mask, 1] - ball[:, None, 1])
        distance = np.where(np.isnan(distance), np.inf, distance).min(axis=1)
        distance[np.isnan(ball[:, 0])] = np.inf
        return distance

    nearest = np.column_stack([nearest_of(team == H), nearest_of(team == A)])
    k = 5
    speed = np.full(players.shape[:2], np.nan)
    delta = players[2 * k:] - players[:-2 * k]
    speed[k:-k] = np.hypot(delta[..., 0], delta[..., 1]) / (2.0 * k / sol.FPS)
    pace = np.nanmedian(speed, axis=1)

    graded = np.diff(np.concatenate(([0], (labels != X).astype(int), [0])))
    blocks = [(index + 1, int(start), int(stop) - 1) for index, (start, stop) in enumerate(
        zip(np.flatnonzero(graded == 1), np.flatnonzero(graded == -1))) if stop - start > 5000]
    return ball, nearest, pace, blocks, labels, (length, width)


SETTINGS = ("ACTIVITY_VETO", "TOUCH_RADIUS_M", "SWITCH_COST", "LONG_UNSEEN_FRAMES",
            "EXTEND_STOPPAGES")
DEFAULTS = {name: getattr(sol, name) for name in SETTINGS}


def evaluate(datasets, **overrides):
    settings = dict(DEFAULTS)
    settings.update(overrides)
    for name, value in settings.items():
        setattr(sol, name, value)
    sol.decode.__defaults__ = (sol.NONE, settings["SWITCH_COST"])
    spell, macro, share = [], [], []
    for ball, nearest, pace, blocks, truth, pitch in datasets:
        predicted = sol.infer(ball, nearest, pace, blocks, pitch, None, verbose=False)
        runs = pm.valid_runs(truth)
        spell.append(pm.spell_scores(truth, pm.fill_abstentions(predicted, runs), runs)["mean_f1"])
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(truth, predicted))))
        share.append(100.0 * (predicted[truth != X] == D).mean())
    return spell, macro, share


if __name__ == "__main__":
    datasets = [load(match_id) for match_id in MATCHES]
    for match_id, (ball, _, _, _, truth, _) in zip(MATCHES, datasets):
        seen = ~np.isnan(ball[:, 0])
        live = (truth == H) | (truth == A)
        print("{}: dead ball {:.1f}%, ball detected {:.1f}% (live {:.1f}%, dead {:.1f}%)".format(
            match_id, 100.0 * (truth == D).sum() / (truth != X).sum(),
            100.0 * seen[truth != X].mean(), 100.0 * seen[live].mean(),
            100.0 * seen[truth == D].mean()))

    tests = [
        ("submitted settings", {}),
        ("no stoppage extension", {"EXTEND_STOPPAGES": False}),
        ("no activity veto", {"ACTIVITY_VETO": 2.0}),
        ("switch cost 6", {"SWITCH_COST": 6}),
        ("switch cost 18", {"SWITCH_COST": 18}),
        ("switch cost 25", {"SWITCH_COST": 25}),
        ("touch radius 1.0 m", {"TOUCH_RADIUS_M": 1.0}),
        ("touch radius 1.5 m", {"TOUCH_RADIUS_M": 1.5}),
        ("touch radius 3.0 m", {"TOUCH_RADIUS_M": 3.0}),
        ("unseen proof 4 s", {"LONG_UNSEEN_FRAMES": 100}),
        ("unseen proof 8 s", {"LONG_UNSEEN_FRAMES": 200}),
    ]
    print()
    print("{:<24} {:>8} {:>8} {:>10} {:>10}  {}".format(
        "setting", "spellF1", "macroF1", "first 4", "last 4", "better than submitted in"))
    base = None
    for name, overrides in tests:
        spell, macro, _ = evaluate(datasets, **overrides)
        if base is None:
            base = spell
        better = sum(1 for a, b in zip(spell, base) if a > b + 1e-9)
        print("{:<24} {:>8.4f} {:>8.4f} {:>10.4f} {:>10.4f}  {}".format(
            name, np.mean(spell), np.mean(macro), np.mean(spell[:4]), np.mean(spell[4:]),
            "-" if overrides == {} else "{}/8 matches".format(better)), flush=True)

    _, _, on = evaluate(datasets)
    _, _, off = evaluate(datasets, EXTEND_STOPPAGES=False)
    print("\nstoppage extension adds, in points of dead ball, per match: {}".format(
        " ".join("{:.1f}".format(a - b) for a, b in zip(on, off))))
