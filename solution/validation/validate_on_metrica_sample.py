"""Validation of the pipeline's thresholds on OTHER matches: Metrica Sports public sample data.

    git clone https://github.com/metrica-sports/sample-data
    set METRICA_SAMPLE_DIR=<path>\\sample-data\\data            (Windows)
    export METRICA_SAMPLE_DIR=<path>/sample-data/data           (macOS / Linux)
    python solution/validation/validate_on_metrica_sample.py

Nothing here touches the competition match. It uses Sample Game 1 and Sample Game 2 (anonymised
matches published by Metrica Sports with tracking and event data), which RULES.md allows:
"Hand-labelled data from other games is fine". The event data was produced by Metrica Sports; we
did no labelling ourselves.

Ground truth for those matches is built the way CHALLENGE.md describes the competition's own:
  - D where the corrected tracking has no ball (Metrica removes the ball when play is dead);
  - otherwise the owning team = the team of the latest on-ball event (set piece, pass, shot,
    recovery). A spell of the other team shorter than 2 s and with fewer than 2 passes is treated
    as "not controlled" and absorbed, following the competition's definition of controlled
    possession.

The sample tracking is cleaner than the competition's automatic tracking, so it is degraded
before use: 0.3 m position jitter, short ball drop-outs, and 0 / 20 / 40 losses of the ball of
6-12 s during live play. Each setting is scored with the competition's own spellF1 and macroF1,
averaged over 2 matches x 3 degradation levels.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "solution"))
import possession_metrics as pm  # noqa: E402
import possession as sol  # noqa: E402

warnings.simplefilter("ignore")

BASE = os.environ.get("METRICA_SAMPLE_DIR") or os.path.join(ROOT, "sample-data", "data")
PITCH = (105.0, 68.0)
H, A, D, X = 0, 1, 2, 3


# ------------------------------------------------------------------------------ loading

def load_tracking(game):
    sides = {}
    for side in ("Home", "Away"):
        path = os.path.join(BASE, "Sample_Game_{0}".format(game),
                            "Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(game, side))
        values = pd.read_csv(path, skiprows=2).values.astype(float)
        players = values[:, 3:-2]
        sides[side] = players.reshape(len(values), players.shape[1] // 2, 2) * PITCH
        ball = values[:, -2:] * PITCH
    return ball, sides["Home"], sides["Away"]


def load_events(game):
    return pd.read_csv(os.path.join(BASE, "Sample_Game_{0}".format(game),
                                    "Sample_Game_{0}_RawEventsData.csv".format(game)))


# ------------------------------------------------------------------------------ ground truth

def events_to_labels(events, n_frames):
    """H/A/D per frame from the event list (dead from ball out / foul / goal to the next event)."""
    changes = []
    for _, e in events.iterrows():
        team = H if e["Team"] == "Home" else A
        kind = e["Type"]
        sub = e["Subtype"] if isinstance(e["Subtype"], str) else ""
        start = int(e["Start Frame"]) - 1
        end = max(int(e["End Frame"]) - 1 if e["End Frame"] > 0 else start, start)
        if kind in ("SET PIECE", "PASS", "RECOVERY"):
            changes.append((start, team))
        elif kind == "SHOT":
            changes.append((start, team))
            if "GOAL" in sub or "OUT" in sub:
                changes.append((end, D))
        elif kind == "BALL LOST":
            changes.append((start, team))
            if "OFFSIDE" in sub or "END HALF" in sub or "HAND BALL" in sub:
                changes.append((end, D))
        elif kind == "BALL OUT":
            changes.append((start, team))
            changes.append((end, D))
        elif kind == "FAULT RECEIVED":
            changes.append((start, D))
    changes.sort(key=lambda change: change[0])
    labels = np.full(n_frames, X, dtype=np.int8)
    state, last = D, 0
    for frame, new in changes:
        frame = min(max(frame, 0), n_frames)
        labels[last:frame] = state
        state, last = new, max(last, frame)
    labels[last:] = state
    labels[:changes[0][0]] = X
    return labels


def absorb_uncontrolled(labels, events, min_frames=50, min_passes=2):
    out = labels.copy()
    passes = {H: [], A: []}
    for _, e in events.iterrows():
        if e["Type"] == "PASS":
            passes[H if e["Team"] == "Home" else A].append(int(e["Start Frame"]) - 1)
    passes = {team: np.array(frames) for team, frames in passes.items()}
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    previous = D
    for start, stop in zip(starts, stops):
        code = labels[start]
        if code in (H, A):
            if previous in (H, A) and previous != code:
                n_passes = int(((passes[code] >= start) & (passes[code] < stop)).sum())
                if stop - start < min_frames and n_passes < min_passes:
                    out[start:stop] = previous
                    continue
            previous = code
        else:
            previous = D
    return out


def ground_truth(game, ball):
    events = load_events(game)
    by_events = absorb_uncontrolled(events_to_labels(events, len(ball)), events)
    n = len(ball)
    team = by_events.astype(np.int16)
    team[(team == D) | (team == X)] = -1
    index = np.where(team >= 0, np.arange(n), -1)
    np.maximum.accumulate(index, out=index)
    forward = np.where(index >= 0, team[np.maximum(index, 0)], -1)
    following = np.where(team >= 0, np.arange(n), n)
    following = np.minimum.accumulate(following[::-1])[::-1]
    backward = np.where(following < n, team[np.minimum(following, n - 1)], -1)
    owner = np.where(forward >= 0, forward, backward)
    labels = np.where(~np.isnan(ball[:, 0]), owner, D).astype(np.int8)
    labels[by_events == X] = X
    return labels


# ------------------------------------------------------------------------------ degradation

def nearest_distance(ball, players):
    distance = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    distance = np.where(np.isnan(distance), np.inf, distance).min(axis=1)
    distance[np.isnan(ball[:, 0])] = np.inf
    return distance


def player_pace(players, k=5):
    speed = np.full(players.shape[:2], np.nan)
    delta = players[2 * k:] - players[:-2 * k]
    speed[k:-k] = np.hypot(delta[..., 0], delta[..., 1]) / (2.0 * k / sol.FPS)
    return np.nanmedian(speed, axis=1)


def reveal_ball_in_stoppages(ball0, truth, first, rng):
    """Copy of the ball track with the ball made visible during part of every stoppage.

    Corrected tracking removes the ball whenever play is dead; automatic tracking keeps seeing
    it. To mimic that, after each stoppage begins the ball rolls on from its last position for
    1-3 s, and for the last 3-10 s before the restart it is either lying on the restart spot or
    being carried to it at walking pace.
    """
    ball = ball0.copy()
    n = len(truth)
    seen = ~np.isnan(ball0[:, 0])
    for start, stop in sol.runs_of(truth == D):
        if start <= first or stop >= n - 1 or stop - start < 50:
            continue
        length = stop - start
        if seen[start - 1]:
            k = int(min(length // 4, rng.integers(25, 75)))
            before = max(start - 6, 0)
            velocity = (ball0[start - 1] - ball0[before]) / 5.0 if seen[before] else np.zeros(2)
            ball[start:start + k] = ball0[start - 1] + velocity * np.cumsum(
                0.9 ** np.arange(1, k + 1))[:, None]
        if seen[stop]:
            k = int(min(length // 2, rng.integers(75, 250)))
            spot = ball0[stop]
            if rng.random() < 0.5:
                ball[stop - k:stop] = spot
            else:
                weight = np.linspace(0, 1, k)[:, None]
                ball[stop - k:stop] = (spot + rng.normal(0, 6, 2)) * (1 - weight) + spot * weight
    return ball


def prepare(game, n_long, seed, reveal=False):
    ball0, home, away = load_tracking(game)
    truth = ground_truth(game, ball0)
    n = len(truth)
    first = int(np.flatnonzero(truth != X)[0])
    pace = player_pace(np.concatenate([home, away], axis=1))
    live = (truth == H) | (truth == A)

    rng = np.random.default_rng(seed)
    if reveal:
        ball0 = reveal_ball_in_stoppages(ball0, truth, first, rng)
    ball = ball0 + rng.normal(0, 0.3, ball0.shape)
    home = home + rng.normal(0, 0.3, home.shape)
    away = away + rng.normal(0, 0.3, away.shape)
    t = 0
    while t < n:                                   # short drop-outs
        t += int(rng.exponential(140)) + 1
        gap = int(rng.exponential(25)) + 1
        ball[t:t + gap] = np.nan
        t += gap
    placed = 0
    while placed < n_long:                         # long losses of a live ball
        start = int(rng.integers(first, n - 400))
        gap = int(rng.integers(150, 300))
        if live[start:start + gap].all():
            ball[start:start + gap] = np.nan
            placed += 1
    nearest = np.column_stack([nearest_distance(ball, home), nearest_distance(ball, away)])
    return ball, nearest, pace, [(1, first, n - 1)], truth


# ------------------------------------------------------------------------------ scoring

DEFAULTS = {name: getattr(sol, name) for name in (
    "ACTIVITY_VETO", "DEAD_PAD_FRAMES", "TOUCH_RADIUS_M", "SWITCH_COST", "LONG_UNSEEN_FRAMES",
    "EXTEND_STOPPAGES")}


def evaluate(datasets, **overrides):
    settings = dict(DEFAULTS)
    settings.update(overrides)
    for name, value in settings.items():
        setattr(sol, name, value)
    sol.decode.__defaults__ = (sol.NONE, settings["SWITCH_COST"])
    spell, macro = [], []
    for ball, nearest, pace, blocks, truth in datasets:
        predicted = sol.infer(ball, nearest, pace, blocks, PITCH, None, verbose=False)
        runs = pm.valid_runs(truth)
        filled = pm.fill_abstentions(predicted, runs)
        spell.append(pm.spell_scores(truth, filled, runs)["mean_f1"])
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(truth, predicted))))
    return float(np.mean(spell)), float(np.mean(macro))


if __name__ == "__main__":
    datasets = [prepare(game, n_long, 11 * game + n_long)
                for game in (1, 2) for n_long in (0, 20, 40)]
    tests = [
        ("reference settings", {}),
        ("activity veto off", {"ACTIVITY_VETO": 2.0}),
        ("activity veto 0.5", {"ACTIVITY_VETO": 0.5}),
        ("activity veto 0.7", {"ACTIVITY_VETO": 0.7}),
        ("switch cost 6", {"SWITCH_COST": 6}),
        ("switch cost 18", {"SWITCH_COST": 18}),
        ("switch cost 25", {"SWITCH_COST": 25}),
        ("unseen proof 4 s", {"LONG_UNSEEN_FRAMES": 100}),
        ("unseen proof 8 s", {"LONG_UNSEEN_FRAMES": 200}),
        ("touch radius 1.5 m", {"TOUCH_RADIUS_M": 1.5}),
        ("touch radius 2.5 m", {"TOUCH_RADIUS_M": 2.5}),
        ("stoppage padding 0", {"DEAD_PAD_FRAMES": 0}),
    ]
    print("A. thresholds, stoppage extension off, ball hidden in stoppages (as published)")
    print("{:<22} {:>8} {:>8}".format("setting", "spellF1", "macroF1"))
    for name, overrides in tests:
        overrides = dict(overrides, EXTEND_STOPPAGES=False)
        spell, macro = evaluate(datasets, **overrides)
        print("{:<22} {:>8.4f} {:>8.4f}".format(name, spell, macro), flush=True)

    print()
    print("B. stoppage extension, by whether the tracking shows the ball during stoppages")
    print("{:<34} {:<10} {:>8} {:>8} {:>8}".format("ball in stoppages", "extension", "spellF1",
                                                 "macroF1", "D share"))
    hidden = [prepare(game, n_long, 11 * game + n_long) for game in (1, 2) for n_long in (0, 20)]
    shown = [prepare(game, n_long, 5 * game + n_long, reveal=True)
             for game in (1, 2) for n_long in (0, 20)]
    for label, group in (("hidden (as published)", hidden), ("partly visible (simulated)", shown)):
        truth_share = np.mean([(d[4] == D).sum() / float((d[4] != X).sum()) for d in group])
        for extend in (False, True):
            spell, macro = evaluate(group, EXTEND_STOPPAGES=extend)
            share = np.mean([(sol.infer(d[0], d[1], d[2], d[3], PITCH, None, verbose=False)
                              == D).sum() / float((d[4] != X).sum()) for d in group])
            print("{:<34} {:<10} {:>8.4f} {:>8.4f} {:>7.1f}%".format(
                label, "on" if extend else "off", spell, macro, 100 * share), flush=True)
        print("{:<34} {:<10} {:>8} {:>8} {:>7.1f}%".format("", "truth", "", "", 100 * truth_share))
