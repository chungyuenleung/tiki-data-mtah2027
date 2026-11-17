"""Frame-level ball possession from the automatic tracking (ATD) and the Smart Tagging.

    python solution/possession.py                     # writes evaluation/results/<name>.csv
    python solution/possession.py --name my_team      # pick the output name
    python solution/possession.py --no-cache          # re-read the raw tracking

numpy + the standard library only. No training, no labels, no constants tied to frames of the
match: every decision below comes from the tracking geometry, the Smart Tagging and the published
half boundaries in games.py.

The method, in four steps
-------------------------
1. FEATURES. One streaming pass over the raw ATD gives, per frame: the ball position, the distance
   from the ball to the nearest home player and to the nearest away player, the ball speed, and
   the median speed of the tracked players.

2. DEAD BALL. A frame is "dead-compatible" when the ball is unseen, on/over a boundary line, or
   lying still. A stretch of dead-compatible frames becomes D only when it is long enough AND
   contains positive proof of a stoppage: ball out of the pitch, ball slow on the line, a ball
   that has not moved for 5 s, or a ball unseen for 6 s or more while the players are not
   busy (median player speed below the match's 60th percentile). A short loss of the ball is
   never proof on its own -- that is the baseline's mistake, and the ball is missing on 41% of
   graded frames.
   Each proven stoppage is then stretched to the kicks either side of it: back to just after
   the ball was last moving fast, forward to the first frame it moves fast again.
   Smart Tagging then sharpens the picture: a tagged set piece ends a dead stretch at its restart,
   a tagged goal is dead until the kick-off that follows, and each half is dead until its kick-off.

3. POSSESSION. Inside every live stretch, a two-state Viterbi decoder (H / A) reads "touch
   evidence": the frames where one team clearly has a player on the ball. Changing team costs
   SWITCH_COST evidence frames, so a deflection, a duel or a one-touch interception that goes
   straight back does not flip possession, while a real turnover flips at the exact frame the new
   team first touched the ball (the decoder looks ahead, so the change is back-dated, not late).
   Frames with no evidence -- ball in flight, ball unseen -- stay with the last team, which is the
   competition's own rule for loose balls.

4. CLEAN-UP. Slivers of live play between two stoppages, and live stretches with no evidence at
   all, are folded into the stoppage. Ungraded frames are X.
"""

import argparse
import difflib
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import games  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Parameters. All in physical units, all chosen from how football is played -- none was fitted to
# the public ground truth (RULES.md forbids it, and EVALUATION.md shows it would pick wrongly).
# ----------------------------------------------------------------------------------------------
FPS = 25

# Touch evidence
TOUCH_RADIUS_M = 2.0        # a player this close to the ball can be playing it
TOUCH_MARGIN_M = 0.5        # ...and must be this much closer than the nearest opponent
TOUCH_MAX_BALL_SPEED = 9.0  # m/s; a ball flying past a player faster than this is not a touch

# Possession decoder
SWITCH_COST = 12            # evidence frames. A spell of the other team with fewer than
#                             2 x SWITCH_COST evidence frames (~1 s of observed touches, a couple
#                             of seconds of real time given how often the ball is unseen) is not
#                             "controlled" and does not change possession

# Dead ball
SPEED_HALF_WINDOW = 5       # frames either side used to measure ball speed
EDGE_M = 1.0                # ball within this of a boundary line counts as "on the line"
OUT_M = 0.5                 # ball this far beyond a line is out of the pitch
SLOW_SPEED = 2.0            # m/s; carried / rolled back, not played
STILL_RADIUS_M = 1.0        # a ball whose sightings all stay inside this circle is not moving
STILL_MIN_FRAMES = 125      # ...for 5 s = placed for a set piece (a defender waiting on the ball
#                             in open play rarely stands on it that long)
STILL_MAX_GAP = 75          # sightings of a placed ball may be up to 3 s apart
STILL_MIN_SEEN = 15         # and there must be at least this many of them
LONG_UNSEEN_FRAMES = 150    # ball unseen for 6 s: flight and occlusion do not last that long,
#                             a ball that has left the pitch does
ACTIVITY_SMOOTH = 50        # player activity = median player speed, averaged over +/- 2 s
ACTIVITY_MIN_SPREAD = 0.2   # m/s between the 10th and 90th percentile, or the signal is unusable
ACTIVITY_VETO = 0.60        # ...as a percentile of the match's own graded frames. An unseen-ball
#                             stretch during which the players are busier than this is the tracker
#                             losing a live ball, not a stoppage. Validated on two other matches
#                             (Metrica public sample games 1 and 2): it discards under 6% of real
#                             stoppages and about 65% of 8 s windows of live play.
BRIDGE_FRAMES = 10          # live-looking blips this short do not break a dead stretch
DEAD_MIN_FRAMES = 100       # 4 s; shorter stretches are not stoppages
DEAD_MIN_ANCHOR = 25        # 1 s of positive proof inside the stretch
DEAD_PAD_FRAMES = 75        # unseen frames kept either side of the proof (3 s)

# Stoppage edges. On in the submitted configuration; `--no-extend` turns it off. On eight matches
# of automatic broadcast tracking (SkillCorner open data) it adds 2.0-3.5 points of dead ball and
# improves spellF1 in seven of them.
EXTEND_STOPPAGES = True
EDGE_FAST_FRAMES = 3        # this many consecutive frames above RESTART_BALL_SPEED = ball played
EXTEND_FORWARD_MAX = 20 * FPS   # look this far after a proven stoppage for the restart...
EXTEND_BACKWARD_MAX = 8 * FPS   # ...and this far before it for the kick that ended play

# Smart Tagging
RESTART_BALL_SPEED = 4.0    # m/s; first frame this fast inside a set-piece window = the restart
RESTART_SNAP_FRAMES = 125   # a dead stretch ending within 5 s of a tagged restart is snapped to it
RESTART_MAX_BACK = {        # how far back a tagged restart may extend a stoppage, in frames
    "throw-in": 30 * FPS, "goal kick": 60 * FPS, "corner kick": 75 * FPS, "free kick": 90 * FPS}
RESTART_MAX_BACK_DEFAULT = 60 * FPS
RESTART_MIN_FRAMES = 50     # a tag-derived stoppage shorter than 2 s is ignored
GOAL_MAX_GAP_FRAMES = 240 * FPS   # a kick-off belongs to a goal at most 4 min earlier

# Clean-up
MIN_LIVE_FRAMES = 25        # live play shorter than 1 s between two stoppages is not live play

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
NONE = -1

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
CACHE_NAME = "_features_cache_v2.npz"   # written next to the data, never inside the repository


# ----------------------------------------------------------------------------------------------
# 1. Features
# ----------------------------------------------------------------------------------------------

def read_meta(meta_path):
    """(team_of_slot, (home_name, away_name)). First team in the file is home (FIFA/EPTS)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    team_of_slot = np.array([H if team_id == home_id else A for team_id in slots], dtype=np.int8)
    names = tuple((team.findtext("Name") or "").strip() for team in teams)
    return team_of_slot, names


def read_tracking(raw_path, team_of_slot, n_frames, pitch):
    """Per-frame ball position (m), distance (m) to the nearest home / away player, and the
    median speed (m/s) of the tracked players.

    Raw line: "<frame_id>:<player>;<player>;...:<ball>", each group "x,y,speed" in 0-1 pitch
    units, "NaN" where there is no observation. frame_id counts from 1, submissions from 0.
    """
    length, width = pitch
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    nearest = np.full((n_frames, 2), np.inf, dtype=np.float64)   # [:, H], [:, A]
    pace = np.full(n_frames, np.nan, dtype=np.float64)
    n_slots = len(team_of_slot)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue

            ball_fields = parts[2].split(",")
            has_ball = len(ball_fields) >= 2 and ball_fields[0][:1] not in ("N", "n", "")
            if has_ball:
                ball_x = float(ball_fields[0]) * length
                ball_y = float(ball_fields[1]) * width
                has_ball = ball_x == ball_x and ball_y == ball_y
            if has_ball:
                ball[frame, 0], ball[frame, 1] = ball_x, ball_y

            best = [np.inf, np.inf]
            speeds = []
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or group[:1] in ("N", "n", ""):
                    continue
                fields = group.split(",")
                if len(fields) < 2:
                    continue
                if len(fields) >= 3 and fields[2][:1] not in ("N", "n", ""):
                    speeds.append(float(fields[2]))
                if has_ball:
                    dx = float(fields[0]) * length - ball_x
                    dy = float(fields[1]) * width - ball_y
                    distance = dx * dx + dy * dy
                    team = team_of_slot[slot]
                    if distance < best[team]:
                        best[team] = distance
            if has_ball:
                nearest[frame, H] = best[H] ** 0.5
                nearest[frame, A] = best[A] ** 0.5
            if speeds:
                speeds.sort()
                middle = len(speeds) // 2
                pace[frame] = (speeds[middle] if len(speeds) % 2
                               else 0.5 * (speeds[middle - 1] + speeds[middle]))
    return ball, nearest, pace


def load_features(game=GAME, use_cache=True):
    """ball (n, 2) in metres with nan where unseen, nearest (n, 2) in metres, pace (n,) in m/s,
    team names."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    team_of_slot, names = read_meta(meta_path)

    cache_path = os.path.join(os.path.dirname(os.path.dirname(raw_path)), CACHE_NAME)
    stamp = "{}:{}".format(os.path.getsize(raw_path), int(os.path.getmtime(raw_path)))
    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["stamp"]) == stamp:
            print("features: loaded from cache")
            return cached["ball"], cached["nearest"], cached["pace"], names

    print("features: reading {} (a minute or two) ...".format(os.path.basename(raw_path)))
    ball, nearest, pace = read_tracking(raw_path, team_of_slot, n_frames, GAMES[game]["pitch"])
    if use_cache:
        try:
            np.savez(cache_path, ball=ball, nearest=nearest, pace=pace, stamp=np.array(stamp))
        except OSError:
            pass   # a read-only data folder is not a reason to fail
    return ball, nearest, pace, names


def ball_speed(ball, half_window=SPEED_HALF_WINDOW):
    """Ball speed in m/s from a centred difference; nan where either end is unseen."""
    n = len(ball)
    speed = np.full(n, np.nan)
    k = half_window
    delta = ball[2 * k:] - ball[:-2 * k]
    speed[k:n - k] = np.hypot(delta[:, 0], delta[:, 1]) / (2.0 * k / FPS)
    return speed


def activity_percentile(pace, blocks, half_window=ACTIVITY_SMOOTH):
    """How busy the players are at each frame, as a 0-1 percentile of this match's graded frames.

    pace is the median speed of the tracked players. It is averaged over +/- half_window frames
    and ranked against the graded frames of the same match, so the result does not depend on how
    a particular tracking system scales its speeds. nan where no player speed is available.
    """
    n = len(pace)
    known = ~np.isnan(pace)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, pace, 0.0))))
    count = np.concatenate(([0], np.cumsum(known)))
    index = np.arange(n)
    low = np.maximum(0, index - half_window)
    high = np.minimum(n, index + half_window + 1)
    seen = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = (total[high] - total[low]) / seen
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    reference = np.sort(smooth[graded & (seen > 0)])
    out = np.full(n, np.nan)
    if len(reference) and reference[int(0.9 * (len(reference) - 1))] - reference[
            int(0.1 * (len(reference) - 1))] < ACTIVITY_MIN_SPREAD:
        return out                           # speeds carry no information: no veto anywhere
    if len(reference):
        ok = seen > 0
        out[ok] = np.searchsorted(reference, smooth[ok], side="right") / float(len(reference))
    return out


# ----------------------------------------------------------------------------------------------
# Small run-length helpers
# ----------------------------------------------------------------------------------------------

def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def bridge(mask, max_gap):
    """Copy of mask with False gaps of at most max_gap frames (between two True runs) filled."""
    out = mask.copy()
    previous_stop = None
    for start, stop in runs_of(mask):
        if previous_stop is not None and start - previous_stop <= max_gap:
            out[previous_stop:start] = True
        previous_stop = stop
    return out


# ----------------------------------------------------------------------------------------------
# 2. Dead ball
# ----------------------------------------------------------------------------------------------

def still_mask(ball):
    """Frames where the ball is lying still, robust to the tracker blinking on and off.

    Greedy clustering of consecutive sightings: a cluster grows while the next sighting is within
    STILL_RADIUS_M of the cluster's mean position and no more than STILL_MAX_GAP frames later. A
    cluster that lasts STILL_MIN_FRAMES is a placed ball, and the unseen frames inside it are
    part of the same stoppage.
    """
    still = np.zeros(len(ball), dtype=bool)
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    i, total = 0, len(seen)
    while i < total:
        start = last = int(seen[i])
        sum_x, sum_y, count = ball[start, 0], ball[start, 1], 1
        j = i + 1
        while j < total:
            frame = int(seen[j])
            if frame - last > STILL_MAX_GAP:
                break
            dx = ball[frame, 0] - sum_x / count
            dy = ball[frame, 1] - sum_y / count
            if dx * dx + dy * dy > STILL_RADIUS_M * STILL_RADIUS_M:
                break
            sum_x += ball[frame, 0]
            sum_y += ball[frame, 1]
            count += 1
            last = frame
            j += 1
        if last - start + 1 >= STILL_MIN_FRAMES and count >= STILL_MIN_SEEN:
            still[start:last + 1] = True
        i = j
    return still


def dead_signals(ball, speed, pitch, activity=None):
    """(deadish, anchor): frames compatible with a stoppage, and frames that prove one.

    activity, if given, is the player-activity percentile; it vetoes "ball unseen for a long
    time" as proof when the players were clearly still playing.
    """
    length, width = pitch
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        to_line = np.minimum(np.minimum(ball[:, 0], length - ball[:, 0]),
                             np.minimum(ball[:, 1], width - ball[:, 1]))
        out = seen & (to_line < -OUT_M)
        edge = seen & (to_line < EDGE_M)
        slow = seen & (speed < SLOW_SPEED)

    still = still_mask(ball)

    long_unseen = np.zeros(len(ball), dtype=bool)
    for start, stop in runs_of(~seen):
        if stop - start < LONG_UNSEEN_FRAMES:
            continue
        if activity is not None:
            window = activity[start:stop]
            window = window[~np.isnan(window)]
            if len(window) and float(np.median(window)) > ACTIVITY_VETO:
                continue                     # players busy throughout: a lost ball, not a dead one
        long_unseen[start:stop] = True

    deadish = ~seen | edge | still
    anchor = out | (edge & slow) | still | long_unseen
    return deadish, anchor


def detect_dead(deadish, anchor, blocks):
    """Boolean D mask from the tracking alone, inside the graded blocks."""
    dead = np.zeros(len(deadish), dtype=bool)
    for _, first, last in blocks:
        window = slice(first, last + 1)
        candidate = bridge(deadish[window], BRIDGE_FRAMES)
        for start, stop in runs_of(candidate):
            start, stop = start + first, stop + first
            proof = np.flatnonzero(anchor[start:stop])
            if stop - start < DEAD_MIN_FRAMES or len(proof) < DEAD_MIN_ANCHOR:
                continue
            begin = max(start, start + int(proof[0]) - DEAD_PAD_FRAMES)
            end = min(stop, start + int(proof[-1]) + 1 + DEAD_PAD_FRAMES)
            if end - begin >= DEAD_MIN_FRAMES:
                dead[begin:end] = True
    return dead


def extend_dead(dead, speed, blocks):
    """Stretch each proven stoppage out to the moments the ball was actually played.

    The proof of a stoppage (ball out, still, or long unseen) usually covers only its middle. The
    ball is dead from the kick that sent it out until the kick that restarts play, and in between
    it may be visible: rolling to a stop, carried to the touchline, placed for a free kick. So
    each stoppage is extended backwards to just after the last frame the ball was moving fast,
    and forwards to the first frame it moves fast again. "Fast" is RESTART_BALL_SPEED for
    EDGE_FAST_FRAMES frames in a row. If no such moment is found within the limits, that side is
    left alone.
    """
    with np.errstate(invalid="ignore"):
        quick = speed > RESTART_BALL_SPEED
    fast = np.zeros(len(dead), dtype=bool)
    for start, stop in runs_of(quick):
        if stop - start >= EDGE_FAST_FRAMES:
            fast[start:stop] = True

    out = dead.copy()
    for _, first, last in blocks:
        for start, stop in runs_of(dead[first:last + 1]):
            start, stop = start + first, stop + first
            limit = min(last + 1, stop + EXTEND_FORWARD_MAX)
            ahead = np.flatnonzero(fast[stop:limit])
            if len(ahead):
                out[stop:stop + int(ahead[0])] = True
            limit = max(first, start - EXTEND_BACKWARD_MAX)
            behind = np.flatnonzero(fast[limit:start])
            if len(behind):
                out[limit + int(behind[-1]) + 1:start] = True
    return out


# ----------------------------------------------------------------------------------------------
# Smart Tagging
# ----------------------------------------------------------------------------------------------

def read_tagging(path):
    """[{code, type, team, start, end}] with lower-cased strings and video-frame intervals."""
    events = []
    for instance in ET.parse(path).iter("instance"):
        tags = {(label.findtext("group") or "").strip().lower():
                (label.findtext("text") or "").strip()
                for label in instance.findall("label")}
        events.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "type": tags.get("type", "").lower(),
            "team": tags.get("team", ""),
            "start": int(round(float(instance.findtext("start")) * FPS)),
            "end": int(round(float(instance.findtext("end")) * FPS)),
        })
    events.sort(key=lambda event: (event["start"], event["end"]))
    return events


def side_of(team_tag, names):
    """H or A for a Team tag, NONE when it does not name a team (e.g. 'N/A')."""
    tag = team_tag.strip().lower()
    if not tag or tag in ("n/a", "na", "none"):
        return NONE
    scores = [difflib.SequenceMatcher(None, tag, name.lower()).ratio() for name in names]
    for index, name in enumerate(names):
        if tag in name.lower() or name.lower() in tag:
            scores[index] += 1.0
    if max(scores) < 0.5 or scores[0] == scores[1]:
        return NONE
    return H if scores[0] > scores[1] else A


def restart_frame(event, speed, n_frames):
    """Best estimate of the frame the ball is put back in play inside a set-piece window."""
    start = max(0, event["start"])
    stop = min(n_frames, max(event["end"], start + 1))
    with np.errstate(invalid="ignore"):
        fast = np.flatnonzero(speed[start:stop] > RESTART_BALL_SPEED)
    if len(fast):
        return start + int(fast[0])
    return (start + stop) // 2


def apply_tagging(dead, deadish, events, speed, blocks, names):
    """Refine the D mask with set pieces, goals and kick-offs. Returns (dead, restarts).

    restarts maps a restart frame to the side taking it (H / A), where the tag names one.
    """
    n = len(dead)
    dead = dead.copy()
    restarts = {}

    def block_of(frame):
        for _, first, last in blocks:
            if first <= frame <= last:
                return first, last
        return None

    set_pieces = [event for event in events if event["code"] == "set pieces"]
    goals = [event for event in events if event["code"] == "goals"]

    for event in set_pieces:
        frame = restart_frame(event, speed, n)
        block = block_of(frame)
        if block is None:
            # A kick-off window can start just before the graded block; clamp into it.
            for _, first, last in blocks:
                if event["start"] <= first <= event["end"]:
                    frame, block = first, (first, last)
            if block is None:
                continue
        first, last = block
        side = side_of(event["team"], names)
        if side != NONE:
            restarts[frame] = side

        kind = event["type"]
        if kind == "kick off (start)":
            dead[first:frame] = True
            continue
        if kind == "kick off (after goal)":
            before = [goal for goal in goals
                      if goal["start"] < frame and frame - goal["start"] <= GOAL_MAX_GAP_FRAMES]
            if before:
                goal = before[-1]
                begin = max(first, (goal["start"] + goal["end"]) // 2)
                dead[begin:frame] = True
                continue

        # Any other restart: a stoppage that ends near it ends AT it.
        lo = max(first, frame - RESTART_SNAP_FRAMES)
        if dead[frame]:
            # the stoppage runs past the restart: cut it back, if it ends soon after
            stop = frame
            while stop <= last and dead[stop]:
                stop += 1
            if stop - frame <= RESTART_SNAP_FRAMES:
                dead[frame:stop] = False
            continue
        if dead[lo:frame].any():
            # the stoppage ends short of the restart: extend it over dead-compatible frames
            stop = lo + int(np.flatnonzero(dead[lo:frame])[-1]) + 1
            if stop == frame or deadish[stop:frame].mean() >= 0.5:
                dead[stop:frame] = True
            continue

        # No stoppage found by the tracking: walk back from the restart over dead-compatible
        # frames, as far as this kind of set piece plausibly lasts.
        limit = max(first, frame - RESTART_MAX_BACK.get(kind, RESTART_MAX_BACK_DEFAULT))
        compatible = bridge(deadish[limit:frame], BRIDGE_FRAMES)
        live = np.flatnonzero(~compatible)
        begin = limit + (int(live[-1]) + 1 if len(live) else 0)
        if frame - begin >= RESTART_MIN_FRAMES:
            dead[begin:frame] = True

    return dead, restarts


# ----------------------------------------------------------------------------------------------
# 3. Possession
# ----------------------------------------------------------------------------------------------

def touch_evidence(nearest, speed):
    """Per frame: H or A where one team clearly has a player on the ball, NONE otherwise."""
    home, away = nearest[:, H], nearest[:, A]
    with np.errstate(invalid="ignore"):
        playable = ~(speed > TOUCH_MAX_BALL_SPEED)       # unknown speed is allowed
    evidence = np.full(len(nearest), NONE, dtype=np.int8)
    evidence[playable & (home < TOUCH_RADIUS_M) & (home + TOUCH_MARGIN_M < away)] = H
    evidence[playable & (away < TOUCH_RADIUS_M) & (away + TOUCH_MARGIN_M < home)] = A
    return evidence


def decode(evidence, first_side=NONE, switch_cost=SWITCH_COST):
    """Most likely H/A sequence for one live stretch (Viterbi, two states).

    Score = number of evidence frames agreeing with the path, minus switch_cost per change of
    team. Frames without evidence score nothing either way, so they inherit their neighbours.
    Ties go to staying with the current team.
    """
    n = len(evidence)
    gain = np.zeros((n, 2))
    gain[evidence == H, H] = 1.0
    gain[evidence == A, A] = 1.0

    score = gain[0].copy()
    if first_side != NONE:
        score[first_side] += switch_cost       # a tagged restart names its taker
    came_from = np.zeros((n, 2), dtype=np.int8)
    came_from[0] = (H, A)
    for t in range(1, n):
        stay_h, swap_h = score[H], score[A] - switch_cost
        stay_a, swap_a = score[A], score[H] - switch_cost
        if swap_h > stay_h:
            new_h, came_from[t, H] = swap_h, A
        else:
            new_h, came_from[t, H] = stay_h, H
        if swap_a > stay_a:
            new_a, came_from[t, A] = swap_a, H
        else:
            new_a, came_from[t, A] = stay_a, A
        score[H] = new_h + gain[t, H]
        score[A] = new_a + gain[t, A]

    path = np.empty(n, dtype=np.int8)
    path[-1] = H if score[H] >= score[A] else A
    for t in range(n - 1, 0, -1):
        path[t - 1] = came_from[t, path[t]]
    return path


# ----------------------------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------------------------

def infer(ball, nearest, pace, blocks, pitch, events=None, names=("", ""), verbose=True):
    """Labels (codes) for every frame from the per-frame features. Pure function of its inputs."""
    n = len(ball)
    speed = ball_speed(ball)
    activity = activity_percentile(pace, blocks) if pace is not None else None
    deadish, anchor = dead_signals(ball, speed, pitch, activity)
    dead = detect_dead(deadish, anchor, blocks)
    proven = int(dead.sum())
    if EXTEND_STOPPAGES:
        dead = extend_dead(dead, speed, blocks)
    extended = int(dead.sum()) - proven

    restarts = {}
    if events is not None:
        dead, restarts = apply_tagging(dead, deadish, events, speed, blocks, names)

    evidence = touch_evidence(nearest, speed)
    labels = np.full(n, X, dtype=np.int8)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    total = float(graded.sum())
    seen = ~np.isnan(ball[:, 0])
    if verbose:
        print("diagnostics (graded frames): ball seen {:.1f}%, touch evidence {:.1f}%, "
              "stoppage proof {:.1f}%".format(
                  100.0 * (seen & graded).sum() / total,
                  100.0 * ((evidence != NONE) & graded).sum() / total,
                  100.0 * (anchor & graded).sum() / total))
        if pace is not None and (~np.isnan(pace[graded])).any():
            long_runs = [(a, b) for a, b in runs_of(~seen & graded)
                         if b - a >= LONG_UNSEEN_FRAMES]
            vetoed = sum(1 for a, b in long_runs if not anchor[a:b].any())
            print("player pace: median {:.2f} m/s; long unseen-ball stretches {}, "
                  "of which treated as live play {}".format(
                      float(np.nanmedian(pace[graded])), len(long_runs), vetoed))
        print("stoppage edges: extension {} and adds {:.1f}% of graded frames".format(
            "on" if EXTEND_STOPPAGES else "off", 100.0 * extended / total))
        stoppages = [stop - start for start, stop in runs_of(dead & graded)]
        if stoppages:
            print("stoppages: {}   mean duration {:.1f} s".format(
                len(stoppages), float(np.mean(stoppages)) / FPS))

    for _, first, last in blocks:
        block_dead = dead[first:last + 1].copy()

        # Clean-up: slivers of live play between two stoppages are part of the stoppage.
        for start, stop in runs_of(~block_dead):
            inside = start > 0 and stop < len(block_dead)
            if inside and stop - start < MIN_LIVE_FRAMES:
                block_dead[start:stop] = True

        labels[first:last + 1] = D
        for start, stop in runs_of(~block_dead):
            start, stop = start + first, stop + first
            stretch = evidence[start:stop]
            if not (stretch != NONE).any():
                continue                    # nothing to go on: leave it as stoppage
            side = NONE
            for frame in range(start - 2, start + 3):   # a restart tagged at this boundary
                if frame in restarts:
                    side = restarts[frame]
            labels[start:stop] = decode(stretch, side)
    return labels


def run(game=GAME, use_cache=True):
    ball, nearest, pace, names = load_features(game, use_cache)
    print("teams: H = {}, A = {}".format(*names))

    events = None
    tagging_path = games.asset_paths(game).get("smart_tagging")
    if tagging_path and os.path.isfile(tagging_path):
        events = read_tagging(tagging_path)
        named = sum(1 for event in events
                    if event["code"] == "set pieces" and side_of(event["team"], names) != NONE)
        print("smart tagging: {} instances, {} set pieces with a named side".format(
            len(events), named))
    else:
        print("smart tagging: file not found, continuing on tracking alone")

    return infer(ball, nearest, pace, half_video_blocks(game), GAMES[game]["pitch"],
                 events, names)


def summarise(labels, blocks):
    """Counts a football analyst can sanity-check without any ground truth."""
    graded = np.zeros(len(labels), dtype=bool)
    spells, alive = 0, 0
    for _, first, last in blocks:
        graded[first:last + 1] = True
        window = labels[first:last + 1]
        change = np.flatnonzero(window[1:] != window[:-1]) + 1
        starts = np.concatenate(([0], change))
        spells += int(np.sum(window[starts] != D))
        alive += int(np.sum(window != D))
    total = int(graded.sum())
    print("\ngraded frames: {}".format(total))
    for code in (H, A, D):
        count = int(np.sum(labels[graded] == code))
        print("  {}  {:>7}  {:5.1f}%".format(LETTERS[code], count, 100.0 * count / total))
    if spells:
        print("possessions: {}   mean duration {:.1f} s".format(spells, alive / float(FPS) / spells))
    describe_possessions(labels, blocks)


BUCKETS = ((0, 2), (2, 4), (4, 8), (8, 15), (15, 30), (30, 10 ** 6))   # seconds


def describe_possessions(labels, blocks):
    """Print-only: how long possessions are, and how quick recoveries were labelled.

    A "quick recovery" is a short spell of one team sandwiched directly between two spells of
    the other, with no stoppage in between -- e.g. Barcelona lose the ball, Athletic control it
    briefly, Barcelona win it back. Nothing here feeds back into the labels.
    """
    names = {H: "H", A: "A"}
    counts = {H: [0] * len(BUCKETS), A: [0] * len(BUCKETS)}
    sandwiches = {H: [0, 0, 0], A: [0, 0, 0]}     # middle team -> [under 2 s, 2-4 s, 4-8 s]
    direct, via_stoppage = 0, 0
    for _, first, last in blocks:
        window = labels[first:last + 1]
        change = np.flatnonzero(window[1:] != window[:-1]) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [len(window)]))
        codes = window[starts]
        seconds = (stops - starts) / float(FPS)
        for index, code in enumerate(codes):
            if code not in (H, A):
                continue
            for bucket, (low, high) in enumerate(BUCKETS):
                if low <= seconds[index] < high:
                    counts[code][bucket] += 1
            if index + 1 < len(codes):
                if codes[index + 1] in (H, A):
                    direct += 1
                elif index + 2 < len(codes) and codes[index + 2] in (H, A):
                    via_stoppage += 1
            if 0 < index < len(codes) - 1 and codes[index - 1] == codes[index + 1] \
                    and codes[index - 1] in (H, A) and codes[index - 1] != code:
                for slot, (low, high) in enumerate(((0, 2), (2, 4), (4, 8))):
                    if low <= seconds[index] < high:
                        sandwiches[code][slot] += 1

    print("\npossessions by length     <2s   2-4s   4-8s  8-15s 15-30s   30s+")
    for code in (H, A):
        print("  {}                    ".format(names[code])
              + "".join("{:>7}".format(value) for value in counts[code]))
    print("hand-overs: {} in open play, {} across a stoppage".format(direct, via_stoppage))
    print("quick recoveries (short spell between two spells of the other team, no stoppage):")
    print("  A between H-H:  under 2 s {:>3}   2-4 s {:>3}   4-8 s {:>3}".format(*sandwiches[A]))
    print("  H between A-A:  under 2 s {:>3}   2-4 s {:>3}   4-8 s {:>3}".format(*sandwiches[H]))


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    letters = LETTERS[labels]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(letters))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="submission", help="output file name, without .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="output folder")
    parser.add_argument("--no-cache", action="store_true", help="re-read the raw tracking")
    parser.add_argument("--no-extend", action="store_true",
                        help="do not extend stoppages out to the kicks either side")
    args = parser.parse_args()
    if args.no_extend:
        EXTEND_STOPPAGES = False

    labels = run(use_cache=not args.no_cache)
    summarise(labels, half_video_blocks(GAME))
    path = write_submission(os.path.join(args.out, "{}.csv".format(args.name)), labels)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("check it:  python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))
    print("score it:  python evaluation/run_evaluation.py")
