# tiki-data — Technical document

More than a Hack 2027, Phase 1: frame-level ball possession (FC Barcelona vs Athletic Club).

## 1. Approach

A rule-based, fully deterministic pipeline with no learned model. It reads the provided automatic
tracking (ATD) and Smart Tagging of the tactical recording and writes one label per video frame.
Everything is in one file, `solution/possession.py` (numpy + standard library).

We chose rules over a learned model for three reasons: the rules forbid labels on the provided
match; the organisers' definition of possession is itself a set of rules ("controlled" possession,
loose balls stay with the last team); and the ranking metric rewards getting whole possessions
right, which is a segmentation problem that a small state model handles directly. The thresholds
were then checked on two other matches (section 3).

The pipeline has four steps.

**1. Features.** One pass over the raw ATD gives, per frame: ball position in metres, distance
from the ball to the nearest home player and to the nearest away player, ball speed (centred
difference over ±5 frames), and the median speed of the tracked players. Track slots are used only
for team membership, never as identities.

**2. Dead ball (`D`).** A frame is *dead-compatible* if the ball is unseen, within 1 m of a
boundary line (or beyond it), or lying still. A stretch of dead-compatible frames is labelled `D`
only if it lasts at least 4 s **and** contains at least 1 s of positive proof of a stoppage:

- ball more than 0.5 m outside the pitch;
- ball on a boundary line and slower than 2 m/s;
- ball lying still (all sightings within a 1 m circle) for 5 s or more, tolerating gaps in detection;
- ball unseen for 6 s or more, **unless the players were busy throughout** (see below).

A short loss of the ball is never proof by itself: the ball is unseen on 41% of graded frames.
The stoppage is trimmed to 3 s either side of its proof.

*Player-activity veto.* The tracker can also lose a live ball for a long time. To tell that apart
from a stoppage we use the players: activity is the median player speed, averaged over ±2 s and
expressed as a percentile of the match's own graded frames (so it does not depend on how a
tracking system scales its speeds). If the median activity during an unseen stretch is above the
60th percentile, the stretch is not treated as proof of a stoppage. On the competition match this
reclassified 17 of 115 long unseen stretches as live play.

Smart Tagging then refines the stoppages:

- each half is `D` from its first graded frame until the tagged `Kick Off (Start)`;
- a tagged goal is `D` until the following `Kick Off (After Goal)`;
- a stoppage that ends within 5 s of any other tagged set piece is snapped to the restart;
- where tracking found no stoppage before a tagged set piece, one is created by walking back from
  the restart over dead-compatible frames (bounded per set-piece type).

The restart instant inside a tagged window is the first frame where the ball moves faster than
4 m/s, or the middle of the window if no such frame is seen.

**3. Possession (`H` / `A`).** Inside every live stretch a two-state Viterbi decoder reads *touch
evidence*: frames where one team has a player within 2 m of the ball, at least 0.5 m closer than
the nearest opponent, with the ball slower than 9 m/s. The path score is the number of evidence
frames that agree with it, minus a cost of 12 evidence frames per change of team. Consequences:

- a deflection, a duel or a one-touch interception that goes straight back does not change
  possession, because a spell of the other team needs about 24 evidence frames to pay for
  switching there and back;
- a real turnover is placed at the frame the new team first touched the ball, because the decoder
  sees the whole stretch and back-dates the change, so there is no smoothing lag;
- frames with no evidence (ball in flight or unseen) inherit the surrounding team, which is the
  competition's rule that a loose ball stays with the last team in possession.

Where a tagged restart names its team, that team is given a head start equal to one switch cost.

**4. Clean-up.** Live stretches shorter than 1 s between two stoppages, and live stretches with no
touch evidence at all, are folded into the stoppage. Frames outside the two graded halves are `X`.

## 2. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (`_atd.txt`, `_atd.xml`) | Yes | ball position, nearest player per team, player speeds, team of each slot |
| Tactical Smart Tagging | Yes | `SET PIECES` (restarts and their type and team) and `GOALS` only |
| Half boundaries in `games.py` | Yes | which frames are graded |
| Smart Tagging tactical phases (Build Up, Progression, ...) | No | they are phases, not possessions, and the sample matches we validated on have no equivalent |
| Tactical homography | No | not needed: the tracking is already in pitch coordinates |
| Tactical video | No | no time to build a detector |
| Broadcast video, ATD and homography | No | not synchronised to the tactical timeline |

## 3. Training and validation data

No model was trained. No hand labels and no human-produced data about the provided match were
used.

Thresholds were validated on **two other matches**: Sample Game 1 and Sample Game 2 of the Metrica
Sports public sample data (https://github.com/metrica-sports/sample-data), anonymised matches
published with tracking and event data. The event data was produced by Metrica Sports; we did no
labelling ourselves. About 286,000 frames in total.

Ground truth for those matches was built by an algorithm, following CHALLENGE.md: `D` where the
corrected tracking has no ball; otherwise the team of the latest on-ball event (set piece, pass,
shot, recovery), with a spell of the other team shorter than 2 s and with fewer than 2 passes
absorbed as "not controlled". The sample tracking is cleaner than the automatic tracking of the
competition match, so it was degraded before use: 0.3 m position jitter, short ball drop-outs, and
0, 20 or 40 losses of the ball of 6 to 12 s during live play. Scores are the competition's own
metrics, averaged over 2 matches × 3 degradation levels.

| Setting | spellF1 | macroF1 |
|---|---|---|
| **submitted settings** | **0.5819** | **0.8708** |
| activity veto off | 0.5765 | 0.8588 |
| activity veto at 0.5 / 0.7 | 0.5636 / 0.5841 | 0.8554 / 0.8683 |
| switch cost 6 / 18 / 25 | 0.5838 / 0.5525 / 0.5213 | 0.8693 / 0.8674 / 0.8652 |
| unseen-ball proof 4 s / 8 s | 0.5857 / 0.5620 | 0.8722 / 0.8665 |
| touch radius 1.5 m / 2.5 m | 0.5731 / 0.5839 | 0.8704 / 0.8701 |
| stoppage padding 0 | 0.5883 | 0.8763 |

The same two matches were used to test three further ideas, all rejected:

| Idea | Result on the sample matches | Decision |
|---|---|---|
| Kick detection (sharp change of ball velocity next to a player) as extra touch evidence | +0.007 spellF1 at best, with the dead ball given | dropped: inside the noise |
| Giving every touch episode a minimum weight | no gain | dropped |
| Extending each stoppage to the kicks either side of it (`--extend`) | −0.022 when the ball is hidden during stoppages, +0.043 when it is made partly visible (our simulation) | dropped, see section 5 |

Reading: the submitted settings sit on a plateau. No alternative improves on them by more than
0.01, which is inside the metric's noise band, so they were left unchanged. Clear losses appear
only for a high switch cost and for an 8 s unseen-ball proof. The validation script is
`solution/validation/validate_on_metrica_sample.py`.

These scores are on other matches with a different tracking source and an approximated ground
truth. They show which settings are sound; they do not predict the score on the competition match.

## 4. Possession rules implemented

- **Controlled possession:** about one second of observed touches by the new team (24 evidence
  frames to justify a spell, 12 to justify a final change), which is typically a couple of seconds
  of real time given how often the ball is unseen.
- **Loose and contested balls:** stay with the last team. A frame where the two teams' nearest
  players are within 0.5 m of each other's distance to the ball is not evidence for either.
- **Restarts:** the first team with touch evidence after a stoppage has possession from the first
  live frame; a tagged restart team is a prior, not an override.
- **Dead ball:** never assigned to a team.
- **Abstention:** `X` is used only on ungraded frames.

## 5. How the parameters were chosen

All parameters are in physical units at the top of `solution/possession.py`. The method went
through three versions:

1. **First version**, thresholds from football reasoning. Its own full-match output (15.8% dead
   ball, 148 possessions of 36 s) was implausible for a professional match. It was never scored.
2. **Second version:** the 6 s unseen-ball proof was added, the still-ball test was made robust to
   intermittent detection, and the switch cost was lowered from 25 to 12.
3. **Final version:** the player-activity veto was added, with its threshold chosen on the two
   sample matches (section 3).

A fourth variant, stoppage extension (`--extend`), was built and rejected. Its value depends on
whether the tracking shows the ball during stoppages, which we cannot observe without labels. On
the sample matches the extension added 1.4 points of dead ball when the ball was hidden and 5.5
when it was partly visible, so we fixed a rule before running it on the competition match: adopt
it only if it adds 3.5 points or more. It added 2.6, so it was not adopted. The code remains in
the repository behind the `--extend` flag, off by default.

Nothing was fitted to the public ground truth. The public 5-minute window was scored after each
version as a correctness check only:

| Version | spellF1 | macroF1 |
|---|---|---|
| second | 0.4625 | 0.6570 |
| final (submitted) | 0.4516 | 0.6834 |
| with `--extend` (rejected) | 0.4645 | 0.6765 |

The three spellF1 values differ by less than a quarter of one possession's worth on a window of
17 possessions, so they cannot rank the versions, and they were not used to. The veto was adopted
by a rule fixed before it was run (reclassify no more than a quarter of the long unseen stretches;
it reclassified 17 of 115), and the extension was rejected by the rule above even though it has
the highest public spellF1 of the three.

## 6. Limitations

- On the public window we find about the right number of stoppages but their total length is
  well under the truth's, and stoppage extension did not close that gap. We do not know where the
  missing dead-ball time goes.
- Dead-ball detection is the weakest part. A player carrying the ball to a throw-in looks like
  possession; a goalkeeper or defender standing on the ball for more than 5 s looks like a set
  piece.
- Stoppages shorter than 4 s (quick throw-ins and free kicks) are not detected.
- A false `D` in the middle of a possession splits it in two, which the ranking metric punishes.
- The activity veto was validated on matches whose tracking is cleaner than this one's.
- Only 59 set pieces are tagged, and every throw-in has no team, so most restarts rely on tracking.
- Touch evidence uses 2D distance, so a ball in the air above a player counts as near that player
  when it is slow enough.
- No use of video, so nothing recovers the ball when the tracker loses it.
- With more time: use the tactical phases as a second evidence stream in the decoder, detect
  quick restarts, and validate on automatic tracking from other matches.

## 7. Reproduction

Python 3.11 (developed on 3.11.9), numpy 2.4.6, pandas 3.1.1. Both are pinned in
`requirements.txt`.

```
pip install -r requirements.txt
set "MTAH_DATA_DIR=<folder containing tactical\>"        (Windows)
export MTAH_DATA_DIR=<folder containing tactical/>       (macOS / Linux)
python solution/possession.py --name tiki-data --no-cache
python evaluation/validate_submission.py evaluation/results/tiki-data.csv
```

The submission is written to `evaluation/results/tiki-data.csv`. The pipeline is deterministic:
the same data gives the same file. Required data files, all under `tactical/`:
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_smart_tagging.xml`.

To reproduce the validation table (optional, needs the public Metrica sample data):

```
git clone https://github.com/metrica-sports/sample-data
set "METRICA_SAMPLE_DIR=<path>\sample-data\data"
python solution/validation/validate_on_metrica_sample.py
```
