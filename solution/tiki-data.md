# tiki-data: technical document

More than a Hack 2027, Phase 1. Frame-level ball possession, FC Barcelona vs Athletic Club.

## 1. What we built

A set of rules, not a trained model. One Python file, `solution/possession.py`, reads the
tactical tracking and the Smart Tagging and writes a label for every frame. It runs in a couple of
minutes on a laptop and gives the same file every time.

We went with rules for a practical reason. We are not allowed labels on this match, we had no
labelled matches of our own, and the organisers' definition of possession is already written as
rules: a team has the ball until the other team controls it, and a loose ball belongs to whoever
had it last. The ranking metric also cares about whole possessions, so the job is really cutting
the match into the right pieces, and a small state machine does that directly.

## 2. How we got here

This section is the honest history, including the parts that did not work.

**Starting point.** We ran the organisers' baseline first and read the metric code. Two things
stood out. The baseline calls it a dead ball whenever the tracker loses the ball, and the ball is
missing on 41% of graded frames. And spellF1 punishes a file that flickers far more than one that
is a bit late. So our plan was: only call a dead ball when there is real proof, and make
possession sticky.

**First version.** Touches come from the nearest player to the ball. A two-state Viterbi decoder
turns touches into possession, with a cost for changing team. Dead ball needed proof: ball outside
the pitch, ball slow on a boundary line, or ball lying still. We ran it on the whole match and
looked at what came out before scoring anything: 15.8% dead ball and 148 possessions averaging
36 seconds. That is not a football match. League-wide figures for La Liga put the ball in play for
roughly 54 to 55 minutes, so close to half the clock should be dead, and possessions do not last
36 seconds on average. We never scored this version.

**Second version.** We made three changes, each from a football argument.

- A ball unseen for 6 seconds or more is proof of a stoppage. We asked ourselves whether this
  would wrongly catch passes. A short pass is in flight for under a second and a long ball or goal
  kick for 2 to 4 seconds, while a throw-in, corner or goal kick takes 8 to 30. Six seconds sits
  between the two.
- The still-ball test was rewritten so it survives the tracker blinking on and off, and set to
  5 seconds so a defender pausing on the ball is not mistaken for a free kick.
- The cost of changing team was halved, from 25 touch frames to 12.

The output became 45.4% dead ball and 267 possessions of 13.0 seconds. We checked 45% was not too
much: the graded window is about 106 minutes, so this leaves about 58 minutes of live play, a
little above the league norm. Barcelona had 63% of the live time. We then scored it on the public
five minutes for the first time, as a check that nothing was broken: spellF1 0.4625.

**The worry we could not settle by argument.** The 6-second rule assumes the tracker only loses
the ball for that long when it has left the pitch. If it loses a live ball for 8 seconds during a
passing move, we cut one possession in two. We could not measure that on this match, so we went
looking for other matches (section 4). On those, how fast the players are moving separates dead
ball from live play well by itself (AUC 0.84 to 0.89). That gave us the third version: an
unseen stretch only counts as a stoppage if the players were not busy during it.

**Questions from watching these teams.** One of us follows Barcelona closely, and we used that to
challenge the rules. We only used general knowledge of how the teams play, never notes on moments
of this match.

- *Does a centre-back ever stand on the ball for 5 seconds?* Rarely, though late in a game it can
  reach 8. We kept 5.
- *A goalkeeper can hold the ball 6 to 8 seconds after going to ground. Should the unseen rule be
  8 seconds?* We tested 8 on the other matches and it was clearly worse (0.562 against 0.582),
  because more real stoppages are lost than keeper holds are saved. We kept 6.
- *Barcelona win the ball back within a second or two all the time. Should that be one
  possession?* If the opponent only gets a touch or is tackled straight away, yes, and that is
  what the switch cost already does. We tested making it stickier and it lost badly (0.553 at 18,
  0.521 at 25). The truth has more short possessions than we produce, not fewer.
- *Is 2 metres too generous for a touch, when the ball is normally within 30 cm of the foot?* We
  tested it (section 4). Tighter is much worse. The tracker reports the middle of the player, not
  his foot, both positions are noisy, and a dribbled ball runs a metre or two ahead.

**First attempt at choosing a version.** We did not want to pick versions by the public score,
since the organisers show that picks the wrong one. So before running each new version on the
match we wrote down a rule for accepting it. For the player-activity veto: accept if it
reclassifies no more than a quarter of the long unseen stretches. It reclassified 17 of 115, so it
went in.

**The stoppage problem, and where we got it wrong.** On the public window we found about the
right number of stoppages, but in total they were less than half as long as the truth's. Our
guess was that the ball stays in view at the start and end of a stoppage, so our proof only
covers the middle. We wrote an extension that stretches each stoppage back to the last kick
before it and forward to the first kick after it. To decide whether to use it we simulated a
visible ball on the Metrica matches and set a threshold: adopt it if it adds 3.5 points of dead
ball on our match. It added 2.6, so we rejected it.

Then we asked a better question: can we check this against real automatic tracking, not our own
simulation? We could not find public tracking for a Barcelona match, but SkillCorner publish
automatic broadcast tracking for A-League matches, with labels for when the ball is in play
(section 4). Those matches look like ours: the ball is detected on 53 to 63% of frames (59% on
ours) and nearly half the match is dead ball. On them the extension improved the score in seven
matches out of eight, and it added only 1.8 to 3.3 points of dead ball. Our 2.6 is in the middle
of that range. So the threshold from our simulation was simply wrong, and we reversed the
decision. The same matches also confirmed the guess behind it: in stoppages of 6 seconds or more
the ball is still detected in the first three seconds 56 to 59% of the time.

**What we submit.** Stoppage extension on, activity veto on: 45.9% dead ball and 235 possessions
averaging 14.6 seconds.

## 3. The method

**Features.** One pass over the tracking file gives, for every frame: where the ball is, how far
it is from the nearest home player and the nearest away player, how fast the ball is moving, and
the median speed of the tracked players. We use the track slots only to know which team a player
is on.

**Dead ball.** A frame could be dead if the ball is unseen, within 1 m of a boundary line, or
lying still. A run of such frames becomes `D` only if it lasts 4 seconds and contains at least
1 second of proof:

- the ball is more than 0.5 m outside the pitch;
- the ball is on a line and moving slower than 2 m/s;
- the ball has stayed inside a 1 m circle for 5 seconds;
- the ball has been unseen for 6 seconds, and the players' median speed over that stretch is below
  the match's own 60th percentile.

Player speed is averaged over 2 seconds either side and ranked within this match, so it does not
matter how the tracking system scales its speeds. Each stoppage is trimmed to 3 seconds either
side of its proof.

The proof usually covers only the middle of a stoppage, so each one is then stretched to the kicks
either side of it: back to just after the last time the ball moved faster than 4 m/s for three
frames running (looking up to 8 seconds back), and forward to the first time it does so again
(up to 20 seconds ahead). If no such moment is found on one side, that side is left alone.

Smart Tagging then adjusts the stoppages. Each half is dead until its tagged kick-off. A tagged
goal is dead until the kick-off that follows. A stoppage ending within 5 seconds of another tagged
set piece is moved to end at the restart, which we take as the first frame in the tag's window
where the ball moves faster than 4 m/s. Where a set piece is tagged and we found no stoppage, we
walk back from the restart over frames that could be dead.

**Possession.** Inside each live stretch, a frame is a touch for a team if that team has a player
within 2 m of the ball, at least 0.5 m closer than the nearest opponent, with the ball slower than
9 m/s. The decoder picks the sequence of `H` and `A` that agrees with the most touch frames, minus
12 for every change of team. Three things follow. A deflection or a one-touch interception does
not change possession, because the other team needs about 24 touch frames to pay for switching
there and back. A real turnover is placed on the new team's first touch, because the decoder sees
the whole stretch and looks ahead, so we are not late. And frames with no touch, such as a ball in
flight, stay with the team around them, which is the organisers' loose-ball rule. If a tagged
restart names the team taking it, that team gets a head start.

**Clean-up.** Less than a second of live play between two stoppages is folded into the stoppage,
and so is a live stretch with no touch at all. Frames outside the two halves are `X`. We never
abstain inside a half.

## 4. Checking the thresholds on other matches

We trained nothing, and used no labels or human-made data about the competition match. We tested
our thresholds on ten other matches from two public sources.

### Clean tracking: two Metrica sample matches

Sample Game 1 and Sample Game 2 from the Metrica Sports
public sample data (https://github.com/metrica-sports/sample-data), about 286,000 frames. They are
anonymised matches with tracking and event data. Metrica produced the events; we labelled nothing
by hand.

We turned those into possession labels with a script, following the description in CHALLENGE.md:
dead where the corrected tracking has no ball, otherwise the team of the latest on-ball event, and
a spell shorter than 2 seconds with fewer than 2 passes is absorbed as not controlled. Their
tracking is cleaner than ours, so we roughened it: 0.3 m of jitter, short ball drop-outs, and 0,
20 or 40 losses of the ball of 6 to 12 seconds in live play. A touch is seen on 28% of graded
frames there and 27% on the competition match, which gave us some confidence the two are
comparable. Scores are the competition's own metrics, averaged over the two matches and three
levels of damage.

The table below was run with the stoppage extension off, which is how we were working at the
time.

| Setting | spellF1 | macroF1 |
|---|---|---|
| reference settings | 0.5819 | 0.8708 |
| no activity veto | 0.5765 | 0.8588 |
| veto at the 50th / 70th percentile | 0.5636 / 0.5841 | 0.8554 / 0.8683 |
| switch cost 6 / 18 / 25 | 0.5838 / 0.5525 / 0.5213 | 0.8693 / 0.8674 / 0.8652 |
| unseen-ball proof 4 s / 8 s | 0.5857 / 0.5620 | 0.8722 / 0.8665 |
| no padding round the proof | 0.5883 | 0.8763 |

| Touch radius | 0.3 m | 0.5 m | 1.0 m | 1.5 m | 2.0 m | 2.5 m | 3.0 m |
|---|---|---|---|---|---|---|---|
| spellF1 | 0.419 | 0.504 | 0.559 | 0.573 | 0.582 | 0.584 | 0.584 |

What we take from this: our settings are on flat ground. Nothing beats them by more than 0.01,
which is noise, so we left them alone. The clear mistakes would have been a high switch cost, an
8-second unseen rule, or a tight touch radius. The first of those is what our first version had.
The jitter we added penalises a tight radius, so on cleaner tracking a smaller one would do better
than the table shows.

We also split the error in two on these matches. Given the true dead ball, our decoder reaches
about 0.75. Given the true team, our dead-ball detection reaches about 0.85. Together they reach
0.58, so both halves lose points and the losses compound. Long possessions are nearly always
found (86% to 100% of those over 8 seconds). Short ones are not (38% to 47% of those lasting 2 to
4 seconds), and when we looked at the missed ones, the ball was hardly ever near a player of the
team that had it.

### Automatic tracking: eight A-League matches

The Metrica matches have corrected tracking, in which the ball disappears exactly when play is
dead. Automatic tracking does not behave like that, so we looked for some. SkillCorner's open data
(https://github.com/SkillCorner/opendata) has broadcast tracking at 10 fps for 2024/25 A-League
matches, with a "phases of play" file. A phase exists only while the ball is in play and names the
team in possession, so a frame inside a phase is `H` or `A` and a frame in no phase is `D`.
SkillCorner produced the tracking and the phases; we labelled nothing. We used eight matches,
about 1.2 million frames once repeated onto a 25 fps grid. A ball that is extrapolated rather than
detected is treated as unseen. There is no Smart Tagging, so the pipeline runs on tracking alone.

These matches resemble ours: dead ball is 42 to 54% of the match, and the ball is detected on
about 92% of live frames and 19% of dead ones. We developed on four and kept four back until the
end.

| Setting | spellF1, all 8 | first 4 | last 4 | better than submitted in |
|---|---|---|---|---|
| what we submit | 0.5393 | 0.5540 | 0.5246 | |
| no stoppage extension | 0.5210 | 0.5411 | 0.5009 | 1 of 8 |
| no activity veto | 0.5449 | 0.5641 | 0.5257 | 5 of 8 |
| switch cost 6 / 18 / 25 | 0.5102 / 0.5424 / 0.5295 | | | 1 / 6 / 1 of 8 |
| touch radius 1.0 / 1.5 / 3.0 m | 0.4839 / 0.5247 / 0.5333 | | | 0 / 0 / 2 of 8 |
| unseen-ball proof 4 s / 8 s | 0.5419 / 0.5335 | | | 6 / 2 of 8 |

The extension is the one clear result: about 0.02 better and ahead in seven matches of eight,
including all four we had kept back. Everything else is within 0.006 of what we submit. The
activity veto, which helped slightly on the Metrica matches, does nothing useful here: of the 28
long unseen stretches it reclassified as live in the first four matches, 22 were really dead. We
left it in because the difference is too small to call and it is the cautious setting, but we
would not defend it strongly.

We also looked at what is still wrong, on the first four matches. With the extension on we label
12% of dead-ball frames as possession (15% without it). About half of that comes from stoppages we never
find, mostly ones of 4 to 20 seconds where the ball stays in view, and about a third from the
start of stoppages we do find. Given the true dead ball our possession decoder reaches 0.67, and
given the true team our dead-ball detection reaches 0.82. Together they reach 0.55.

All of these numbers come from other matches, other tracking systems and other people's labels.
They tell us which settings are sensible. They do not predict our score.

## 5. Things we tried and dropped

- **Kick detection.** A sharp change in the ball's velocity next to a player, counted as an extra
  touch. It was meant to rescue the short possessions. Best gain +0.007, so no.
- **Minimum weight per touch.** Making every separate touch count for at least a few frames. No
  gain at all.

The public five minutes scored 0.4625 for the second version, 0.4516 with the activity veto added
and 0.4645 for the version we submit. On 17 possessions, one possession is worth about 0.06, so
those three are the same number. We used the public window to check the pipeline was sane and to
notice that our stoppages were too short. We did not choose a setting on it: the extension went in
because of the eight A-League matches.

## 6. What we used and what we left out

Used: the tactical tracking (`_atd.txt` and `_atd.xml`) for ball position, nearest players, player
speeds and team membership; the set pieces and goals from the Smart Tagging; the half boundaries
in `games.py`.

Left out: the tactical phase tags, because they are phases and not possessions and our sample
matches have nothing like them to test against; the homography, because the tracking is already
in pitch coordinates; both videos, because we had no time to build a ball detector; the broadcast
tracking, because it is not synchronised to the tactical timeline.

## 7. Where it fails

- Stoppages are our weakest area. A player carrying the ball to a throw-in looks like possession.
  A goalkeeper standing on the ball for more than 5 seconds looks like a set piece.
- Quick restarts under 4 seconds are missed completely.
- We still miss stoppages where the ball stays in view, such as a free kick in midfield. On the
  A-League matches we find only about 55% of stoppages lasting 4 to 10 seconds.
- Short possessions of a few seconds are often merged into their neighbours.
- A wrong stoppage in the middle of a possession splits it, and the metric punishes that twice.
- Distance is measured on the ground, so a slow ball dropping over a player's head counts as a
  touch.
- The activity veto is not earning its place on automatic tracking, as section 4 says.
- The A-League tracking is from a broadcast camera at 10 fps and ours is from a tactical camera at
  25, so that comparison is close but not exact.

With more time we would find the ball in the video for the frames where the tracker loses it, use
the tactical phases as a second source of evidence, and test on automatic tracking from other
matches.

## 8. How to run it

Python 3.11 (we used 3.11.9), numpy 2.4.6 and pandas 3.1.1, pinned in `requirements.txt`.

```
pip install -r requirements.txt
set "MTAH_DATA_DIR=<folder containing tactical\>"        (Windows)
export MTAH_DATA_DIR=<folder containing tactical/>       (macOS / Linux)
python solution/possession.py --name tiki-data --no-cache
python evaluation/validate_submission.py evaluation/results/tiki-data.csv
```

The file is written to `evaluation/results/tiki-data.csv`. It needs three files under `tactical/`:
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and `BAR-ATH_tactical_smart_tagging.xml`.

To rerun the checks in section 4, with the two public data sets:

```
git clone https://github.com/metrica-sports/sample-data
set "METRICA_SAMPLE_DIR=<path>\sample-data\data"
python solution/validation/validate_on_metrica_sample.py

git lfs install
git clone https://github.com/SkillCorner/opendata
set "SKILLCORNER_DIR=<path>\opendata\data\matches"
python solution/validation/validate_on_skillcorner.py
```
