Claude
Alex’s MacBook Air, Connected
Ran 2 commands
Here is the exact order the bot goes through, using the settings in the code. The first four steps happen every morning.

1. Collect real data

It pulls every finished NBA game since the season started (Oct 19) from ESPN, with full box scores. It also pulls today’s schedule and betting lines (spread, total, moneyline), and the injury report.
If the game data fails, it keeps yesterday’s results and shows a warning. It never makes up numbers.
2. Rate each team from the box scores

Raw points mislead, because fast teams score more. So it counts possessions: field goal attempts, minus offensive rebounds, plus turnovers, plus 0.44 times free throw attempts.
Each team gets offensive rating, defensive rating and pace, per 100 possessions. These are adjusted for opponent strength, by re-running the ratings several times so a team that played good defenses gets credit.
Early in the season, ratings are pulled hard toward the league average, because a few games prove very little. The pull fades as games pile up.
It also tracks the four factors (shooting, turnovers, rebounding, free throws), the last 5 games, and rest days.
3. Project the game

Each team’s expected points come from its offense against the other team’s defense, at the expected pace.
It adds 2.2 points for home court. It subtracts 1.5 points from a team on the second night of a back-to-back.
That gives a projected margin and total.
4. Blend with the market

The betting line is smart because it already reflects injuries and sharp money. So the bot blends its number with the line instead of trusting itself alone.
The model’s share is 0.5 × games ÷ (games + 12), where games is the smaller of the two teams’ game counts. With only a few games played, that share is tiny, and the final number stays very close to the market. It grows as the season goes on.
5. Turn the number into a win chance

NBA results miss the line by about 12 points on margin and 17.5 on totals. Using that, it converts the final number into a chance of covering, going over, and winning outright.
Then it compares that chance to the price. A -110 bet implies about 52%. The difference is the edge.
6. Claude’s role

Claude reads the same data and writes the analysis: lean, reasons, risks, matchups, things to verify. It can also nudge the projection, but only by small capped amounts (about 2 points on the margin, 5 on the total) and can flag a pick to skip. It does not choose the picks.
7. Game bets: what gets published

It keeps a bet only if the edge is at least 2 points. It publishes at most 2 per game, and never two of the same type.
Confidence comes from fixed rules. It adds points for sample size (15+ games is best, under 8 games is automatically Low) and edge size (5+ points scores highest). It subtracts points for a missing injury report, a key player of uncertain status, or a big Claude adjustment.
An edge over 8 points is treated as probably a data error. It is capped at Low and flagged.
Stake is quarter Kelly, rounded to half-unit steps, between 0.5 and 2 units.
8. Player props

It matches each prop to a player’s real box-score history, so players aren’t placed on teams from memory.
Expected stat = 0.5 × season average + 0.3 × last 10 + 0.2 × last 5. The spread comes from the player’s variability, shrunk toward typical values.
It adjusts for the game environment: if the projected total is high, scoring goes up a bit, and low means down (limited to ±6%).
It computes the chance of going over and under, picks the better side, and needs an edge of at least 2 points.
Confidence tiers:
High: 12+ games, 18+ minutes per game, 4+ point edge, and the last 10 games agree with the pick.
Medium: 8+ games and a 2.5+ point edge.
Everything else is Low.
Props are skipped completely if the player is out, doubtful, questionable, day-to-day or a game-time decision. An edge over 12 points is capped at Low, like game bets.
9. Ranking and logging

Picks are ranked by edge × a confidence weight (High 1.0, Medium 0.75, Low 0.4). Picks with a real price rank above ones where I had to assume -110.
Every published pick is saved and graded the next day from the real final box scores. The Tracker shows whether the confidence levels were honest.
What the bot does not know: coaching tendencies, referees, travel, shot quality and player tracking aren’t loaded. Any coaching comments from Claude come from its memory and are tagged that way. Also, the bot hasn’t been run on real ESPN data much yet, so confidence on early picks will stay Low for a while.

how can i feed the bot more data to raise confidence on picks, and also track things like shot quality, hot zones, and player tracking. also i wanna make it look nicer and clean it up to organize more but also add team logos and player headshots nect to their picks

Ran 5 commands
Working on the logos and headshots now. Team logos will go next to teams, and player photos next to player picks and props. I’m also grouping the Picks tab into best bets and leans. Then I’ll answer your data and shot-tracking question honestly.

Shared 2 files, read 3 files, ran 5 commands
Logos and headshots are in, along with a cleaner Picks tab. I haven’t confirmed that ESPN’s image links work from the live site, because I can’t reach ESPN from here.

What changed

Logos: team logos sit next to teams on the slate, picks and parlay legs.
Headshots: player photos appear next to prop picks, in the props tables, in rotation tables and on parlay legs.
Backup if an image won’t load: a team shows its abbreviation and a player shows initials.
Picks tab: it now has two sections, “Best bets” (Medium and High confidence) and “Leans” (Low confidence).
To install it: replace both index.html and eplocks.py in your repository. For each one, click the file, click the pencil, press Cmd+A, and delete. Paste in the new contents and click Commit changes. The engine change is small: it adds player IDs so the headshots can match. After both are committed, go to Actions and run the workflow again.

How to feed the bot more data

Time is the biggest factor. Right now, any team with fewer than 8 games is capped at Low, and High needs 15 or more. Confidence will rise on its own through November.
Last season as a starting point. I can have the bot load last season’s games and give them a small, shrinking weight early on. This is the biggest thing I can add now. It would give the ratings something real to stand on in the first weeks. Rosters changed over the summer, so the weight has to stay light. I can’t test it against live ESPN data from here, so the first run will show if it works.
Real prop prices. Without prices, the bot assumes -110, and those picks rank lower. Adding the Odds API key (or pasting lines into props.txt) gives real edges.
Shot quality, hot zones and player tracking

The official NBA data is unreliable from here. NBA.com holds the real shot-zone and tracking numbers (touches, drives, defender distance), but it mostly blocks cloud servers like GitHub’s. Free tools that read it usually time out. I won’t build on it and pretend it’s solid.
What I can build from ESPN. ESPN’s play-by-play appears to include shot locations. If so, I can build shot-zone profiles (rim, mid-range, corner 3, above-the-break 3) for teams and players, hot zones, and a rough expected-shooting number from where shots are taken. That isn’t true tracking: there’s no defender distance, speed or passes. I haven’t confirmed ESPN’s location data from this environment, so the first real run decides it.
Real tracking data is paid. Providers like Second Spectrum and Synergy sell it by subscription, and I don’t have a way to connect to them.
More data doesn’t automatically mean better picks. Betting lines already price in most public information. I won’t make the bot show higher confidence just because it has more numbers. New data only changes confidence if it measurably reduces uncertainty, and the Tracker’s calibration table is how we check that.
Which should I build first: the last-season starting point, the shot-zone and hot-zone panels, or both? I recommend the last-season starting point first. It fixes the low-sample problem you’re hitting today, and the shot zones can come after the first run shows what ESPN provides.

Index
Code
·HTML
Eplocks
PY
both why not the more data the better know

Drawing court zones and shading by shot-quality value
Building both. I’m reading the engine’s data-loading and ratings code first so I can wire them in without breaking what already works.

Reading eplocks.py




Claude is responding

6 earlier steps
Add props sorter and hover polish to the site
, Done
Test the new page and publish it
, Done
Setting up daily refresh
, In progress
Build the hands-off EpLocks engine (data, model, Claude analysis, grading)
, Done
Build the GitHub Pages site and workflow files
, Done
Test everything offline with fixtures and deliver the files
, Done
Review site code and pick data shape
, Not started
Restyle site: navy, white, emerald; reorganize layout
, Not started
Build Parlays tab (2-5 legs + ultimate pick)
, Not started
Test with jsdom and send updated index.html
, Not started
Engine: last-season prior (ratings + player means)
, Not started
Engine: shot zones, hot zones, shot-quality from ESPN play-by-play
, Not started
Site: shot-profile court maps in game cards and props
, Not started
Test everything, redeliver files, update instructions
, Not started


nba_picks_bot.py
Artifact
Eplocks
PY
Index
HTML
Setup
MD
Props
TXT
Daily
YML
Index
HTML

EpLocks AI
Artifact
Eplocks site v2
HTML
Eplocks site
HTML
Eplocks ai
PY
Nba picks bot v2
PY
Nba picks bot
PY
Web search: 8 searches, 8 sites read
Web search



+5
Memory: Nba Picks Bot, NBA Picks Bot
Memory
Nba Picks Bot, NBA Picks Bot
