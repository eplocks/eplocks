#!/usr/bin/env python3
"""EpLocks AI daily engine.

What it does every run (GitHub Actions calls it once a day, and again when props.txt changes):
  1. Pulls every finished NBA game and its box score from ESPN's public endpoints (cached in data/games.json).
  2. Builds team ratings (opponent-adjusted offense, defense, pace, four factors, recent form) and player game logs
     from those box scores.
  3. Pulls today's games, the market lines and the injury report.
  4. Projects each game, blends the projection with the market, and prices spread / total / moneyline edges.
  5. Reads your props.txt, routes each player to his team using the box-score data, and prices every prop from
     his game log.
  6. Asks Claude (Anthropic API) to write the analysis and to propose small, bounded adjustments. All probabilities,
     edges, confidence tiers and unit sizes are computed here in code, never by the model.
  7. Grades yesterday's picks from real box scores and writes everything to data/*.json for the website.
  8. Auditor: before today's picks, reviews the whole graded record (each market, confidence tier, smart signal and
     edge size), rates what is really working, asks Claude for short lessons and feeds them into every game's analysis.

Only the Python standard library is used. Every data source is wrapped so one failure never stops the run; the
Status tab on the site shows what loaded and what did not.
"""
import os, sys, json, re, math, time, difflib, unicodedata, hashlib
import datetime as dt
import urllib.request, urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = DATA   # the website reads data/latest.json and data/history.json from the repository root
CT = ZoneInfo('America/Chicago')
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
SEASON_START = os.environ.get('EPLOCKS_SEASON_START') or '2026-10-19'
MODEL = os.environ.get('EPLOCKS_MODEL') or 'claude-sonnet-5-5'
API_KEY = os.environ.get('ANTHROPIC_API_KEY', '').strip()
HCA = float(os.environ.get('EPLOCKS_HOME_EDGE') or 2.2)      # home-court points
SD_MARGIN, SD_TOTAL = 12.0, 17.5                              # typical NBA error vs closing lines
B2B_PTS = 1.5                                                 # heuristic back-to-back penalty (points of margin)
MAX_PROP_ROWS = 60
ENGINE_VERSION = '7.1'
SZ_V = 4               # box-score format version: 2 adds player positions, 3 adds officials and team fouls, 4 adds player shooting splits and +/- (older stored games are re-read once)
PRIOR_V = 2            # last-season summary format version (2 adds defense by position)
CLAUDE_BUDGET = float(os.environ.get('EPLOCKS_CLAUDE_SECONDS') or 1500)   # stop asking Claude after this many seconds so a run always finishes
PROP_ROWS_TO_CLAUDE = 30
T0 = time.time()
CL_T0 = None           # when the Claude phase starts (after last season has loaded), so a long first run does not eat the Claude budget
W_JUDG = float(os.environ.get('EPLOCKS_JUDGMENT_WEIGHT') or 0.10)   # how far Claude's own probability moves the model's (0 switches the judgment layer off)
JUDG_CAP = 0.04        # and never by more than this much probability, whatever the weight has grown to
JUDG = {'w': W_JUDG, 'n': 0, 'brier_j': None, 'brier_m': None}
BLEND_MAX = 0.5        # most weight the model's own number gets against the market's (last season's replay can lower it)
CLOSE_WINDOW_MIN = int(os.environ.get('EPLOCKS_CLOSE_MINUTES') or 25)    # closing mode records a pick's closing line this many minutes before tip-off
PROPS_FROM_HOUR = int(os.environ.get('EPLOCKS_PROPS_FROM_HOUR') or 15)   # props are bought only from this hour (Central) on: one fresh pull before tip-off, when lines and injuries have settled
PROPS_CLOSE_MIN_CREDITS = int(os.environ.get('EPLOCKS_PROPS_CLOSE_MIN_CREDITS') or 150)   # prop closing lines are only bought while this many credits remain
PROPS_STALE_H = 8      # reused props older than this are dropped rather than priced
PROPS_REUSE_MIN = int(os.environ.get('EPLOCKS_PROPS_REUSE_MINUTES') or 60)   # props pulled this recently are reused instead of paid for again
ODDS_MIN_CREDITS = int(os.environ.get('EPLOCKS_ODDS_MIN_CREDITS') or 40)  # never spend the Odds API key below this many credits

# Last season as a starting point, and shot locations (both optional: if either fails the run carries on without it)
PRIOR_ON = (os.environ.get('EPLOCKS_PRIOR') or 'on').lower() not in ('off', '0', 'no', 'false')
PRIOR_START = os.environ.get('EPLOCKS_PRIOR_START') or '2025-10-21'
PRIOR_END = os.environ.get('EPLOCKS_PRIOR_END') or '2026-04-13'
PRIOR_SECONDS = float(os.environ.get('EPLOCKS_PRIOR_SECONDS') or 1500)   # time budget per run for loading last season; it resumes next run
PRIOR_KEEP = 0.5       # share of last season's team strength kept as the starting point (rosters change)
PRIOR_MIN_GAMES = 300  # last-season games needed before it is used for team ratings
PRIOR_MIN_GP = 15      # last-season games a player needs before his averages are used
PRIOR_CREDIT = 4       # games of credit the confidence rules give a team that has a last-season starting point
ZONES = ('rim', 'pnt', 'ms', 'ml', 'c3', 'a3')
ZONE_LABEL = {'rim': 'Rim (0-4 ft)', 'pnt': 'Close (4-8 ft)', 'ms': 'Short mid (8-16 ft)', 'ml': 'Long mid (16 ft to the arc)', 'c3': 'Corner 3', 'a3': 'Above-the-break 3'}
RECENCY_HL = 40.0      # team games back at which a game counts half as much in the ratings
ZONE_PTS = {'rim': 2, 'pnt': 2, 'ms': 2, 'ml': 2, 'c3': 3, 'a3': 3}

STATUS = {'sources': {}, 'notes': []}


def src(name, ok, msg=''):
    STATUS['sources'][name] = {'ok': bool(ok), 'msg': str(msg)[:300]}


def note(m):
    STATUS['notes'].append(str(m)[:300])
    print('NOTE', m, flush=True)


# ----------------------------------------------------------------------------- small helpers
def today_ct():
    o = os.environ.get('EPLOCKS_TODAY')
    return dt.date.fromisoformat(o) if o else dt.datetime.now(CT).date()


def jload(path, default):
    try:
        with open(path, encoding='utf8') as f:
            return json.load(f)
    except Exception:
        return default


def jsave(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf8') as f:
        json.dump(obj, f, separators=(',', ':'), ensure_ascii=False)
    os.replace(tmp, path)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def impl(o):
    return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def r1(x):
    return None if x is None else round(float(x), 1)


def r3(x):
    return None if x is None else round(float(x), 3)


def norm_name(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"[.'`]", '', s).replace('-', ' ')
    s = re.sub(r'[^a-z ]', ' ', s)
    parts = [p for p in s.split() if p]
    while parts and parts[-1] in ('jr', 'sr', 'ii', 'iii', 'iv', 'v'):
        parts.pop()
    return ' '.join(parts)


ABBR = {'GS': 'GSW', 'NY': 'NYK', 'SA': 'SAS', 'NO': 'NOP', 'UTAH': 'UTA', 'WSH': 'WAS', 'PHO': 'PHX', 'NOR': 'NOP'}


def ab(x):
    x = (x or '').upper()
    return ABBR.get(x, x)


def http_json(url, tries=3, timeout=25):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (EpLocks AI)', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError('%s: %s' % (url, last))


# ----------------------------------------------------------------------------- ESPN: events, odds, box scores
def parse_odds(o, ev):
    """ESPN odds block -> {spread_home, total, ml_home, ml_away, spread_odds_home, spread_odds_away, provider}."""
    if not o:
        return None
    m = {'provider': (o.get('provider') or {}).get('name')}
    ou = o.get('overUnder')
    try:
        m['total'] = float(ou) if ou not in (None, '') else None
    except (TypeError, ValueError):
        m['total'] = None
    ho, ao = o.get('homeTeamOdds') or {}, o.get('awayTeamOdds') or {}
    home_ab = (ev.get('home') or {}).get('abbr')
    sp = None
    det = str(o.get('details') or '')
    mt = re.match(r'^\s*([A-Za-z]{2,4})\s+([+-]?\d+(?:\.\d+)?)\s*$', det)
    if mt:
        fav, num = ab(mt.group(1)), float(mt.group(2))
        sp = num if fav == home_ab else -num
    elif det.strip().upper() in ('EVEN', 'PK', 'PICK'):
        sp = 0.0
    if sp is None:
        try:
            raw = abs(float(o.get('spread')))
            if ho.get('favorite'):
                sp = -raw
            elif ao.get('favorite'):
                sp = raw
        except (TypeError, ValueError):
            sp = None
    m['spread_home'] = sp

    def toint(v):
        try:
            v = int(v)
            return v if abs(v) >= 100 else None
        except (TypeError, ValueError):
            return None
    m['ml_home'], m['ml_away'] = toint(ho.get('moneyLine')), toint(ao.get('moneyLine'))
    m['spread_odds_home'] = toint(ho.get('spreadOdds')) or -110
    m['spread_odds_away'] = toint(ao.get('spreadOdds')) or -110
    m['over_odds'] = toint(o.get('overOdds')) or -110
    m['under_odds'] = toint(o.get('underOdds')) or -110
    if m['total'] is None and m['spread_home'] is None and m['ml_home'] is None:
        return None
    return m


def parse_event(e):
    comp = (e.get('competitions') or [{}])[0]
    st = ((comp.get('status') or e.get('status') or {}).get('type')) or {}
    out = {'id': str(e.get('id')), 'utc': e.get('date'), 'completed': bool(st.get('completed')),
           'state': st.get('state') or ('post' if st.get('completed') else 'pre'),
           'season_type': (e.get('season') or {}).get('type')}
    try:
        d = dt.datetime.fromisoformat(str(e.get('date')).replace('Z', '+00:00')).astimezone(CT)
        out['date'] = d.date().isoformat()
        out['tip'] = d.strftime('%I:%M %p %Z').lstrip('0')
    except Exception:
        out['date'], out['tip'] = None, ''
    for c in comp.get('competitors') or []:
        side = c.get('homeAway')
        t = c.get('team') or {}
        rec = ''
        for r in c.get('records') or []:
            if r.get('type') == 'total' or r.get('name') == 'overall':
                rec = r.get('summary', '')
        if not rec and c.get('records'):
            rec = c['records'][0].get('summary', '')
        sc = c.get('score')
        if isinstance(sc, dict):
            sc = sc.get('value')
        try:
            sc = int(float(sc))
        except (TypeError, ValueError):
            sc = None
        if side in ('home', 'away'):
            out[side] = {'id': str(t.get('id')), 'abbr': ab(t.get('abbreviation')), 'name': t.get('displayName') or t.get('name') or '', 'score': sc, 'record': rec}
    odds = (comp.get('odds') or [None])[0]
    out['market'] = parse_odds(odds, out) if odds else None
    return out


def lookup(stats, names=(), labels=()):
    for s in stats or []:
        nm = str(s.get('name') or '').lower()
        lb = str(s.get('label') or s.get('abbreviation') or '').lower()
        if (nm and nm in names) or (lb and lb in labels):
            return s.get('displayValue', s.get('value'))
    return None


def pair(v):
    m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*$', str(v if v is not None else ''))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def toi(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- shot locations (ESPN play-by-play)
def median(xs):
    xs = sorted(xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) and abs(f) < 1e5 else None   # ESPN uses huge sentinel numbers for "no coordinate"
    except (TypeError, ValueError):
        return None


RIM_WORDS = ('layup', 'dunk', 'tip shot', 'tip-in', 'putback', 'finger roll', 'alley')


def zone_from_text(s):
    """Fallback when a shot has no usable coordinates: classify from the play description."""
    tl, ft = s['tl'], s.get('feet')
    if s['three']:
        return 'a3'          # the text does not say corner or not
    if any(w in tl for w in RIM_WORDS):
        return 'rim'
    if 'hook' in tl and ft is None:
        return 'pnt'
    if ft is None:
        return None
    return 'rim' if ft <= 4 else 'pnt' if ft <= 8 else 'ms' if ft <= 16 else 'ml'


def classify_shots(shots):
    """Sets s['z'] on every shot. Uses coordinates when most shots have them (self-calibrating for the unit scale and the
    hoop position, so it does not depend on how ESPN happens to number its court), otherwise the play text.
    Returns how it was done: 'coords', 'text' or 'none'."""
    withc = [s for s in shots if s.get('x') is not None and s.get('y') is not None]
    if len(withc) >= max(20, 0.5 * len(shots)):
        xs, ys = [s['x'] for s in withc], [s['y'] for s in withc]
        ymin, ymax = min(ys), max(ys)
        mid = (ymin + ymax) / 2.0
        for s in withc:
            s['fx'], s['fy'] = s['x'], (s['y'] if s['y'] <= mid else 2 * mid - s['y'])   # fold both ends onto one hoop
        rim = [s for s in withc if (not s['three']) and any(w in s['tl'] for w in RIM_WORDS)]
        if len(rim) >= 8:
            hx, hy = median([s['fx'] for s in rim]), median([s['fy'] for s in rim])
        else:
            hx, hy = (min(xs) + max(xs)) / 2.0, ymin + 0.056 * (ymax - ymin)
        for s in withc:
            s['d'] = math.hypot(s['fx'] - hx, s['fy'] - hy)
        d3 = [s['d'] for s in withc if s['three']]
        f = None
        if len(d3) >= 8:
            m3 = median(d3)
            f = 1.0 if 15 <= m3 <= 40 else 0.1 if 150 <= m3 <= 400 else 10.0 if 1.5 <= m3 <= 4 else None
        else:
            span = ymax - ymin
            f = 0.1 if span > 300 else 10.0 if span < 20 else 1.0
        if f is not None:
            for s in withc:
                d = s['d'] * f
                if s['three']:
                    if d > 38:
                        s['z'] = None                    # heave
                    elif (s['fy'] - hy) * f <= 9.0 and abs(s['fx'] - hx) * f >= 18:
                        s['z'] = 'c3'
                    else:
                        s['z'] = 'a3'
                else:
                    s['z'] = 'rim' if d <= 4 else 'pnt' if d <= 8 else 'ms' if d <= 16 else 'ml' if d <= 30 else None
            for s in shots:
                if s not in withc:
                    s['z'] = zone_from_text(s)
            return 'coords'
    for s in shots:
        s['z'] = zone_from_text(s)
    return 'text' if any(s['z'] for s in shots) else 'none'


def parse_plays(js, ev, pid_team):
    """ESPN play-by-play -> shot counts by zone, per team and per player. None when the feed has no shot plays."""
    ids = {ev['home']['id']: ev['home']['abbr'], ev['away']['id']: ev['away']['abbr']}
    shots = []
    for p in js.get('plays') or []:
        if not p.get('shootingPlay'):
            continue
        tl = str(p.get('text') or p.get('shortDescription') or '').lower()
        tp = str((p.get('type') or {}).get('text') or '').lower()
        pa = p.get('pointsAttempted')
        if 'free throw' in tl or 'free throw' in tp or pa == 1:
            continue
        three = pa == 3 or any(k in tl for k in ('three point', 'three-point', '3-pt', '3pt', 'three pointer'))
        made = bool(p.get('scoringPlay'))
        if ' misses ' in tl or ' missed ' in tl:
            made = False
        elif ' makes ' in tl or ' made ' in tl:
            made = True
        parts = p.get('participants') or []
        pid = str(((parts[0] or {}).get('athlete') or {}).get('id') or '') if parts else ''
        tid = str((p.get('team') or {}).get('id') or '')
        c = p.get('coordinate') or {}
        mf = re.search(r'(\d+)\s*-?\s*(?:foot|ft)', tl)
        shots.append({'three': three, 'made': made, 'pid': pid, 'ab': ids.get(tid) or pid_team.get(pid, ''), 'x': _num(c.get('x')), 'y': _num(c.get('y')),
                      'tl': tl, 'feet': int(mf.group(1)) if mf else None})
    if not shots:
        return None
    how = classify_shots(shots)
    sz = {'t': {}, 'p': {}, 'n': len(shots), 'c': 0, 'x': 0, 'u': 0}
    for s in shots:
        z = s.get('z')
        if not z:
            sz['u'] += 1
            continue
        sz['c' if how == 'coords' and s.get('x') is not None else 'x'] += 1
        for key, k in (('t', s['ab']), ('p', s['pid'])):
            if k:
                cell = sz[key].setdefault(k, {}).setdefault(z, [0, 0])
                cell[0] += 1
                cell[1] += 1 if s['made'] else 0
    return sz


def officials_from(js):
    """Names of the game officials in an ESPN summary (crew chief first when ESPN orders them)."""
    out = []
    try:
        for o in ((js.get('gameInfo') or {}).get('officials') or []):
            nm = (o.get('displayName') or o.get('fullName') or '').strip() if isinstance(o, dict) else ''
            if nm and nm not in out:
                out.append(nm)
    except Exception:  # noqa
        return []
    return out[:4]


def parse_summary(js, ev):
    """Box score -> compact team + player stats."""
    box = js.get('boxscore') or {}
    teams = {}
    for t in box.get('teams') or []:
        tid = str((t.get('team') or {}).get('id'))
        s = t.get('statistics') or []
        fgm, fga = pair(lookup(s, ('fieldgoalsmade-fieldgoalsattempted',), ('fg',)))
        tpm, tpa = pair(lookup(s, ('threepointfieldgoalsmade-threepointfieldgoalsattempted',), ('3pt',)))
        ftm, fta = pair(lookup(s, ('freethrowsmade-freethrowsattempted',), ('ft',)))
        teams[tid] = {
            'fgm': fgm, 'fga': fga, 'tpm': tpm, 'tpa': tpa, 'ftm': ftm, 'fta': fta,
            'orb': toi(lookup(s, ('offensiverebounds',), ('offensive rebounds', 'oreb'))),
            'drb': toi(lookup(s, ('defensiverebounds',), ('defensive rebounds', 'dreb'))),
            'reb': toi(lookup(s, ('totalrebounds', 'rebounds'), ('rebounds', 'reb'))),
            'tov': toi(lookup(s, ('totalturnovers', 'turnovers'), ('turnovers', 'to'))),
            'pf': toi(lookup(s, ('fouls', 'totalfouls', 'personalfouls'), ('fouls', 'pf', 'personal fouls'))),
            'ast': toi(lookup(s, ('assists',), ('assists', 'ast')))}
        if teams[tid]['drb'] is None and teams[tid]['reb'] is not None and teams[tid]['orb'] is not None:
            teams[tid]['drb'] = teams[tid]['reb'] - teams[tid]['orb']
    players = []
    for tp in box.get('players') or []:
        tid = str((tp.get('team') or {}).get('id'))
        for grp in tp.get('statistics') or []:
            labels = [str(x).upper() for x in (grp.get('labels') or grp.get('names') or [])]
            idx = {l: i for i, l in enumerate(labels)}
            for a in grp.get('athletes') or []:
                if a.get('didNotPlay'):
                    continue
                st = a.get('stats') or []

                def g(l):
                    i = idx.get(l)
                    return st[i] if i is not None and i < len(st) else None
                mn = toi(g('MIN'))
                if not mn:
                    continue
                ath = a.get('athlete') or {}
                players.append({'id': str(ath.get('id')), 'name': ath.get('displayName') or '', 'tid': tid, 'min': mn, 'pos': posgrp((ath.get('position') or {}).get('abbreviation')),
                                'pts': toi(g('PTS')) or 0, 'reb': toi(g('REB')) or 0, 'ast': toi(g('AST')) or 0,
                                'tpm': (pair(g('3PT'))[0] or 0), 'stl': toi(g('STL')) or 0, 'blk': toi(g('BLK')) or 0, 'tov': toi(g('TO')) or 0,
                                'fgm': pair(g('FG'))[0], 'fga': pair(g('FG'))[1], 'ftm': pair(g('FT'))[0], 'fta': pair(g('FT'))[1],
                                'orb': toi(g('OREB')), 'pf': toi(g('PF')), 'pm': toi(g('+/-'))})
    out = {'id': ev['id'], 'date': ev['date'], 'utc': ev['utc'], 'players': []}
    for side in ('home', 'away'):
        e = ev[side]
        out[side] = dict(teams.get(e['id'], {}), id=e['id'], ab=e['abbr'], name=e['name'], pts=e['score'])
    ab_by_id = {ev['home']['id']: ev['home']['abbr'], ev['away']['id']: ev['away']['abbr']}
    for p in players:
        p['t'] = ab_by_id.get(p.pop('tid'), '')
        out['players'].append(p)
    try:
        out['sz'] = parse_plays(js, ev, {p['id']: p['t'] for p in out['players']})
    except Exception as e:  # noqa
        out['sz'] = None
        note('shot locations could not be read for game %s: %s' % (ev.get('id'), str(e)[:100]))
    out['sz_v'] = SZ_V
    out['refs'] = officials_from(js)
    return out


def update_games(today):
    """Fetch any missing finished games (and their box scores) from SEASON_START to today. Returns the cache and today's events.
    One bad day never stops the others: each day is fetched on its own and retried next run."""
    path = os.path.join(DATA, 'games.json')
    cache = jload(path, {'games': {}, 'days_done': []})
    games, done = cache.setdefault('games', {}), set(cache.setdefault('days_done', []))
    d = dt.date.fromisoformat(SEASON_START)
    fetched = failed = bad_days = consec = 0
    today_ok = True
    slate_events = []
    while d <= today:
        ds = d.isoformat()
        if ds in done:
            d += dt.timedelta(days=1)
            continue
        try:
            js = http_json('%s/scoreboard?dates=%s&limit=100' % (ESPN, d.strftime('%Y%m%d')))
            consec = 0
        except Exception as e:  # noqa
            bad_days += 1
            consec += 1
            note('Scoreboard for %s could not be read (it will be retried next run): %s' % (ds, str(e)[:120]))
            if d == today:
                today_ok = False
            if consec >= 3 and d < today:
                d = today          # ESPN is not answering for older days; still try today's slate
                consec = 0
            else:
                d += dt.timedelta(days=1)
            continue
        events = [parse_event(e) for e in js.get('events') or []]
        all_final, had_fail = True, False
        for ev in events:
            if 'home' not in ev or 'away' not in ev:
                continue
            if d == today:
                slate_events.append(ev)
            if not ev['completed']:
                all_final = False
                continue
            if ev['season_type'] not in (None, 2, 3) or (ev['id'] in games and (games[ev['id']].get('sz_v') or 0) >= SZ_V):
                continue
            try:
                games[ev['id']] = parse_summary(http_json('%s/summary?event=%s' % (ESPN, ev['id'])), ev)
                fetched += 1
            except Exception as e:  # noqa
                failed += 1
                had_fail = True
                note('box score failed for game %s: %s' % (ev['id'], str(e)[:120]))
        # a day is finished once every game on it is final and stored; a postponed game must not make us re-read the day forever
        if d < today and not had_fail and (all_final or (today - d).days > 3):
            done.add(ds)
        d += dt.timedelta(days=1)
    src('espn_games', today_ok, '%d games stored, %d new, %d failed%s' % (len(games), fetched, failed, ('; %d day(s) will be retried next run' % bad_days) if bad_days else '') if today_ok
        else 'ESPN did not return today\'s scoreboard')
    try:
        n_re = refresh_shots(games)
        if n_re:
            note('Re-read %d older box scores to add player positions and shot locations' % n_re)
    except Exception as e:  # noqa
        note('Shot-location refresh failed: %s' % str(e)[:120])
    cache['days_done'] = sorted(done)
    jsave(path, cache)
    return cache, slate_events


def fetch_summaries(evs, workers=6):
    """Box scores for several games at once. Returns {game id: parsed game or the exception}."""
    out = {}

    def one(ev):
        try:
            return ev['id'], parse_summary(http_json('%s/summary?event=%s' % (ESPN, ev['id'])), ev)
        except Exception as e:  # noqa
            return ev['id'], e
    if not evs:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, v in ex.map(one, evs):
            out[k] = v
    return out


def refresh_shots(games, budget=420):
    """Games stored before shot locations were collected get their box score fetched again, once."""
    stale = [g for g in games.values() if (g.get('sz_v') or 0) < SZ_V and (g.get('home') or {}).get('id') and (g.get('away') or {}).get('id')]
    if not stale:
        return 0
    t0, done = time.time(), 0
    for i in range(0, len(stale), 12):
        if time.time() - t0 > budget:
            break
        batch = []
        for g in stale[i:i + 12]:
            batch.append({'id': g['id'], 'date': g.get('date'), 'utc': g.get('utc'),
                          'home': {'id': g['home']['id'], 'abbr': g['home']['ab'], 'name': g['home'].get('name', ''), 'score': g['home'].get('pts')},
                          'away': {'id': g['away']['id'], 'abbr': g['away']['ab'], 'name': g['away'].get('name', ''), 'score': g['away'].get('pts')}})
        res = fetch_summaries(batch)
        for ev in batch:
            r = res.get(ev['id'])
            if isinstance(r, dict):
                games[ev['id']] = r
                done += 1
    return done


# ----------------------------------------------------------------------------- referees
REF_K = 20.0           # games of league-average behaviour each official starts with, so a few games never swing anything
REF_MIN_N = 10         # games an official needs before his number can touch the projection
REF_TOTAL_W = 0.5      # share of the crew's measured total effect used, and
REF_TOTAL_CAP = 1.0    # the most it can ever move the projected total (points)


def build_ref_stats(games):
    """What each official's games look like compared with what the two teams would normally produce.
    For every game: actual total points / fouls / free-throw attempts minus the teams' own season averages, then averaged per official and shrunk toward zero."""
    gs = [g for g in games.values() if g.get('refs') and g['home'].get('pts') is not None and g['away'].get('pts') is not None]
    if len(gs) < 8:
        return None
    fns = {'tot': lambda g: g['home']['pts'] + g['away']['pts'],
           'pf': lambda g: (g['home']['pf'] + g['away']['pf']) if g['home'].get('pf') is not None and g['away'].get('pf') is not None else None,
           'fta': lambda g: (g['home']['fta'] + g['away']['fta']) if g['home'].get('fta') is not None and g['away'].get('fta') is not None else None}
    lg, resid, raw = {}, {}, {}
    for k, fn in fns.items():
        vals = [(g, fn(g)) for g in gs]
        vals = [(g, v) for g, v in vals if v is not None]
        if len(vals) < 8:
            continue
        L = mean(v for _, v in vals)
        lg[k] = L
        per = {}
        for g, v in vals:
            for t in (g['home']['ab'], g['away']['ab']):
                per.setdefault(t, []).append(v)
        tm = {t: (sum(v) + 4.0 * L) / (len(v) + 4.0) for t, v in per.items()}          # shrunk team average of this measure
        for g, v in vals:
            exp = tm[g['home']['ab']] + tm[g['away']['ab']] - L
            for nm in g['refs']:
                resid.setdefault(nm, {}).setdefault(k, []).append(v - exp)
                raw.setdefault(nm, {}).setdefault(k, []).append(v)
    refs = {}
    for nm in {g_nm for g in gs for g_nm in g['refs']}:
        n = sum(1 for g in gs if nm in g['refs'])
        e = {'n': n}
        for k in lg:
            rs = resid.get(nm, {}).get(k) or []
            e[k] = r1(sum(rs) / (len(rs) + REF_K)) if rs else 0.0
            e['avg_' + k] = r1(mean(raw[nm][k])) if raw.get(nm, {}).get(k) else None
        refs[nm] = e
    return {'refs': refs, 'lg': {k: r1(v) for k, v in lg.items()}, 'games': len(gs)}


def crew_info(names, rs, source):
    """The crew for a game with each official's history. 'adj_total' is the (small, capped) amount the projected total is moved by."""
    if not names:
        return None
    idx = {norm_name(k): v for k, v in ((rs or {}).get('refs') or {}).items()}
    rows = []
    for nm in names:
        r = idx.get(norm_name(nm))
        rows.append(dict({'name': nm, 'n': 0, 'tot': 0.0, 'pf': 0.0, 'fta': 0.0}, **(r or {})))
    solid = [x for x in rows if x['n'] >= REF_MIN_N]
    eff = {k: r1(sum(x.get(k) or 0.0 for x in rows)) for k in ('tot', 'pf', 'fta')}
    adj = 0.0
    if len(solid) >= 2 and len(solid) >= len(rows) - 1:
        adj = clamp(REF_TOTAL_W * eff['tot'], -REF_TOTAL_CAP, REF_TOTAL_CAP)
    return {'names': names, 'source': source, 'refs': rows, 'effect': eff, 'adj_total': r1(adj), 'known': sum(1 for x in rows if x['n'] > 0), 'lg': (rs or {}).get('lg')}


def crew_text(c):
    if not c:
        return None
    parts = ['%s (%d g%s)' % (x['name'], x['n'], (', total %+.1f, fouls %+.1f' % (x['tot'], x['pf'])) if x['n'] >= 3 else ', little history') for x in c['refs']]
    t = 'Officials (%s): %s. Crew effect vs what the two teams normally produce: total %+.1f pts, fouls %+.1f, free-throw attempts %+.1f (shrunk, early-season samples are small).' % (
        c['source'], '; '.join(parts), c['effect']['tot'], c['effect']['pf'], c['effect']['fta'])
    if c['adj_total']:
        t += ' The projected total was moved %+.1f for this crew.' % c['adj_total']
    return t


def fetch_assignments(events, workers=6):
    """Officials posted for today's games, from ESPN's game summary (the NBA posts crews on game day, so early runs may find none).
    Returns {game id: [names]} for the games where a crew was found."""
    out = {}

    def one(ev):
        try:
            return ev['id'], officials_from(http_json('%s/summary?event=%s' % (ESPN, ev['id']), tries=2))
        except Exception:  # noqa
            return ev['id'], None
    evs = [e for e in events if e.get('state') == 'pre']
    if evs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, v in ex.map(one, evs):
                if v:
                    out[k] = v
    return out


def parse_refs_file(path, events):
    """refs.txt (optional, owner-typed): 'LAL @ DEN: Scott Foster, Tony Brothers, Marc Davis'. Either team order works."""
    out = {}
    try:
        with open(path, encoding='utf8') as f:
            lines = f.read().splitlines()
    except Exception:  # noqa
        return out, []
    bad = []
    pairs = {frozenset((e['home']['abbr'], e['away']['abbr'])): e['id'] for e in events}
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        if ':' not in ln:
            bad.append(ln[:60])
            continue
        head, tail = ln.split(':', 1)
        tms = [ab(t.strip().upper()) for t in re.split(r'\s+(?:@|at|vs\.?|v)\s+|\s*[@/-]\s*', head, flags=re.I) if t.strip()]
        names = [n.strip() for n in tail.split(',') if n.strip()][:4]
        gid = pairs.get(frozenset(tms)) if len(tms) == 2 else None
        if gid and names:
            out[gid] = names
        else:
            bad.append(ln[:60])
    return out, bad


# ----------------------------------------------------------------------------- last season as a starting point
def zadd(dst, z, a, m, w=1.0):
    c = dst.setdefault(z, [0.0, 0.0])
    c[0] += a * w
    c[1] += m * w


def fold_prior_game(st, g):
    ab_h, ab_a = g['home']['ab'], g['away']['ab']
    keep = {k: v for k, v in g.items() if k not in ('players', 'sz')}
    st['games'][g['id']] = keep
    for p in g.get('players') or []:
        if p.get('pos'):
            opp_ = ab_a if p['t'] == ab_h else ab_h
            c = st.setdefault('posdef', {}).setdefault(opp_, {}).setdefault(p['pos'], [0, 0, 0, 0, 0])
            for i, k in enumerate(('min', 'pts', 'reb', 'ast', 'tpm')):
                c[i] += p.get(k) or 0
    for p in g.get('players') or []:
        pr = st['players'].setdefault(p['id'], {'name': p['name'], 'team': p['t'], 'last': g['date'] or '', 'n': 0, 'min': 0, 'pts': 0, 'reb': 0, 'ast': 0, 'tpm': 0, 'stl': 0, 'blk': 0, 'tov': 0})
        pr['n'] += 1
        for k in ('min', 'pts', 'reb', 'ast', 'tpm', 'stl', 'blk', 'tov'):
            pr[k] += p.get(k) or 0
        if (g['date'] or '') >= pr['last']:
            pr['last'], pr['team'], pr['name'] = g['date'] or '', p['t'], p['name']
    sz = g.get('sz')
    if sz:
        for t, zs in (sz.get('t') or {}).items():
            opp = ab_a if t == ab_h else ab_h if t == ab_a else None
            for z, (a, m) in zs.items():
                zadd(st['zones_off'].setdefault(t, {}), z, a, m)
                if opp:
                    zadd(st['zones_def'].setdefault(opp, {}), z, a, m)
        for pid, zs in (sz.get('p') or {}).items():
            for z, (a, m) in zs.items():
                zadd(st['zones_p'].setdefault(pid, {}), z, a, m)


def update_prior(today):
    """Loads last regular season in chunks (it resumes on the next run if the time budget runs out) and keeps a compact summary
    in data/prior.json. Returns the state dict, or None when switched off."""
    if not PRIOR_ON:
        return None
    path = os.path.join(DATA, 'prior.json')
    rng = [PRIOR_START, PRIOR_END]
    st = jload(path, None)
    if not st or st.get('range') != rng or st.get('v') != PRIOR_V:
        st = {'v': PRIOR_V, 'range': rng, 'posdef': {}, 'days_done': [], 'gids': [], 'games': {}, 'players': {}, 'zones_off': {}, 'zones_def': {}, 'zones_p': {}, 'complete': False}
    if st.get('complete'):
        return st
    t0 = time.time()
    d0, d1 = dt.date.fromisoformat(PRIOR_START), dt.date.fromisoformat(PRIOR_END)
    last = min(d1, today - dt.timedelta(days=1))
    done, gids = set(st['days_done']), set(st['gids'])
    bad = 0
    try:
        d = d0
        while d <= last:
            if time.time() - t0 >= PRIOR_SECONDS:
                break
            ds = d.isoformat()
            if ds not in done:
                js = http_json('%s/scoreboard?dates=%s&limit=100' % (ESPN, d.strftime('%Y%m%d')))
                evs = [parse_event(e) for e in js.get('events') or []]
                evs = [e for e in evs if 'home' in e and 'away' in e and e['completed'] and e['season_type'] in (None, 2) and e['id'] not in gids]
                res = fetch_summaries(evs)
                ok_day = True
                for e in evs:
                    r = res.get(e['id'])
                    if isinstance(r, dict):
                        fold_prior_game(st, r)
                        gids.add(e['id'])
                    else:
                        ok_day = False
                        bad += 1
                if ok_day:
                    done.add(ds)
            d += dt.timedelta(days=1)
    except Exception as e:  # noqa
        src('prior_season', False, 'last season could not be fully loaded: %s' % str(e)[:140])
    total_days = (d1 - d0).days + 1
    st['days_done'], st['gids'] = sorted(done), sorted(gids)
    st['complete'] = last >= d1 and len(done) >= total_days
    jsave(path, st)
    return st


def prior_ratings(st):
    """Team ratings for last season, built with the same code as this season's."""
    if not st or not st.get('complete') or len(st.get('games') or {}) < PRIOR_MIN_GAMES:
        return None
    r, lo, lp = build_ratings(st['games'])
    return (r, lo, lp) if r else None


def prior_player_means(st):
    out = {}
    for pid, p in ((st or {}).get('players') or {}).items():
        n = p.get('n') or 0
        if n >= PRIOR_MIN_GP:
            out[pid] = {'n': n, 'team': p.get('team'), **{k: p[k] / n for k in ('min', 'pts', 'reb', 'ast', 'tpm', 'stl', 'blk', 'tov')}}
    return out


# ----------------------------------------------------------------------------- shot profiles
def zfull(zs):
    return {z: list((zs or {}).get(z, (0, 0))) for z in ZONES}


def ztot(zs):
    return sum(v[0] for v in zs.values())


def build_shot_data(games, prior=None):
    off, dfn, pl = {}, {}, {}
    ng = tot = coord = text = unc = 0
    for g in games.values():
        sz = g.get('sz')
        if not sz:
            continue
        ng += 1
        tot += sz.get('n', 0)
        coord += sz.get('c', 0)
        text += sz.get('x', 0)
        unc += sz.get('u', 0)
        h, a = g['home']['ab'], g['away']['ab']
        for t, zs in (sz.get('t') or {}).items():
            opp = a if t == h else h if t == a else None
            for z, (at, ma) in zs.items():
                zadd(off.setdefault(t, {}), z, at, ma)
                if opp:
                    zadd(dfn.setdefault(opp, {}), z, at, ma)
        for pid, zs in (sz.get('p') or {}).items():
            for z, (at, ma) in zs.items():
                zadd(pl.setdefault(pid, {}), z, at, ma)
    used_prior = False
    if prior and prior.get('complete'):
        W = 0.4          # last season counts for less than this season
        for src_, dst in ((prior.get('zones_off') or {}, off), (prior.get('zones_def') or {}, dfn), (prior.get('zones_p') or {}, pl)):
            for k, zs in src_.items():
                for z, (a, m) in zs.items():
                    zadd(dst.setdefault(k, {}), z, a, m, W)
                    used_prior = True
    league = {}
    for t, zs in off.items():
        for z, (a, m) in zs.items():
            zadd(league, z, a, m)
    lfg = {z: league[z][1] / league[z][0] for z in ZONES if league.get(z) and league[z][0] >= 50}
    return {'off': off, 'def': dfn, 'pl': pl, 'lg': league, 'lfg': lfg, 'games': ng, 'shots': tot, 'coord': coord, 'text': text, 'unclassified': unc, 'with_prior': used_prior}


def zone_profile(zs, lfg, k=12.0):
    """{zone: [attempts, makes, shrunk FG%]}. FG% is pulled toward the league zone average so small samples do not paint false hot zones."""
    out = {}
    for z in ZONES:
        a, m = (zs or {}).get(z, (0, 0))
        fl = lfg.get(z)
        fg = (m + k * fl) / (a + k) if fl is not None else None
        out[z] = [round(a, 1), round(m, 1), r3(fg) if fg is not None else None]
    return out


def zone_quality(zs, lfg):
    zs = zfull(zs)
    tot = ztot(zs)
    if tot <= 0 or any(z not in lfg for z in ZONES):
        return None
    mix = {z: zs[z][0] / tot for z in ZONES}
    return {'n': round(tot), 'mix': {z: r3(v) for z, v in mix.items()},
            'xpps': r3(sum(mix[z] * lfg[z] * ZONE_PTS[z] for z in ZONES)), 'pps': r3(sum(zs[z][1] * ZONE_PTS[z] for z in ZONES) / tot)}


def matchup_pps(off_zs, def_zs, lfg, k=12.0):
    """Expected points per shot for an offense against a defense: average their shot mixes, and combine the two teams' zone
    accuracy as (offense accuracy + defense-allowed accuracy - league)."""
    o, d = zfull(off_zs), zfull(def_zs)
    to, td = ztot(o), ztot(d)
    if to < 40 or td < 40 or any(z not in lfg for z in ZONES):
        return None
    tot = 0.0
    for z in ZONES:
        share = (o[z][0] / to + d[z][0] / td) / 2.0
        fo = (o[z][1] + k * lfg[z]) / (o[z][0] + k)
        fd = (d[z][1] + k * lfg[z]) / (d[z][0] + k)
        tot += share * clamp(fo + fd - lfg[z], 0.05, 0.95) * ZONE_PTS[z]
    return tot


def shot_profile(home, away, sd, logs, extra_pids):
    if not sd or sd['games'] < 3 or len(sd['lfg']) < len(ZONES):
        return None
    lfg = sd['lfg']
    out = {'games': sd['games'], 'with_prior': sd['with_prior']}
    for side, t in (('away', away), ('home', home)):
        ab_ = t['abbr']
        o, d = sd['off'].get(ab_), sd['def'].get(ab_)
        if not o or not d or ztot(o) < 40 or ztot(d) < 40:
            return None
        out[side] = {'off': zone_profile(o, lfg), 'def': zone_profile(d, lfg), 'off_q': zone_quality(o, lfg), 'def_q': zone_quality(d, lfg)}
    lg = zfull(sd['lg'])
    lt = ztot(lg)
    out['matchup'] = {'away_pps': None, 'home_pps': None, 'league_pps': r3(sum(lg[z][0] / lt * lfg[z] * ZONE_PTS[z] for z in ZONES)) if lt else None}
    ma = matchup_pps(sd['off'].get(away['abbr']), sd['def'].get(home['abbr']), lfg)
    mh = matchup_pps(sd['off'].get(home['abbr']), sd['def'].get(away['abbr']), lfg)
    out['matchup']['away_pps'], out['matchup']['home_pps'] = (r3(ma) if ma else None), (r3(mh) if mh else None)
    pids, seen, players = [], set(), []
    for t in (away, home):
        pids += [p['id'] for p in (t.get('rotation') or [])[:6] if p.get('id')]
    pids += [p for p in extra_pids if p]
    for pid in pids:
        if pid in seen or pid not in logs:
            continue
        seen.add(pid)
        zs = sd['pl'].get(pid)
        if not zs or ztot(zs) < 25:
            continue
        L = logs[pid]
        players.append({'id': pid, 'name': L['name'], 'team': L['team'], 'n': round(ztot(zs)), 'zones': zone_profile(zs, lfg)})
        if len(players) >= 16:
            break
    out['players'] = players
    return out


def fmt_shots(g):
    sp = g.get('shot_profile')
    if not sp:
        return None

    def mixs(q):
        return ', '.join('%s %d%%' % (z, round(100 * q['mix'][z])) for z in ZONES) if q else 'n/a'
    L = ['Shot profile from ESPN play-by-play locations (zones: rim 0-4 ft, pnt 4-8 ft, ms short mid, ml long mid, c3 corner three, a3 above-break three; early-season samples are small, treat as context):']
    for side, t in (('away', g['away']), ('home', g['home'])):
        s = sp[side]
        oq, dq_ = s['off_q'], s['def_q']
        L.append('  %s shot mix: %s; expected points/shot %s, actual %s. Allows: %s; expected %s, actual allowed %s.' % (
            t['abbr'], mixs(oq), oq['xpps'] if oq else 'n/a', oq['pps'] if oq else 'n/a', mixs(dq_), dq_['xpps'] if dq_ else 'n/a', dq_['pps'] if dq_ else 'n/a'))
    m = sp['matchup']
    L.append('  Matchup expected points/shot: %s %s, %s %s (league %s).' % (g['away']['abbr'], m['away_pps'], g['home']['abbr'], m['home_pps'], m['league_pps']))
    hot = []
    for p in sp['players'][:12]:
        zz = [(z, p['zones'][z]) for z in ZONES if p['zones'][z][0] >= 8]
        if zz:
            z, v = max(zz, key=lambda kv: kv[1][2] or 0)
            hot.append('%s best zone %s %d/%d' % (p['name'], z, round(v[1]), round(v[0])))
    if hot:
        L.append('  Hot zones: ' + '; '.join(hot[:8]))
    return '\n'.join(L)


def fetch_injuries(name_to_abbr):
    try:
        js = http_json(ESPN + '/injuries')
    except Exception as e:  # noqa
        src('espn_injuries', False, str(e))
        return None
    out = {}
    for t in js.get('injuries') or []:
        team = name_to_abbr.get(str(t.get('displayName') or '').lower()) or ab((t.get('team') or {}).get('abbreviation'))
        for i in t.get('injuries') or []:
            ath = i.get('athlete') or {}
            name = ath.get('displayName') or i.get('name') or ''
            status = str(i.get('status') or (i.get('type') or {}).get('description') or '').strip()
            if not name:
                continue
            det = i.get('shortComment') or (i.get('details') or {}).get('detail') or ''
            out.setdefault(team, []).append({'name': name, 'status': status or 'Listed', 'note': str(det)[:200]})
    if not out:
        src('espn_injuries', False, 'the injury feed answered but listed no players, so injuries are treated as NOT LOADED')
        return None
    src('espn_injuries', True, '%d teams listed' % len(out))
    return out


# ----------------------------------------------------------------------------- ratings
def build_ratings(games, prior=None):
    recs = {}
    allg = sorted(games.values(), key=lambda g: (g.get('date') or '', g['id']))
    poss_list, pts_all, poss_all = [], 0.0, 0.0
    rows = []
    for g in allg:
        h, a = g['home'], g['away']
        if h.get('pts') is None or a.get('pts') is None:
            continue

        def poss(t):
            if None in (t.get('fga'), t.get('orb'), t.get('tov'), t.get('fta')):
                return None
            return t['fga'] - t['orb'] + t['tov'] + 0.44 * t['fta']
        ph, pa = poss(h), poss(a)
        p = mean([x for x in (ph, pa) if x is not None]) if (ph is not None or pa is not None) else None
        rows.append((g, p))
        if p:
            poss_list.append(p)
    lg_pace = mean(poss_list) if poss_list else 99.0
    for g, p in rows:
        if p is None:
            p = lg_pace
        pts_all += g['home']['pts'] + g['away']['pts']
        poss_all += 2 * p
    lg_ortg = 100.0 * pts_all / poss_all if poss_all else 112.0
    for g, p in rows:
        if p is None:
            p = lg_pace
        for me, op, loc in ((g['home'], g['away'], 'H'), (g['away'], g['home'], 'A')):
            recs.setdefault(me['ab'], []).append({
                'date': g['date'], 'poss': p, 'ortg': 100.0 * me['pts'] / p, 'drtg': 100.0 * op['pts'] / p, 'opp': op['ab'], 'loc': loc,
                'pts': me['pts'], 'opp_pts': op['pts'], 'me': me, 'op': op, 'id': g['id']})
    K, KP = 10.0, 8.0
    tgt_off, tgt_dfn, tgt_pace = {}, {}, {}
    if prior:
        pr_r, pr_lo, pr_lp = prior
        for t in recs:
            x = pr_r.get(t)
            if x:
                tgt_off[t] = lg_ortg + PRIOR_KEEP * (x['off'] - pr_lo)
                tgt_dfn[t] = lg_ortg + PRIOR_KEEP * (x['def'] - pr_lo)
                tgt_pace[t] = lg_pace + 0.7 * (x['pace'] - pr_lp)
    for t, rs_ in recs.items():
        rs_.sort(key=lambda r: (r['date'] or '', r['id']))
        for i_, r_ in enumerate(rs_):
            r_['w'] = 0.5 ** ((len(rs_) - 1 - i_) / RECENCY_HL)
    off = {t: lg_ortg for t in recs}
    dfn = {t: lg_ortg for t in recs}
    pace = {t: lg_pace for t in recs}
    for _ in range(14):
        n_off, n_dfn, n_pace = {}, {}, {}
        for t, rs in recs.items():
            n = sum(r['w'] for r in rs)
            so = sum(r['w'] * (r['ortg'] - (dfn.get(r['opp'], lg_ortg) - lg_ortg)) for r in rs)
            sd = sum(r['w'] * (r['drtg'] - (off.get(r['opp'], lg_ortg) - lg_ortg)) for r in rs)
            sp = sum(r['w'] * (r['poss'] - (pace.get(r['opp'], lg_pace) - lg_pace)) for r in rs)
            n_off[t] = (so + K * tgt_off.get(t, lg_ortg)) / (n + K)
            n_dfn[t] = (sd + K * tgt_dfn.get(t, lg_ortg)) / (n + K)
            n_pace[t] = (sp + KP * tgt_pace.get(t, lg_pace)) / (n + KP)
        off, dfn, pace = n_off, n_dfn, n_pace
    ratings = {}
    for t, rs in recs.items():
        rs = sorted(rs, key=lambda r: r['date'] or '')
        n = len(rs)
        w = sum(1 for r in rs if r['pts'] > r['opp_pts'])

        def ff(side):
            fga = sum(r[side]['fga'] or 0 for r in rs)
            fgm = sum(r[side]['fgm'] or 0 for r in rs)
            tpm = sum(r[side]['tpm'] or 0 for r in rs)
            fta = sum(r[side]['fta'] or 0 for r in rs)
            tov = sum(r[side]['tov'] or 0 for r in rs)
            orb = sum(r[side]['orb'] or 0 for r in rs)
            return fga, fgm, tpm, fta, tov, orb
        fga, fgm, tpm, fta, tov, orb = ff('me')
        ofga, ofgm, otpm, ofta, otov, oorb = ff('op')
        drb = sum(r['me']['drb'] or 0 for r in rs)
        odrb = sum(r['op']['drb'] or 0 for r in rs)
        tp = sum(r['poss'] for r in rs)
        last5 = rs[-5:]
        ratings[t] = {
            'n': n, 'w': w, 'l': n - w, 'prior': t in tgt_off, 'n_eff': n + (PRIOR_CREDIT if t in tgt_off else 0),
            'off': r1(off[t]), 'def': r1(dfn[t]), 'net': r1(off[t] - dfn[t]), 'pace': r1(pace[t]),
            'raw_off': r1(mean(r['ortg'] for r in rs)), 'raw_def': r1(mean(r['drtg'] for r in rs)),
            'ppg': r1(mean(r['pts'] for r in rs)), 'oppg': r1(mean(r['opp_pts'] for r in rs)),
            'efg': r3((fgm + 0.5 * tpm) / fga) if fga else None, 'tov_pct': r3(tov / tp) if tp else None,
            'orb_pct': r3(orb / (orb + odrb)) if (orb + odrb) else None, 'ftr': r3(fta / fga) if fga else None,
            'd_efg': r3((ofgm + 0.5 * otpm) / ofga) if ofga else None, 'd_tov_pct': r3(otov / tp) if tp else None,
            'd_orb_pct': r3(oorb / (oorb + drb)) if (oorb + drb) else None, 'd_ftr': r3(ofta / ofga) if ofga else None,
            'last5_net': r1(mean(r['ortg'] - r['drtg'] for r in last5)) if last5 else None,
            'last_date': rs[-1]['date'] if rs else None,
            'recent': [{'res': 'W' if r['pts'] > r['opp_pts'] else 'L', 'score': '%d-%d' % (r['pts'], r['opp_pts']), 'loc': r['loc'], 'opp': r['opp'], 'date': r['date']} for r in reversed(rs[-5:])]}
    return ratings, lg_ortg, lg_pace


def rest_info(rt, today):
    last = (rt or {}).get('last_date')
    if not last:
        return 'no games played yet in the data', False, None
    d = (today - dt.date.fromisoformat(last)).days
    if d <= 0:
        return 'already played today', False, 0
    if d == 1:
        return 'back-to-back (played yesterday)', True, 1
    return '%d day(s) off' % (d - 1), False, d


def project(hab, aab, ratings, lg_ortg, lg_pace, today):
    rh, ra = ratings.get(hab), ratings.get(aab)
    if not rh or not ra:
        return None
    P = rh['pace'] + ra['pace'] - lg_pace
    ph = (rh['off'] + ra['def'] - lg_ortg) * P / 100.0
    pa = (ra['off'] + rh['def'] - lg_ortg) * P / 100.0
    _, hb, _ = rest_info(rh, today)
    _, ab2, _ = rest_info(ra, today)
    margin = ph - pa + HCA - (B2B_PTS if hb else 0) + (B2B_PTS if ab2 else 0)
    return {'margin': margin, 'total': ph + pa, 'pace': P, 'home_pts': ph + HCA / 2, 'away_pts': pa - HCA / 2}


# ----------------------------------------------------------------------------- player logs and props
STAT_NAMES = {'points': 'Points', 'rebounds': 'Rebounds', 'assists': 'Assists', 'threes': '3-pointers', 'pra': 'Pts+Reb+Ast', 'pts_reb': 'Pts+Reb',
              'pts_ast': 'Pts+Ast', 'reb_ast': 'Reb+Ast', 'steals': 'Steals', 'blocks': 'Blocks', 'turnovers': 'Turnovers'}
STAT_FN = {
    'points': lambda g: g['pts'], 'rebounds': lambda g: g['reb'], 'assists': lambda g: g['ast'], 'threes': lambda g: g['tpm'],
    'pra': lambda g: g['pts'] + g['reb'] + g['ast'], 'pts_reb': lambda g: g['pts'] + g['reb'], 'pts_ast': lambda g: g['pts'] + g['ast'],
    'reb_ast': lambda g: g['reb'] + g['ast'], 'steals': lambda g: g['stl'], 'blocks': lambda g: g['blk'], 'turnovers': lambda g: g['tov']}
SD_K = {'points': 0.32, 'rebounds': 0.40, 'assists': 0.45, 'threes': 0.62, 'pra': 0.26, 'pts_reb': 0.28, 'pts_ast': 0.29, 'reb_ast': 0.34,
        'steals': 0.80, 'blocks': 0.90, 'turnovers': 0.60}
# Counting noise: a player's night-to-night swing never gets smaller than this, however steady his first few games look.
# Variance ~ DISP x mean (points come in 2s and 3s, so they swing more than a pure count). Keeps low-minute players from looking like locks.
PROP_DISP = {'points': 1.6, 'rebounds': 1.3, 'assists': 1.3, 'threes': 1.1, 'pra': 2.2, 'pts_reb': 2.0, 'pts_ast': 2.0, 'reb_ast': 1.6,
             'steals': 1.1, 'blocks': 1.2, 'turnovers': 1.1}
STAT_ALIAS = {
    'pts': 'points', 'point': 'points', 'points': 'points', 'reb': 'rebounds', 'rebs': 'rebounds', 'rebounds': 'rebounds', 'ast': 'assists', 'asts': 'assists',
    'assists': 'assists', '3pm': 'threes', '3s': 'threes', 'threes': 'threes', '3-pointers': 'threes', '3 pointers made': 'threes', 'three pointers made': 'threes',
    'pra': 'pra', 'pts+reb+ast': 'pra', 'pts+rebs+asts': 'pra', 'points+rebounds+assists': 'pra', 'pts+reb': 'pts_reb', 'pts+rebs': 'pts_reb',
    'points+rebounds': 'pts_reb', 'pts+ast': 'pts_ast', 'pts+asts': 'pts_ast', 'points+assists': 'pts_ast', 'reb+ast': 'reb_ast', 'rebs+asts': 'reb_ast',
    'rebounds+assists': 'reb_ast', 'stl': 'steals', 'steals': 'steals', 'blk': 'blocks', 'blocks': 'blocks', 'to': 'turnovers', 'tov': 'turnovers', 'turnovers': 'turnovers'}


def build_player_logs(games, prior_means=None):
    logs = {}
    for g in sorted(games.values(), key=lambda g: (g.get('date') or '', g['id'])):
        for p in g.get('players') or []:
            opp = g['away']['ab'] if p['t'] == g['home']['ab'] else g['home']['ab']
            L = logs.setdefault(p['id'], {'id': p['id'], 'name': p['name'], 'team': p['t'], 'games': []})
            L['team'], L['name'] = p['t'], p['name']
            L['games'].append(dict(p, date=g['date'], opp=opp, gid=g['id']))
    index = {}
    for pid, L in logs.items():
        index.setdefault(norm_name(L['name']), []).append(pid)
        ps_ = [g.get('pos') for g in L['games'] if g.get('pos')]
        L['pos'] = max(set(ps_), key=ps_.count) if ps_ else None
        if prior_means and pid in prior_means:
            L['prior'] = prior_means[pid]
    return logs, index


def prop_model(log, stat, line, env=1.0, mult=1.0, rm=1.0):
    fn = STAT_FN.get(stat)
    if not fn or not log:
        return None
    allg = [g for g in log['games'] if g['min'] > 0]
    cur = [g for g in allg if g.get('t') == log['team']]
    gl, changed, prior_team = allg, None, None
    if len(cur) < len(allg) and len(cur) >= 4:
        gl, changed = cur, 'traded'          # a mid-season move: only the games with his new team describe his role now
    vals = [fn(g) for g in gl]
    n = len(vals)
    if n < 3:
        return None
    season, l10, l5 = mean(vals), mean(vals[-10:]), mean(vals[-5:])
    w0, w1, w2 = PROP_CFG['w']                     # chosen by the replay on games it never fit on (see run_replay)
    base = w0 * season + w1 * l10 + w2 * l5
    if PROP_CFG.get('permin'):                     # production per minute times the minutes he is projected to play
        mins = [g['min'] for g in gl]

        def rate(v, m):
            return sum(v) / max(sum(m), 1.0)
        base = (w0 * rate(vals, mins) + w1 * rate(vals[-10:], mins[-10:]) + w2 * rate(vals[-5:], mins[-5:])) * (0.5 * mean(mins) + 0.5 * mean(mins[-5:]))
    n_eff, prior_mean = n, None
    pp = log.get('prior')
    if pp and pp.get('n', 0) >= PRIOR_MIN_GP:
        prior_mean = fn(pp)
        cur_min = mean(g['min'] for g in gl)
        pt = pp.get('team')
        if pt and pt != log['team']:
            kp, changed, prior_team = 2.0, changed or 'new team', pt       # different team than last season: a different role and system
        elif abs(cur_min - pp['min']) > 0.25 * max(pp['min'], 1.0):
            kp, changed = 3.0, changed or 'role'                          # same team but minutes changed a lot
        else:
            kp = 6.0
        base = (n * base + kp * prior_mean) / (n + kp)
        n_eff = n + (1 if kp <= 2 else 2 if kp < 6 else 4)
    prior = max(1.0, SD_K[stat] * base, math.sqrt(PROP_DISP.get(stat, 1.5) * max(base, 0.5)))
    sd = max(math.sqrt((n * pstdev(vals) ** 2 + 5 * prior ** 2) / (n + 5)), 0.9 * prior)   # a few quiet games never shrink the swing below the noise floor
    mu = base * env * mult * rm
    fx = None if REPLAY_OFF['off'] else REPLAY_FIX.get(stat)
    if fx:                       # corrections learned by replaying last season (see run_replay)
        mu, sd = mu * fx['b'], sd * fx['s']
    half = abs(line - math.floor(line)) > 1e-9
    if half:
        p_over = 1 - phi((line - mu) / sd)
        p_under = 1 - p_over
    else:
        p_over = 1 - phi((line + 0.5 - mu) / sd)
        p_under = phi((line - 0.5 - mu) / sd)
    last10 = vals[-10:]
    return {'n': n, 'n_eff': n_eff, 'prior_mean': r1(prior_mean) if prior_mean is not None else None, 'changed': changed, 'prior_team': prior_team,
            'season': r1(season), 'l10': r1(l10), 'l5': r1(l5), 'mean': r1(mu), 'sd': r1(sd), 'p_over': r3(p_over), 'p_under': r3(p_under),
            'hit_l10': sum(1 for v in last10 if v > line), 'n_l10': len(last10), 'hit_all': sum(1 for v in vals if v > line),
            'min_avg': r1(mean(g['min'] for g in gl)), 'min_l5': r1(mean(g['min'] for g in gl[-5:]))}


# ----------------------------------------------------------------------------- role changes, injuries, calibration
STAT_COMP = {'points': ['pts'], 'rebounds': ['reb'], 'assists': ['ast'], 'threes': ['tpm'], 'pra': ['pts', 'reb', 'ast'], 'pts_reb': ['pts', 'reb'],
             'pts_ast': ['pts', 'ast'], 'reb_ast': ['reb', 'ast'], 'steals': ['stl'], 'blocks': ['blk'], 'turnovers': ['tov']}


def team_gids(games):
    out = {}
    for g in sorted(games.values(), key=lambda g: (g.get('date') or '', g['id'])):
        for s in ('home', 'away'):
            out.setdefault(g[s]['ab'], []).append(g['id'])
    return out


def injury_role(pid, team_inj, logs, tg):
    """Extra minutes and shots a player should absorb tonight because rotation teammates are out or doubtful (rule-based and bounded).
    A player who has been out for weeks is already baked into the recent averages, so only the part not yet reflected is applied."""
    L = logs.get(pid)
    if not L:
        return None
    avg = lambda x, k: mean(g[k] for g in x['games'])
    my_min = avg(L, 'min')
    if my_min < 8:
        return None
    team = L['team']
    roster = [x for x in logs.values() if x['team'] == team and avg(x, 'min') >= 10]
    tgl = tg.get(team, [])
    lost_min = lost_pts = 0.0
    names, out_ids = [], set()
    for i in team_inj or []:
        w = inj_weight(i.get('status'))
        xid = i.get('id')
        if not w or not xid or xid == pid or xid not in logs:
            continue
        X = logs[xid]
        xm = avg(X, 'min')
        if xm < 15:
            continue
        have = {g['gid'] for g in X['games']}
        miss = lambda ids: (sum(1 for g in ids if g not in have) / float(len(ids))) if ids else 0.0
        refl = 0.2 * miss(tgl[-5:]) + 0.3 * miss(tgl[-10:]) + 0.5 * miss(tgl)
        f = w * (1.0 - refl)
        out_ids.add(xid)
        if f > 0.05:
            lost_min += f * xm
            lost_pts += f * avg(X, 'pts')
            names.append(X['name'])
    if lost_min <= 0:
        return None
    others = sum(avg(x, 'min') for x in roster if x['id'] not in out_ids) or 1.0
    extra = min(0.85 * lost_min * my_min / others, 7.0, 0.2 * my_min)   # a rotation spot rarely gains more than a handful of minutes
    new_min = min(40.0, my_min + extra)
    team_pts = sum(avg(x, 'pts') for x in roster) or 1.0
    return {'mf': new_min / my_min, 'usage': 1.0 + min(0.12, 0.25 * lost_pts / team_pts), 'names': names, 'extra_min': new_min - my_min}


def blowout_mult(L, margin):
    """Starters sit more in lopsided games: shave a point of minutes per point of spread beyond 8, floor 7%."""
    gap = abs(margin) - 8.0
    return max(0.93, 1.0 - 0.01 * gap) if (gap > 0 and mean(g['min'] for g in L['games']) >= 26) else 1.0


def role_mult(role, L, stat, bm, pf=1.0, pf_note=''):
    comp = STAT_COMP.get(stat)
    if not comp:
        return bm * pf, pf_note
    notes, m = [], 1.0
    if role:
        base = {k: max(mean(g[k] for g in L['games']), 0.01) for k in comp}
        use = {'pts': role['usage'], 'ast': role['usage'], 'tpm': role['usage'], 'tov': role['usage']}
        m = sum(base[k] * role['mf'] * use.get(k, 1.0) for k in comp) / sum(base.values())
        notes.append('teammates out: %s (about %+d min)' % (', '.join(role['names'][:3]), round(role['extra_min'])))
    if bm < 1.0:
        notes.append('blowout risk trims minutes')
    if pf_note:
        notes.append(pf_note)
    return clamp(m * bm * pf, 0.85, 1.3), '; '.join(notes)


def adj_text(r):
    parts = []
    if r.get('rm_note'):
        parts.append('%s (x%.2f)' % (r['rm_note'], r.get('rm', 1.0)))
    m = r.get('model') or {}
    ch = m.get('changed')
    if ch:
        parts.append({'traded': 'traded this season, only games with the new team used', 'new team': 'new team this season (was %s), last season counted less' % m.get('prior_team'),
                      'role': 'minutes changed a lot from last season, last season counted less'}.get(ch, ch))
    return '; '.join(parts) or 'none'


BOARD_STATS = ['points', 'rebounds', 'assists', 'threes', 'pra', 'pts_reb', 'pts_ast', 'reb_ast', 'steals', 'blocks', 'turnovers']


def build_player_board(teams, logs, tg, env, margin, inj_by_team, pd=None):
    """Projection for every rotation player in the game on every market, whether or not a line was posted:
    [model mean, season average, last-5 average] with the same role, injury, blowout and last-season rules as the priced props."""
    out = []
    for ti, t in enumerate(teams):
        opp = teams[1 - ti]['abbr']
        for rot in (t.get('rotation') or [])[:8]:
            L = logs.get(rot['id'])
            if not L:
                continue
            status = inj_by_team.get(t['abbr'], {}).get(norm_name(L['name']), '')
            if is_out(status):
                continue
            try:
                role = injury_role(L['id'], t.get('injuries'), logs, tg)
            except Exception:  # noqa
                role = None
            bm = blowout_mult(L, margin)
            proj = {}
            for st in BOARD_STATS:
                pf, pn_ = pos_factor(pd, opp, L.get('pos'), st)
                rm, _ = role_mult(role, L, st, bm, pf, pn_)
                pm = prop_model(L, st, 0.5, env=env, rm=rm)
                if pm:
                    proj[st] = [pm['mean'], pm['season'], pm['l5']]
            if proj:
                pf, pn_ = pos_factor(pd, opp, L.get('pos'), 'points')
                rmv, rmn = role_mult(role, L, 'points', bm, pf, pn_)
                out.append({'pid': L['id'], 'name': L['name'], 'team': t['abbr'], 'min': rot['min'], 'inj': status, 'pos': L.get('pos'), 'proj': proj,
                            'note': adj_text({'rm_note': rmn, 'rm': rmv, 'model': pm_changed(L)})})
    return out


def plog(L, n=10):
    """Last games of one player, compact, for the click-through charts: [date, opp, min, pts, reb, ast, 3pm, stl, blk, tov]."""
    gs = [g for g in L['games'] if g['min'] > 0][-n:]
    return {'name': L['name'], 'team': L['team'], 'pos': L.get('pos'), 'gp': len(L['games']),
            'avg': {k: r1(mean(g[k] for g in L['games'])) for k in ('min', 'pts', 'reb', 'ast', 'tpm', 'stl', 'blk', 'tov')},
            'l5': {k: r1(mean(g[k] for g in L['games'][-5:])) for k in ('min', 'pts', 'reb', 'ast', 'tpm')},
            'log': [[g['date'], g['opp'], g['min'], g['pts'], g['reb'], g['ast'], g['tpm'], g['stl'], g['blk'], g['tov']] for g in gs]}


def pm_changed(L):
    pm = prop_model(L, 'points', 0.5)
    return {'changed': pm['changed'], 'prior_team': pm['prior_team']} if pm else {}


CAL_PRIOR = {'game': 1.0, 'prop': 0.7}   # starting belief about how much of our edge is real (props are less proven than game bets)
CAL_STRENGTH = 1.0                       # a few hundred graded picks before the results outweigh the starting belief (results are noisy)
CAL = dict(CAL_PRIOR)


SAMPLE_K = float(os.environ.get('EPLOCKS_SAMPLE_K') or 10)   # games of evidence at which a player's own numbers get half a say against the market price


def cal_p(p, odds, cls, fair=None, n_eff=None):
    """Shrink a model probability toward the market by the learned factor, and further when the player has few games. The anchor is the
    no-vig price when known (the market's real opinion), otherwise the price itself."""
    lam = CAL.get(cls, 1.0)
    if n_eff is not None:
        lam *= n_eff / (n_eff + SAMPLE_K)
    if abs(lam - 1.0) < 1e-9:
        return p
    q = fair if fair is not None else impl(odds)
    return clamp(q + lam * (p - q), 0.01, 0.99)


def build_calibration(history):
    """Learns, from graded picks, how much of the model's edge over the price is real: outcome - price ~ lambda * (estimate - price).
    Ridge-shrunk toward a starting belief because a few hundred picks say very little."""
    rows = {'game': [], 'prop': []}
    for h in history:
        if h.get('result') not in ('W', 'L') or h.get('odds') is None or h.get('kind', 'priced') != 'priced':
            continue
        est = h.get('est_raw', h.get('est'))
        if est is None:
            continue
        q = h['fair'] if h.get('fair') is not None else impl(h['odds'])
        rows['prop' if h.get('type') == 'prop' else 'game'].append((est - q, (1.0 if h['result'] == 'W' else 0.0) - q))
    res = {}
    for k, rs in rows.items():
        sxy, sxx = sum(x * y for x, y in rs), sum(x * x for x, y in rs)
        res[k] = clamp((sxy + CAL_STRENGTH * CAL_PRIOR[k]) / (sxx + CAL_STRENGTH), 0.3, 1.0)
    return res, {k: len(v) for k, v in rows.items()}


# ----------------------------------------------------------------------------- coaching tendencies, matchups and position defense
POS_GROUP = {'PG': 'G', 'SG': 'G', 'G': 'G', 'SF': 'F', 'PF': 'F', 'F': 'F', 'GF': 'F', 'FG': 'G', 'C': 'C', 'FC': 'C', 'CF': 'C'}
POS_NAME = {'G': 'guards', 'F': 'forwards', 'C': 'centers'}
POS_STATS = ('pts', 'reb', 'ast', 'tpm')
POS_K = 300.0          # minutes of league-average play each team starts with before its own results count


def posgrp(a):
    return POS_GROUP.get(str(a or '').upper())


def build_pos_defense(games, prior=None):
    """What each defense gives up to guards, forwards and centers, per minute, against the league rate for that position.
    Sums are [minutes, points, rebounds, assists, threes]; last season counts half."""
    dfn = {}
    lg = {}

    def add(team, pos, row, w=1.0):
        c = dfn.setdefault(team, {}).setdefault(pos, [0.0] * 5)
        L = lg.setdefault(pos, [0.0] * 5)
        for i, v in enumerate(row):
            c[i] += v * w
            L[i] += v * w
    n = 0
    for g in games.values():
        used = False
        for p in g.get('players') or []:
            pos = p.get('pos')
            if not pos:
                continue
            opp = g['away']['ab'] if p['t'] == g['home']['ab'] else g['home']['ab']
            add(opp, pos, [p['min'], p['pts'], p['reb'], p['ast'], p['tpm']])
            used = True
        n += 1 if used else 0
    used_prior = False
    for team, pd in ((prior or {}).get('posdef') or {}).items():
        for pos, row in pd.items():
            add(team, pos, row, 0.5)
            used_prior = True
    if n < 8 and not used_prior:
        return None
    return {'def': dfn, 'lg': lg, 'games': n, 'prior': used_prior}


def pos_table(pd, opp):
    """{pos: {stat: ratio}}: how many points, rebounds, assists and threes this defense allows to each position versus the league."""
    out = {}
    for pos, row in ((pd or {}).get('def', {}).get(opp) or {}).items():
        lgrow = pd['lg'].get(pos)
        if not lgrow or lgrow[0] <= 0 or row[0] <= 0:
            continue
        r = {}
        for i, k in enumerate(POS_STATS, start=1):
            lrate = lgrow[i] / lgrow[0]
            if lrate <= 0:
                continue
            r[k] = (row[i] + POS_K * lrate) / ((row[0] + POS_K) * lrate)
        out[pos] = r
    return out


def pos_factor(pd, opp, pos, stat):
    """Small, bounded multiplier (at most 6%) for how this opponent defends this player's position on the stats in the market."""
    comp = [k for k in STAT_COMP.get(stat, []) if k in POS_STATS]
    if not pd or not pos or not comp:
        return 1.0, ''
    r = pos_table(pd, opp).get(pos)
    if not r:
        return 1.0, ''
    lg = pd['lg'][pos]
    w = {k: lg[1 + POS_STATS.index(k)] / lg[0] for k in comp}
    tot = sum(w.values())
    if tot <= 0:
        return 1.0, ''
    ratio = sum(w[k] * r.get(k, 1.0) for k in comp) / tot
    m = clamp(1.0 + 0.5 * (ratio - 1.0), 0.94, 1.06)
    note = ('%s defense vs %s %+d%%' % (opp, POS_NAME.get(pos, pos), round((ratio - 1.0) * 100))) if abs(ratio - 1.0) >= 0.04 else ''
    return m, note


def build_team_profiles(games):
    """How each team actually plays, measured from this season's box scores: rotation depth, how hard the starters are played,
    how soon they come out of blowouts, how stable the core is, shot choices and how the team fares on back-to-backs."""
    per = {}
    for g in sorted(games.values(), key=lambda g: (g.get('date') or '', g['id'])):
        h, a = g['home'], g['away']
        if h.get('pts') is None or a.get('pts') is None:
            continue
        for me, op, mg in ((h, a, h['pts'] - a['pts']), (a, h, a['pts'] - h['pts'])):
            ps = sorted([p for p in g.get('players') or [] if p['t'] == me['ab']], key=lambda p: -p['min'])
            per.setdefault(me['ab'], []).append({'date': g.get('date'), 'margin': mg, 'ps': ps, 'me': me, 'op': op})
    out = {}
    for t, gs in per.items():
        tot_min = {}
        for g in gs:
            for p in g['ps']:
                tot_min[p['id']] = tot_min.get(p['id'], 0) + p['min']
        core = {pid for pid, _ in sorted(tot_min.items(), key=lambda kv: -kv[1])[:5]}

        def starters(sel):
            vals = [mean(p['min'] for p in g['ps'][:5]) for g in sel if len(g['ps']) >= 5]
            return (r1(mean(vals)) if vals else None), len(vals)
        close = [g for g in gs if abs(g['margin']) <= 8]
        blow = [g for g in gs if abs(g['margin']) >= 15]
        fga = sum(g['me'].get('fga') or 0 for g in gs)
        tpa = sum(g['me'].get('tpa') or 0 for g in gs)
        fgm = sum(g['me'].get('fgm') or 0 for g in gs)
        ast = sum(g['me'].get('ast') or 0 for g in gs)
        ofga = sum(g['op'].get('fga') or 0 for g in gs)
        otpa = sum(g['op'].get('tpa') or 0 for g in gs)
        b2b, rest = [], []
        for i, g in enumerate(gs):
            try:
                gap = (dt.date.fromisoformat(g['date']) - dt.date.fromisoformat(gs[i - 1]['date'])).days if i and g['date'] and gs[i - 1]['date'] else None
            except ValueError:
                gap = None
            if gap == 1:
                b2b.append(g['margin'])
            elif gap is not None:
                rest.append(g['margin'])
        s_all, _ = starters(gs)
        s_close, n_close = starters(close)
        s_blow, n_blow = starters(blow)
        out[t] = {
            'n': len(gs), 'depth': r1(mean(sum(1 for p in g['ps'] if p['min'] >= 12) for g in gs)), 'starter_min': s_all,
            'close_min': s_close, 'n_close': n_close, 'blow_min': s_blow, 'n_blow': n_blow,
            'core': r1(mean(len([p for p in g['ps'][:5] if p['id'] in core]) for g in gs)),
            'three_rate': r3(tpa / fga) if fga else None, 'd_three_rate': r3(otpa / ofga) if ofga else None, 'ast_rate': r3(ast / fgm) if fgm else None,
            'b2b_n': len(b2b), 'b2b_margin': r1(mean(b2b)) if b2b else None, 'rest_margin': r1(mean(rest)) if rest else None}
    lg = {}
    for k in ('depth', 'starter_min', 'core', 'three_rate', 'd_three_rate', 'ast_rate'):
        v = [p[k] for p in out.values() if p.get(k) is not None]
        lg[k] = r3(mean(v)) if v else None
    return out, lg


def matchup_lines(home, away, prof, lg, ratings, pd):
    """Plain-language matchup tendencies for one game, with numbers. Used both for the site and for Claude's packet."""
    L = []
    ph, pa = prof.get(home), prof.get(away)
    rh, ra = ratings.get(home) or {}, ratings.get(away) or {}
    if rh.get('pace') is not None and ra.get('pace') is not None:
        L.append('Pace: %s %s, %s %s possessions per game (the faster team usually sets the tempo).' % (away, ra['pace'], home, rh['pace']))
    if pa and ph:
        if pa.get('three_rate') is not None and ph.get('d_three_rate') is not None:
            L.append('Threes: %s take %d%% of shots from three; %s allow %d%% (league %d%%).' % (away, round(100 * pa['three_rate']), home, round(100 * ph['d_three_rate']), round(100 * (lg.get('three_rate') or 0))))
            L.append('Threes: %s take %d%% of shots from three; %s allow %d%% (league %d%%).' % (home, round(100 * ph['three_rate']), away, round(100 * pa['d_three_rate']), round(100 * (lg.get('three_rate') or 0))))
        for t, p in ((away, pa), (home, ph)):
            if p.get('starter_min') is not None and p.get('n', 0) >= 4:
                s = '%s: starters average %s min, %s players get 12+ minutes' % (t, p['starter_min'], p['depth'])
                if p.get('blow_min') is not None and p.get('close_min') is not None and p['n_blow'] >= 2 and p['n_close'] >= 2:
                    s += '; starters play %s min in blowouts versus %s in close games (%d and %d games)' % (p['blow_min'], p['close_min'], p['n_blow'], p['n_close'])
                L.append(s + '.')
            if p.get('b2b_n', 0) >= 3 and p.get('rest_margin') is not None:
                L.append('%s: average margin %+.1f on back-to-backs (%d games) versus %+.1f otherwise.' % (t, p['b2b_margin'], p['b2b_n'], p['rest_margin']))
    if pd:
        for _, dfn in ((away, home), (home, away)):
            tb = pos_table(pd, dfn)
            bits = []
            for pos in ('G', 'F', 'C'):
                r = tb.get(pos)
                if not r:
                    continue
                for k, nm in (('pts', 'points'), ('reb', 'rebounds'), ('ast', 'assists'), ('tpm', 'threes')):
                    if k in r and abs(r[k] - 1.0) >= 0.06:
                        bits.append('%s %s %+d%%' % (POS_NAME[pos], nm, round((r[k] - 1.0) * 100)))
            if bits:
                L.append('%s defense allows (per minute, versus league): %s.' % (dfn, ', '.join(bits[:6])))
    return L


def parse_notes(path, name_to_abbr, name_index, logs):
    """notes.txt: one note per line, 'TEAM: note' (TEAM is the three-letter code or the team name) or 'Player Name: note'.
    Notes are the owner's own opinions: they are shown to Claude and on the site as context and never change a number by themselves."""
    try:
        with open(path, encoding='utf8') as f:
            raw = f.read()
    except Exception:
        return None
    teams, players, bad = {}, {}, []
    known = set(name_to_abbr.values())
    n = 0
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#') or ':' not in ln:
            if ln and not ln.startswith('#'):
                bad.append(ln[:60])
            continue
        who, txt = [x.strip() for x in ln.split(':', 1)]
        txt = txt[:300]
        if not who or not txt:
            bad.append(ln[:60])
            continue
        ab_ = ab(who)
        if ab_ in known:
            teams.setdefault(ab_, []).append(txt)
        elif who.lower() in name_to_abbr:
            teams.setdefault(name_to_abbr[who.lower()], []).append(txt)
        else:
            ids = name_index.get(norm_name(who))
            if ids:
                pid = max(ids, key=lambda i: len(logs[i]['games']))
                players.setdefault(pid, []).append(txt)
            else:
                bad.append(ln[:60])
                continue
        n += 1
    return {'teams': teams, 'players': players, 'bad': bad, 'n': n}


def notes_for_game(notes, home, away, logs):
    if not notes:
        return []
    out = []
    for ab_ in (away, home):
        for t in notes['teams'].get(ab_, []):
            out.append({'who': ab_, 'text': t})
    for pid, ts in notes['players'].items():
        L = logs.get(pid)
        if L and L['team'] in (home, away):
            for t in ts:
                out.append({'who': L['name'], 'text': t})
    return out[:12]


def inj_weight(s):
    s = str(s or '').lower()
    if 'out' in s or 'suspen' in s:
        return 1.0
    if 'doubtful' in s:
        return 0.8
    if any(k in s for k in ('questionable', 'day-to-day', 'game time', 'gtd')):
        return 0.3
    return 0.0


def is_out(s):
    s = str(s or '').lower()
    return 'out' in s or 'suspen' in s


def is_blocked(s):
    return inj_weight(s) > 0


def vs_opp(log, stat, opp):
    fn = STAT_FN[stat]
    v = [fn(g) for g in log['games'] if g['opp'] == opp and g['min'] > 0]
    return {'n': len(v), 'avg': r1(mean(v))} if v else None


PROP_PIPE = re.compile(r'^\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*(?:\|\s*([+-]?\d{3,4})?\s*(?:\|\s*([+-]?\d{3,4})?\s*)?)?$')


def odds_or_none(v):
    try:
        v = int(str(v).replace('+', ''))
        return v if abs(v) >= 100 else None
    except (TypeError, ValueError):
        return None


def parse_props_local(lines):
    """Deterministic parse of 'Player | Stat | line | over | under' lines. Returns (props, leftovers)."""
    props, left = [], []
    for ln in lines:
        m = PROP_PIPE.match(ln)
        stat = STAT_ALIAS.get(m.group(2).strip().lower()) if m else None
        if m and stat:
            props.append({'player': m.group(1).strip(), 'stat': stat, 'line': float(m.group(3)), 'over': odds_or_none(m.group(4)), 'under': odds_or_none(m.group(5)), 'raw': ln[:100]})
        else:
            left.append(ln)
    return props, left


def clean_prop(r):
    if not isinstance(r, dict):
        return None
    try:
        line = float(r.get('line'))
    except (TypeError, ValueError):
        return None
    player = str(r.get('player') or '').strip()[:60]
    stat = STAT_ALIAS.get(str(r.get('stat') or '').strip().lower(), str(r.get('stat') or '').strip().lower())
    if not player or stat not in STAT_FN:
        return None
    return {'player': player, 'stat': stat, 'line': line, 'over': odds_or_none(r.get('over')), 'under': odds_or_none(r.get('under')), 'raw': str(r.get('raw') or '')[:100]}


ODDS_KEY = os.environ.get('ODDS_API_KEY', '').strip()
ODDS_BOOK = os.environ.get('EPLOCKS_BOOKMAKER') or 'draftkings'
ODDS_MARKETS = {'player_points': 'points', 'player_rebounds': 'rebounds', 'player_assists': 'assists', 'player_threes': 'threes',
                'player_points_rebounds_assists': 'pra', 'player_points_rebounds': 'pts_reb', 'player_points_assists': 'pts_ast',
                'player_rebounds_assists': 'reb_ast', 'player_steals': 'steals', 'player_blocks': 'blocks', 'player_turnovers': 'turnovers'}


ODDS_BASE = 'https://api.the-odds-api.com/v4/sports/basketball_nba'
ODDS_REGIONS = os.environ.get('EPLOCKS_REGIONS') or 'us'
ODDS = {'left': None, 'used': 0, 'calls': 0}
STAT_MARKET = {v: k for k, v in ODDS_MARKETS.items()}


def odds_get(path, params, tries=2):
    """One Odds API call. Returns parsed JSON. Records the credits the API says are left. The key never appears in an error message."""
    q = dict(params, apiKey=ODDS_KEY)
    url = '%s/%s?%s' % (ODDS_BASE, path, urllib.parse.urlencode(q, safe=','))
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (EpLocks AI)', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=25) as r:
                body = json.loads(r.read().decode('utf-8'))
                for hk, k in (('x-requests-remaining', 'left'),):
                    try:
                        ODDS[k] = int(float(r.headers.get(hk)))
                    except (TypeError, ValueError):
                        pass
                try:
                    ODDS['used'] += int(float(r.headers.get('x-requests-last')))
                except (TypeError, ValueError):
                    pass
            ODDS['calls'] += 1
            return body
        except urllib.error.HTTPError as e:
            last = 'HTTP %s' % e.code
            if e.code in (401, 403, 404, 422, 429):
                break
            time.sleep(2 * (i + 1))
        except Exception as e:  # noqa
            last = str(e)
            time.sleep(2 * (i + 1))
    raise RuntimeError(str(last).replace(ODDS_KEY, '***')[:160])


def odds_ok(cost):
    """Stop before the monthly credits run out, so the key is never burnt to zero by the engine."""
    return bool(ODDS_KEY) and (ODDS['left'] is None or ODDS['left'] - cost >= ODDS_MIN_CREDITS)


def nick(name):
    parts = str(name or '').lower().replace('.', '').split()
    return parts[-1] if parts else ''


def devig2(o1, o2):
    """No-vig probability of side 1 from both American prices, using the power method (fairer than a straight split on lopsided prices)."""
    if o1 is None or o2 is None:
        return None
    a, b = impl(o1), impl(o2)
    if a <= 0 or b <= 0:
        return None
    if a + b <= 1.0:
        return a / (a + b)
    lo, hi = 1.0, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if a ** mid + b ** mid > 1.0:
            lo = mid
        else:
            hi = mid
    return a ** ((lo + hi) / 2.0)


def best_of(pairs):
    pairs = [p for p in pairs if p[0] is not None]
    return max(pairs, key=lambda p: p[0]) if pairs else (None, None)


def mode_line(lines):
    """Most common line; ties go to the one nearest the median."""
    if not lines:
        return None
    cnt = {}
    for l in lines:
        cnt[l] = cnt.get(l, 0) + 1
    med = median(lines)
    return sorted(cnt, key=lambda l: (-cnt[l], abs(l - med)))[0]


def agg_game_market(ev):
    """One Odds API event (every book) -> one market: the most common line, the best price at that line, no-vig fair probabilities and a book table."""
    home, away = ev.get('home_team'), ev.get('away_team')
    books = []
    for bk in ev.get('bookmakers') or []:
        d = {'book': bk.get('title') or bk.get('key') or '?'}
        for mk in bk.get('markets') or []:
            key = mk.get('key')
            for o in mk.get('outcomes') or []:
                pr, nm = odds_or_none(o.get('price')), o.get('name')
                try:
                    pt = float(o.get('point'))
                except (TypeError, ValueError):
                    pt = None
                if key == 'h2h':
                    if nm == home:
                        d['ml_home'] = pr
                    elif nm == away:
                        d['ml_away'] = pr
                elif key == 'spreads' and pt is not None:
                    if nm == home:
                        d['sp'], d['sph'] = pt, pr
                    elif nm == away:
                        d['sp_a'], d['spa'] = pt, pr
                elif key == 'totals' and pt is not None:
                    d['t'] = pt
                    if str(nm).lower() == 'over':
                        d['o'] = pr
                    elif str(nm).lower() == 'under':
                        d['u'] = pr
        if d.get('sp_a') is not None and d.get('sp') is not None and abs(d['sp_a'] + d['sp']) > 0.01:
            d.pop('sp', None)
        books.append(d)
    if not books:
        return None
    m, fair, shop = {}, {}, {}
    sb = [b for b in books if b.get('sp') is not None and b.get('sph') is not None and b.get('spa') is not None]
    if sb:
        L = mode_line([b['sp'] for b in sb])
        at = [b for b in sb if b['sp'] == L]
        m['spread_home'] = L
        m['spread_odds_home'], shop['spread_home'] = best_of([(b['sph'], b['book']) for b in at])
        m['spread_odds_away'], shop['spread_away'] = best_of([(b['spa'], b['book']) for b in at])
        fair['home_cover'] = r3(mean(devig2(b['sph'], b['spa']) for b in at))
    tb = [b for b in books if b.get('t') is not None and b.get('o') is not None and b.get('u') is not None]
    if tb:
        T = mode_line([b['t'] for b in tb])
        at = [b for b in tb if b['t'] == T]
        m['total'] = T
        m['over_odds'], shop['over'] = best_of([(b['o'], b['book']) for b in at])
        m['under_odds'], shop['under'] = best_of([(b['u'], b['book']) for b in at])
        fair['over'] = r3(mean(devig2(b['o'], b['u']) for b in at))
    mb = [b for b in books if b.get('ml_home') is not None and b.get('ml_away') is not None]
    if mb:
        m['ml_home'], shop['ml_home'] = best_of([(b['ml_home'], b['book']) for b in mb])
        m['ml_away'], shop['ml_away'] = best_of([(b['ml_away'], b['book']) for b in mb])
        fair['home_win'] = r3(mean(devig2(b['ml_home'], b['ml_away']) for b in mb))
    if not m:
        return None
    m['provider'] = '%d books' % len(books)
    m['n_books'] = len(books)
    m['fair'], m['shop'] = fair, shop
    m['books'] = [{k: v for k, v in (('b', b['book']), ('sp', b.get('sp')), ('sph', b.get('sph')), ('spa', b.get('spa')), ('t', b.get('t')), ('o', b.get('o')), ('u', b.get('u')),
                                    ('mh', b.get('ml_home')), ('ma', b.get('ml_away'))) if v is not None} for b in books[:12]]
    return m


def merge_market(espn, books):
    """Books' consensus wins wherever it has a number; ESPN's single line fills any gap."""
    if not books:
        return espn
    m = dict(espn or {})
    for k, v in books.items():
        if v is not None:
            m[k] = v
    if espn and espn.get('provider') and books.get('provider'):
        m['espn_provider'] = espn['provider']
    return m


def fetch_game_books(events):
    """Spread, total and moneyline from every US book for today's games in ONE call (3 credits). Returns {espn event id: market}."""
    if not ODDS_KEY:
        return {}
    cost = 3 * len(ODDS_REGIONS.split(','))
    if not odds_ok(cost):
        note('Odds API credits are nearly used up (%s left), so game prices come from ESPN only this run' % ODDS['left'])
        return {}
    try:
        data = odds_get('odds', {'regions': ODDS_REGIONS, 'markets': 'h2h,spreads,totals', 'oddsFormat': 'american'})
    except Exception as e:  # noqa
        src('odds_games', False, 'multi-book prices failed: %s' % e)
        return {}
    mine = {(nick(e['home']['name']), nick(e['away']['name'])): e for e in events if e.get('state', 'pre') == 'pre'}
    out = {}
    for e in data if isinstance(data, list) else []:
        ev = mine.get((nick(e.get('home_team')), nick(e.get('away_team'))))
        if ev:
            m = agg_game_market(e)
            if m:
                m['odds_event'] = e.get('id')
                out[ev['id']] = m
    nb = max([m['n_books'] for m in out.values()] or [0])
    src('odds_games', bool(out), ('%d games priced across up to %d books (best price per side, no-vig fair line)' % (len(out), nb)) if out else 'The Odds API returned no lines for today\'s games')
    return out


def agg_props(js):
    """One Odds API event with player-prop markets (every book) -> one row per player and stat at the most common line, with the best over and under price,
    no-vig fair P(over), the other lines on offer and a short book table."""
    per = {}
    for bk in js.get('bookmakers') or []:
        name = bk.get('title') or bk.get('key') or '?'
        for mk in bk.get('markets') or []:
            stat = ODDS_MARKETS.get(mk.get('key'))
            if not stat:
                continue
            tmp = {}
            for oc in mk.get('outcomes') or []:
                who = str(oc.get('description') or '').strip()
                try:
                    line = float(oc.get('point'))
                except (TypeError, ValueError):
                    continue
                side = str(oc.get('name') or '').lower()
                if who and side in ('over', 'under'):
                    tmp.setdefault((who, line), {})[side] = odds_or_none(oc.get('price'))
            for (who, line), sd in tmp.items():
                per.setdefault((who, stat), {}).setdefault(line, []).append({'b': name, 'o': sd.get('over'), 'u': sd.get('under')})
    out = []
    for (who, stat), lines in per.items():
        L = mode_line([l for l, bs in lines.items() for _ in bs])
        bs = lines[L]
        over, ob = best_of([(b['o'], b['b']) for b in bs])
        under, ub = best_of([(b['u'], b['b']) for b in bs])
        fs = [f for f in (devig2(b['o'], b['u']) for b in bs) if f is not None]
        out.append({'player': who, 'stat': stat, 'line': L, 'over': over, 'under': under, 'raw': 'Odds API, %d books' % len(bs), 'n_books': len(bs),
                    'fair_over': r3(mean(fs)) if fs else None, 'over_book': ob, 'under_book': ub,
                    'shop': [b for b in sorted(bs, key=lambda b: -(b['o'] if b['o'] is not None else -9999))][:6],
                    'alts': sorted(({'line': l, 'n': len(b)} for l, b in lines.items() if l != L), key=lambda a: -a['n'])[:3]})
    return out


def fetch_odds_props(slate_events):
    """Optional: pull player props from The Odds API for today's games, from every US book. Returns a list of prop dicts, or None on failure."""
    if not ODDS_KEY:
        return None
    markets = [m for m in (os.environ.get('EPLOCKS_PROP_MARKETS') or 'player_points,player_rebounds,player_assists,player_threes').split(',') if m.strip() in ODDS_MARKETS]
    try:
        events = odds_get('events', {})
    except Exception as e:  # noqa
        src('odds_api', False, str(e)[:200])
        return None
    mine = {}
    for ev in slate_events:
        if ev['state'] == 'pre':
            mine[(nick(ev['home']['name']), nick(ev['away']['name']))] = ev
    out, fails, per = [], 0, len(markets) * len(ODDS_REGIONS.split(','))
    # today's props are kept per game, so a run that cannot (or need not) spend credits reuses the last good lines instead of losing them
    cpath = os.path.join(DATA, 'props_auto.json')
    day = today_ct().isoformat()
    cache = jload(cpath, {})
    if cache.get('date') != day:
        cache = {'date': day, 'games': {}}
    now = time.time()
    reused, short, waited = 0, False, False
    for e in events if isinstance(events, list) else []:
        if (nick(e.get('home_team')), nick(e.get('away_team'))) not in mine:
            continue
        ce = cache['games'].get(e['id'])
        if ce and now - ce.get('t', 0) > PROPS_STALE_H * 3600:
            ce = None                       # too old: prices this stale would show edges that are no longer on offer
        fresh = ce and now - ce.get('t', 0) < PROPS_REUSE_MIN * 60
        early = dt.datetime.now(CT).hour < PROPS_FROM_HOUR and not ce
        if early:
            waited = True
            continue                        # too early to buy props: they are pulled once in the afternoon and reused after that
        if ce and not fresh and dt.datetime.now(CT).hour >= PROPS_FROM_HOUR and ODDS['left'] is not None and ODDS['left'] < 400:
            fresh = True                    # on a small plan: keep the afternoon pull instead of buying the same props again
        if fresh or not odds_ok(per):
            if not fresh:
                short = True
            if ce:
                out.extend(ce['props'])
                reused += 1
            continue
        try:
            js = odds_get('events/%s/odds' % e['id'], {'regions': ODDS_REGIONS, 'markets': ','.join(markets), 'oddsFormat': 'american'})
        except Exception as ex:  # noqa
            fails += 1
            note('Props fetch failed for %s: %s' % (e.get('id'), str(ex)[:120]))
            if ce:
                out.extend(ce['props'])
                reused += 1
            continue
        got = agg_props(js)
        if got:
            cache['games'][e['id']] = {'t': now, 'props': got}
        elif ce:
            got = ce['props']
            reused += 1
        out.extend(got)
    jsave(cpath, cache)
    if short:
        note('Odds API credits are nearly used up (%s left), so props were not refreshed this run%s' % (ODDS['left'], '; the last lines pulled today are used instead' if reused else ''))
    elif reused:
        note('Props for %d games pulled earlier today are reused to save Odds API credits' % reused)
    ok = bool(out)
    nbk = max([p['n_books'] for p in out] or [0])
    if waited and not ok:
        src('odds_api', True, 'props are pulled once a day from %d:00 Central to save credits; %s credits left' % (PROPS_FROM_HOUR, ODDS['left'] if ODDS['left'] is not None else '?'))
        return None
    src('odds_api', ok, ('%d props for %d games from up to %d books; %s credits left%s' % (len(out), len(mine), nbk, ODDS['left'] if ODDS['left'] is not None else '?', ('; %d games failed' % fails) if fails else '')) if ok else
        'The key worked but returned no player props. Your plan may not include them, or lines are not posted yet.')
    return out if ok else None


PARSE_SYS = ('You convert pasted sportsbook or pick-em text into JSON. Reply with only a JSON array. '
             'One object per player, stat and line: {"player":"full name","stat":"points|rebounds|assists|threes|pra|pts_reb|pts_ast|reb_ast|steals|blocks|turnovers|other",'
             '"line":number,"over":American odds integer or null,"under":American odds integer or null,"raw":"short source text"}. '
             'A matchup header applies to nothing here; ignore it. Never invent a line or a price. Skip anything that is not a player prop.')


def parse_props_llm(lines, cache):
    out = []
    for i in range(0, len(lines), 40):
        chunk = lines[i:i + 40]
        key = hashlib.md5('\n'.join(chunk).encode()).hexdigest()
        if key in cache:
            out.extend(cache[key])
            continue
        txt = call_claude('LINES:\n' + '\n'.join(chunk), PARSE_SYS, max_tokens=4000)
        arr = extract_json(txt) if txt else None
        got = [p for p in (clean_prop(x) for x in (arr if isinstance(arr, list) else [])) if p]
        if arr is not None:
            cache[key] = got
        out.extend(got)
    return out


def route_props(props, logs, index, slate_abbrs):
    """Attach each prop to a player log (so team and game are known from real data)."""
    keys = list(index.keys())
    routed, unmatched = [], []
    seen = set()
    for p in props:
        k = (norm_name(p['player']), p['stat'], p['line'])
        if k in seen:
            continue
        seen.add(k)
        nn = norm_name(p['player'])
        ids = index.get(nn)
        how = 'exact'
        if not ids:
            close = difflib.get_close_matches(nn, keys, n=2, cutoff=0.88)
            if len(close) == 1 or (len(close) == 2 and difflib.SequenceMatcher(None, nn, close[0]).ratio() - difflib.SequenceMatcher(None, nn, close[1]).ratio() > 0.05):
                ids, how = index[close[0]], 'fuzzy'
        if not ids:
            unmatched.append(dict(p, why='Player not found in this season\'s box scores (rookie, new to the team, or spelled differently)'))
            continue
        pid = max(ids, key=lambda i: len(logs[i]['games']))
        L = logs[pid]
        routed.append(dict(p, pid=pid, team=L['team'], how=how, name=L['name']))
    return routed, unmatched


# ----------------------------------------------------------------------------- Claude
SYSTEM = """You are EpLocks AI, a disciplined NBA betting analyst. You are given a data packet for one game: team ratings built from real box scores, a projection, the market line, injuries, player production and a table of player props with game-log numbers.

RULES
- The DATA section is the only source of 2026-27 facts. Never invent a stat, an injury, a line or a price. If something is missing, say it is missing.
- Measured coaching and matchup tendencies (rotation, minutes, blowout habits, shot mix, defense by position) are in the DATA. Use them and cite the numbers. Anything about roster changes or scheme beyond the DATA comes from your own memory and may be out of date after the 2026 offseason: tag it (from memory) and never let it drive a pick on its own.
- Owner notes, when present, are the owner's own unverified opinions. Treat them as context to check against the numbers, never as facts and never as a reason to change a number by themselves.
- The market line already reflects public injury news. When a market line is given, keep margin_adj and total_adj at 0 unless the packet shows a specific, quantifiable reason the line misses (a rating or rest mismatch with numbers). Never exceed +/-2.0 margin or +/-5.0 total when a line exists.
- Officials, when given, are measured from this season's box scores and shrunk hard. A crew number is a small nudge to totals and free throws, never a reason for a pick on its own.
- Smart signals (hit-rate, form, cushion, matchup, role, pace, minutes) are pattern flags computed from history. They are context, never a reason to move a number.
- The projection already includes who is playing (value of players out or doubtful, measured from box scores) and schedule spots when they are listed under "Already inside our projection". Never adjust margin or total again for those same absences or spots.
- TRACK RECORD, when present, is how past EpLocks picks actually graded out. Follow its lessons when you set mult and skip, respect the sample sizes it states, and say so in a prop note when a lesson changed your call. It is never a reason to invent or move a game number.
- Early season samples are small. Say so when it matters and do not overstate.
- You do not decide probabilities, confidence or stake sizes. The code computes them from your adjustments.
- For props, mult scales the player's expected output (0.90 to 1.10). Change it only for a stated reason that is in the DATA (a starter listed out raising usage, a minutes change visible in the last games). Use skip=true for any player listed Out, Doubtful or Questionable, or whose role looks unclear.
- JUDGMENT. After you have read the DATA, step back and answer as a sharp, honest handicapper who is betting their own money: what is the realistic chance of each outcome, counting what numbers cannot see (motivation and schedule spot, how this coach manages minutes, how this matchup tends to play, how much a late injury really changes things)? This is your own view, and it is scored against the model's, so be calibrated: 0.50 means no view. Stay close to the market's fair price (given in the DATA) unless you can name a concrete reason from the DATA or from firm basketball knowledge. If you have no extra insight, give the market's fair number. Probabilities above 0.80 or below 0.20 are not allowed. Set concern=true only when something specific makes betting this game risky (a key player's status unclear near tip, a line that looks stale or wrong) and say what in why.
- Plain language. No hype. Be specific and cite the numbers you use.

Reply with one JSON object only:
{"margin_adj": number, "total_adj": number,
 "judgment": {"home_win": 0.xx, "cover_home": 0.xx, "over": 0.xx, "conf": "low|med|high", "concern": false, "why": "one short sentence"},
 "analysis": {"lean": "one sentence", "why": ["3 to 5 short data-led points"], "risks": ["1 to 3 short points"], "matchups": ["0 to 3 short matchup notes, tagged (from memory) when they are"], "verify": ["what to check before betting"]},
 "props": [{"id": "p3", "mult": 1.0, "p_over": 0.xx or null, "skip": false, "note": "one short sentence"}]}
(cover_home = chance the home team covers the posted spread; over = chance the game total goes over the posted number; p_over = your own chance the player goes over his line, or null when you have no view.)"""


def call_claude(prompt, system, max_tokens=3500, tries=3):
    if not API_KEY:
        return None
    if time.time() - (CL_T0 or T0) > CLAUDE_BUDGET:
        if not STATUS.get('_budget_note'):
            STATUS['_budget_note'] = True
            note('Claude time budget used up (%d s); the remaining games get numbers but no written analysis this run' % CLAUDE_BUDGET)
        return None
    body = json.dumps({'model': MODEL, 'max_tokens': max_tokens, 'system': [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
                       'messages': [{'role': 'user', 'content': prompt}]}).encode()
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request('https://api.anthropic.com/v1/messages', data=body, headers={
                'x-api-key': API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'})
            with urllib.request.urlopen(req, timeout=240) as r:
                j = json.loads(r.read().decode('utf-8'))
            return ''.join(b.get('text', '') for b in j.get('content', []) if b.get('type') == 'text')
        except urllib.error.HTTPError as e:
            last = 'HTTP %s %s' % (e.code, e.read().decode('utf-8', 'ignore')[:200])
            if e.code in (400, 401, 403, 404):
                break
            time.sleep(20 * (i + 1))
        except Exception as e:  # noqa
            last = str(e)
            time.sleep(10 * (i + 1))
    src('anthropic', False, last)
    note('Claude call failed: %s' % str(last)[:160])
    return None


def extract_json(t):
    if not t:
        return None
    t = re.sub(r'^\s*```(?:json)?\s*', '', t.strip())
    t = re.sub(r'\s*```\s*$', '', t)
    dec = json.JSONDecoder()
    for m in re.finditer(r'[\{\[]', t):
        try:
            v, _ = dec.raw_decode(t[m.start():])
            return v
        except ValueError:
            continue
    return None


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def num_or(x, d=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def fmt_team_packet(t, side):
    r = t['ratings'] or {}
    L = ['%s: %s (%s), record %s' % (side, t['name'], t['abbr'], t['record'] or 'n/a')]
    if r:
        L.append('  Games in data: %d. Opponent-adjusted (shrunk toward league average early): offense %s, defense %s, net %s per 100 poss, pace %s.' % (r['n'], r['off'], r['def'], r['net'], r['pace']))
        L.append('  Raw: %s scored / %s allowed per game; last-5 net rating %s.' % (r['ppg'], r['oppg'], r['last5_net']))
        L.append('  Four factors (offense / defense): eFG %s / %s, TOV%% %s / %s, ORB%% %s / %s, FT rate %s / %s.' % (
            r['efg'], r['d_efg'], r['tov_pct'], r['d_tov_pct'], r['orb_pct'], r['d_orb_pct'], r['ftr'], r['d_ftr']))
        if r['recent']:
            L.append('  Recent: ' + '; '.join('%s %s %s %s' % (x['res'], x['score'], 'vs' if x['loc'] == 'H' else 'at', x['opp']) for x in r['recent']))
    L.append('  Rest: %s' % t['rest'])
    if t['rotation']:
        L.append('  Rotation (season avg min/pts/reb/ast/3pm; last-5 pts): ' + '; '.join(
            '%s %s/%s/%s/%s/%s (L5 %s)' % (p['name'], p['min'], p['pts'], p['reb'], p['ast'], p['tpm'], p['pts_l5']) for p in t['rotation']))
    if t['injuries']:
        L.append('  Injury report: ' + '; '.join('%s - %s%s' % (i['name'], i['status'], (' (' + i['note'] + ')') if i.get('note') else '') + (
            (' [avg %s min, %s pts, %s reb, %s ast]' % (i['avg']['min'], i['avg']['pts'], i['avg']['reb'], i['avg']['ast'])) if i.get('avg') else '') for i in t['injuries']))
    else:
        L.append('  Injury report: none listed' if t.get('injuries_ok') else '  Injury report: NOT LOADED')
    return '\n'.join(L)


def packet(g):
    L = ['Game: %s at %s, tip %s' % (g['away']['name'], g['home']['name'], g['tip']), fmt_team_packet(g['away'], 'Away'), fmt_team_packet(g['home'], 'Home')]
    m = g['market']
    if m:
        L.append('Market (%s): home spread %s, total %s, moneyline home %s / away %s.' % (m.get('provider') or 'ESPN', m.get('spread_home'), m.get('total'), m.get('ml_home'), m.get('ml_away')))
    else:
        L.append('Market: NO LINE LOADED for this game.')
    fr_ = market_fair(m) if m else {}
    if fr_:
        L.append('Market fair price with the bookmaker margin removed%s: home covers %s, over %s, home wins %s.' % (
            (', averaged over %d books' % m['n_books']) if m.get('n_books') else '', fr_.get('home_cover'), fr_.get('over'), fr_.get('home_win')))
    mo = g['model']
    if mo:
        L.append('Our projection from ratings: home margin %+.1f, total %.1f, pace %.1f. Blended with the market at weight %.2f -> margin %+.1f, total %.1f.' % (
            mo['margin'], mo['total'], mo['pace'], g['blend']['weight'], g['blend']['margin'], g['blend']['total']))
    mv = g.get('move') or {}
    if mv.get('sp') or mv.get('tot'):
        L.append('Market movement since this line was first seen today: spread %s, total %s (context; the market line already includes it).' % (
            ('%+g' % mv['sp']) if mv.get('sp') else 'unchanged', ('%+g' % mv['tot']) if mv.get('tot') else 'unchanged'))
    sd_ = g.get('side')
    if sd_ and sd_.get('lines'):
        L.append('Already inside our projection above (do not count again): who is playing and schedule, %+.1f margin, %+.1f total. Details:' % (sd_['margin'], sd_['total']))
        L.extend('  ' + x for x in sd_['lines'])
    if g['h2h']:
        L.append('Earlier meetings this season: ' + '; '.join(g['h2h']))
    sh = fmt_shots(g)
    if sh:
        L.append(sh)
    td = g.get('tendencies')
    if td and td.get('lines'):
        L.append('Coaching and matchup tendencies measured from this season\'s box scores (small samples early; context only):')
        L.extend('  ' + x for x in td['lines'])
    if g.get('notes'):
        L.append("Owner's notes (unverified opinions, context only):")
        L.extend('  %s: %s' % (n_['who'], n_['text']) for n_ in g['notes'])
    ct = crew_text(g.get('crew'))
    L.append(ct if ct else 'Officials: not posted yet (referee data is not part of this analysis).')
    L.append('Not loaded: play-calling and timeout data, travel distance, true player tracking (defender distance, speed, touches)' + ('' if sh else ', shot quality'))
    rows = (g.get('prop_rows') or [])[:PROP_ROWS_TO_CLAUDE]
    if rows:
        L.append('\nPROP TABLE (id | player team | stat line | over/under price | n games | season/L10/L5 avg | model mean | model P(over) | hit L10 over | minutes avg/L5 | vs this opponent):')
        for r in rows:
            m_ = r['model']
            L.append('%s | %s %s | %s %s | %s/%s | n=%d | %s/%s/%s | mean %s | P(over) %s | L10 over %d/%d | min %s/%s | vs opp %s | adjustments: %s' % (
                r['id'], r['player'], r['team'], STAT_NAMES.get(r['stat'], r['stat']), r['line'], r['over'], r['under'], m_['n'], m_['season'], m_['l10'], m_['l5'], m_['mean'],
                m_['p_over'], m_['hit_l10'], m_['n_l10'], m_['min_avg'], m_['min_l5'], ('%s avg in %d g' % (r['vs_opp']['avg'], r['vs_opp']['n'])) if r.get('vs_opp') else 'none', adj_text(r)) + ((' | market fair P(over) %s from %s books' % (r['fair_over'], r.get('n_books'))) if r.get('fair_over') is not None else '') + (
                ' | signals (%s): %s' % (r['sig']['side'], ', '.join(i['k'] for i in r['sig']['items'])) if r.get('sig') else ''))
    return '\n'.join(L)


# ----------------------------------------------------------------------------- smart signals
# A signal is a pattern in the history that supports the side of a bet. Signals never change a probability or a stake.
# They are shown to the user, and every graded pick remembers which signals it had so the site can report each signal's own record.
SIG_TIERS = ((4.0, 'Strong'), (2.5, 'Solid'), (1.5, 'Notable'))


def sig_pack(items):
    if not items:
        return None
    w = sum(i['w'] for i in items)
    return {'items': items, 'w': r1(w), 'tier': next((n for t, n in SIG_TIERS if w >= t), None), 'score': int(min(100, round(w / 6.0 * 100)))}


def prop_signals(row, side=None):
    """Patterns behind one prop row, for the side given (or the side the model prefers). Players on the injury report get none."""
    m = row.get('model')
    if not m or is_blocked(row.get('injury')):
        return None
    line = row['line']
    if side is None:
        eo = m['p_over'] - impl(row['over'] if row.get('over') is not None else -110)
        eu = m['p_under'] - impl(row['under'] if row.get('under') is not None else -110)
        side = 'over' if eo >= eu else 'under'
    over = side == 'over'
    nm = STAT_NAMES.get(row['stat'], row['stat']).lower()
    items = []
    n10 = m['n_l10']
    hit = m['hit_l10'] if over else n10 - m['hit_l10']
    if n10 >= 8 and hit / float(n10) >= 0.8:
        items.append({'k': 'hit', 'w': 1.5, 't': '%s %g %s in %d of his last %d games' % ('Cleared' if over else 'Stayed at or under', line, nm, hit, n10)})
    if m['n'] >= 6 and m['season'] > 0:
        if over and m['l5'] >= 1.10 * m['season'] and m['l5'] > line:
            items.append({'k': 'form', 'w': 1.0, 't': 'Trending up: %s over the last 5 against %s for the season' % (m['l5'], m['season'])})
        elif not over and m['l5'] <= 0.90 * m['season'] and m['l5'] < line:
            items.append({'k': 'form', 'w': 1.0, 't': 'Trending down: %s over the last 5 against %s for the season' % (m['l5'], m['season'])})
    if m['sd'] and ((m['mean'] - line) / m['sd']) * (1 if over else -1) >= 0.55:
        items.append({'k': 'cushion', 'w': 1.5, 't': 'The model expects %s against a line of %g, well clear of his usual swing (about %s)' % (m['mean'], line, m['sd'])})
    pf = row.get('pf') or 1.0
    if (over and pf >= 1.03) or (not over and pf <= 0.97):
        items.append({'k': 'matchup', 'w': 1.0, 't': row.get('pf_note') or ('Opponent defense by position favors this side (x%.2f)' % pf)})
    rm = row.get('rm') or 1.0
    if (over and rm >= 1.04) or (not over and rm <= 0.96):
        items.append({'k': 'role', 'w': 1.0, 't': (row.get('rm_note') or 'Role and minutes expected to change') + ' (x%.2f)' % rm})
    env = row.get('env') or 1.0
    if (over and env >= 1.02) or (not over and env <= 0.98):
        items.append({'k': 'pace', 'w': 0.5, 't': 'The game is projected %s than the league norm for pace and scoring' % ('faster and higher scoring' if over else 'slower and lower scoring')})
    vo = row.get('vs_opp')
    if vo and vo['n'] >= 2 and ((vo['avg'] - line) / max(line, 1.0)) * (1 if over else -1) >= 0.08:
        items.append({'k': 'opp', 'w': 0.5, 't': 'Averaged %s against this opponent in %d games (line %g)' % (vo['avg'], vo['n'], line)})
    mn, ma = m.get('min_l5') or 0, m.get('min_avg') or 0
    if ma and mn >= 20 and ((over and mn >= 1.08 * ma) or (not over and mn <= 0.92 * ma)):
        items.append({'k': 'mins', 'w': 0.5, 't': 'Minutes %s: %s over the last 5 against %s for the season' % ('rising' if over else 'falling', mn, ma)})
    out = sig_pack(items)
    if out:
        out['side'] = side
    return out


def game_signals(g, p, lg_pace):
    """Patterns behind a game pick (spread, moneyline or total)."""
    fin, mk, mo = g.get('final'), g.get('market'), g.get('model')
    a, h = g['away'], g['home']
    side, typ = p['side'], p['type']
    items = []
    if fin and mk:
        if typ in ('spread', 'moneyline') and mk.get('spread_home') is not None:
            gap = (fin['margin'] + mk['spread_home']) * (1 if side == 'home' else -1)
            if gap >= 3:
                items.append({'k': 'gap', 'w': 1.5, 't': 'Our number is %.1f points more favorable to %s than the market (%s %+g)' % (gap, g[side]['abbr'], h['abbr'], mk['spread_home'])})
        if typ == 'total' and fin.get('total') is not None and mk.get('total') is not None:
            gap = (fin['total'] - mk['total']) * (1 if side == 'over' else -1)
            if gap >= 4:
                items.append({'k': 'gap', 'w': 1.5, 't': 'Our total of %s is %.1f points %s the market (%g)' % (fin['total'], gap, 'above' if side == 'over' else 'below', mk['total'])})
    if typ in ('spread', 'moneyline'):
        me, opp = (h, a) if side == 'home' else (a, h)
        r = me.get('ratings') or {}
        if r.get('n', 0) >= 5 and r.get('last5_net') is not None and r.get('net') is not None and r['last5_net'] - r['net'] >= 4:
            items.append({'k': 'tform', 'w': 1.0, 't': '%s are playing above their season level: last-5 net rating %+.1f against %+.1f for the season' % (me['abbr'], r['last5_net'], r['net'])})
        if opp.get('b2b') and not me.get('b2b'):
            items.append({'k': 'rest', 'w': 1.0, 't': '%s are on the second night of a back-to-back and %s are rested' % (opp['abbr'], me['abbr'])})
        sp = ((g.get('shot_profile') or {}).get('matchup')) or {}
        if sp.get('home_pps') is not None and sp.get('away_pps') is not None:
            d = (sp['home_pps'] - sp['away_pps']) * (1 if side == 'home' else -1)
            if d >= 0.03:
                items.append({'k': 'shots', 'w': 1.0, 't': 'Shot quality favors %s: %.2f expected points per shot against %.2f' % (me['abbr'], sp['home_pps' if side == 'home' else 'away_pps'], sp['away_pps' if side == 'home' else 'home_pps'])})
    if typ == 'total':
        over = side == 'over'
        if mo and lg_pace and ((over and mo['pace'] >= lg_pace + 1.5) or (not over and mo['pace'] <= lg_pace - 1.5)):
            items.append({'k': 'pace', 'w': 1.0, 't': 'Projected pace %.1f against a league average of %.1f' % (mo['pace'], lg_pace)})
        sp = ((g.get('shot_profile') or {}).get('matchup')) or {}
        if sp.get('home_pps') is not None and sp.get('away_pps') is not None and sp.get('league_pps') is not None:
            d = ((sp['home_pps'] + sp['away_pps']) / 2.0 - sp['league_pps']) * (1 if over else -1)
            if d >= 0.02:
                items.append({'k': 'shots', 'w': 1.0, 't': 'Shot quality points to %s scoring in this matchup' % ('higher' if over else 'lower')})
        cr = g.get('crew')
        if cr and cr.get('adj_total') and cr['adj_total'] * (1 if over else -1) > 0:
            items.append({'k': 'crew', 'w': 0.5, 't': 'This crew\'s games run %s totals (effect %+.1f points)' % ('higher' if over else 'lower', cr['effect']['tot'])})
    if p.get('alt') and mk:
        cons = mk.get('spread_home') if typ == 'spread' else mk.get('total') if typ == 'total' else None
        if cons is not None:
            cons_side = cons if (typ == 'total' or side == 'home') else -cons
            items.append({'k': 'price', 'w': 1.0, 't': '%s has %s while most books have %s' % (
                p.get('book') or 'One book', p['label'], ('%g' % cons_side) if typ == 'total' else '%+g' % cons_side)})
    mv = g.get('move') or {}
    if typ in ('spread', 'moneyline') and mv.get('sp'):
        toward = -mv['sp'] if side == 'home' else mv['sp']          # the home spread falling means money came in on the home side
        if toward >= 1.0:
            items.append({'k': 'move', 'w': 1.0, 't': 'The line moved %.1f points toward %s since it was first seen today (%s %+g to %+g)' % (
                toward, g[side]['abbr'], h['abbr'], mv['sp0'], mk.get('spread_home'))})
    if typ == 'total' and mv.get('tot') and mv['tot'] * (1 if side == 'over' else -1) >= 1.5:
        items.append({'k': 'move', 'w': 1.0, 't': 'The total moved %+.1f since it was first seen today (%g to %g)' % (mv['tot'], mv['tot0'], mk.get('total'))})
    sd_ = g.get('side') or {}
    if sd_.get('x'):
        x_ = sd_['x']
        imp = SIDE['coef'].get('impact', 0.0) * x_.get('impact', 0.0)
        spot = sum(SIDE['coef'].get(k, 0.0) * x_.get(k, 0.0) for k in SIDE_FEATS[1:])
        if typ in ('spread', 'moneyline'):
            sg = 1 if side == 'home' else -1
            if imp * sg >= 1.0:
                items.append({'k': 'avail', 'w': 1.0, 't': 'Who is playing favors %s by %.1f points of margin' % (g[side]['abbr'], abs(imp))})
            if spot * sg >= 0.75:
                items.append({'k': 'sched', 'w': 1.0, 't': 'Schedule spot favors %s by %.1f points (measured on last season)' % (g[side]['abbr'], abs(spot))})
        elif typ == 'total' and sd_.get('total') and sd_['total'] * (1 if side == 'over' else -1) >= 1.0:
            items.append({'k': 'avail', 'w': 1.0, 't': 'Players out move the projected total %+.1f points' % sd_['total']})
    if p.get('edge', 0) >= 0.05:
        items.append({'k': 'cushion', 'w': 1.0, 't': 'The model sees a %.1f-point edge over the price' % (p['edge'] * 100)})
    return sig_pack(items)


# ----------------------------------------------------------------------------- pricing
def conf_weight(c):
    return {'High': 1.0, 'Medium': 0.75, 'Low': 0.4}.get(c, 0.4)


def kelly_units(p, odds):
    b = (odds / 100.0) if odds > 0 else (100.0 / -odds)
    f = (p * b - (1 - p)) / b
    if f <= 0:
        return 0.5
    return clamp(round(f * 100 * 0.25 * 2) / 2, 0.5, 2.0)


def game_conf(edge, min_n, inj_ok, uncertain_key, adj_abs, market_ok):
    s = 2 if min_n >= 15 else 1 if min_n >= 8 else 0
    s += 2 if edge >= 0.05 else 1 if edge >= 0.03 else 0
    if not inj_ok:
        s -= 1
    if uncertain_key:
        s -= 1
    if adj_abs > 2.5:
        s -= 1
    if not market_ok or min_n < 8:
        return 'Low'
    return 'High' if (s >= 4 and min_n >= 15) else 'Medium' if s >= 2 else 'Low'


def market_fair(m):
    """No-vig probabilities for a market: the books' consensus when we have it, otherwise worked out from ESPN's two prices."""
    f = dict((m or {}).get('fair') or {})
    if not m:
        return f
    if f.get('home_cover') is None and m.get('spread_home') is not None:
        f['home_cover'] = devig2(m.get('spread_odds_home'), m.get('spread_odds_away'))
    if f.get('over') is None and m.get('total') is not None:
        f['over'] = devig2(m.get('over_odds'), m.get('under_odds'))
    if f.get('home_win') is None and m.get('ml_home') is not None and m.get('ml_away') is not None:
        f['home_win'] = devig2(m.get('ml_home'), m.get('ml_away'))
    return f


# ----------------------------------------------------------------------------- judgment: Claude's own read, a small and tracked part of the decision
def prob_or_none(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 0.02 <= v <= 0.98 else None


def judg_mix(p, jp, fair, scale=1.0):
    """Blend Claude's own probability into the model's, a little. Its number is first held within 12 points of the market's fair price (a view far from the market
    needs a reason the data can show), then moved only JUDG['w'] of the way, and never more than JUDG_CAP of probability either way. Returns (p, jp used)."""
    if jp is None or JUDG['w'] <= 0:
        return p, None
    ref = fair if fair is not None else p
    jp = clamp(ref + clamp(jp - ref, -0.12, 0.12), 0.03, 0.97)
    shift = clamp(JUDG['w'] * scale * (jp - p), -JUDG_CAP, JUDG_CAP)
    return clamp(p + shift, 0.01, 0.99), jp


def build_judgment(history):
    """Scores Claude's own probabilities against the model's on graded picks (Brier score, lower is better). Once there are enough, the weight follows the evidence:
    it grows (to at most double) if Claude's read has been the sharper of the two, and shrinks to nothing if it has not."""
    rows = [(h['jp'], h.get('est_nj', h.get('est')), 1.0 if h['result'] == 'W' else 0.0) for h in history
            if h.get('result') in ('W', 'L') and h.get('jp') is not None and h.get('kind', 'priced') == 'priced' and h.get('est_nj', h.get('est')) is not None]
    JUDG.update(w=W_JUDG, n=len(rows), brier_j=None, brier_m=None)
    if len(rows) >= 40 and W_JUDG > 0:
        bj = mean((j - y) ** 2 for j, m_, y in rows)
        bm = mean((m_ - y) ** 2 for j, m_, y in rows)
        JUDG['brier_j'], JUDG['brier_m'] = r3(bj), r3(bm)
        skill = (bm - bj) / bm if bm > 0 else 0.0
        JUDG['w'] = clamp(W_JUDG * clamp(1.0 + 8.0 * skill, 0.0, 2.0), 0.0, 0.2)
    return dict(JUDG)


def game_picks(g, final_margin, final_total, min_n, inj_ok, uncertain_key, adj_abs, judg=None):
    m = g['market']
    picks = []
    away, home = g['away']['abbr'], g['home']['abbr']
    if not m:
        return picks
    fr, jd, shop = market_fair(m), judg or {}, m.get('shop') or {}

    def inv(x):
        return None if x is None else 1 - x
    cands = []
    if m.get('spread_home') is not None:
        L = m['spread_home']
        p_home = phi((final_margin + L) / SD_MARGIN)
        cands.append(('spread', 'home', '%s %+g' % (home, L), L, m['spread_odds_home'], p_home, fr.get('home_cover'), jd.get('cover_home'), 'spread_home'))
        cands.append(('spread', 'away', '%s %+g' % (away, -L), -L, m['spread_odds_away'], 1 - p_home, inv(fr.get('home_cover')), inv(jd.get('cover_home')), 'spread_away'))
    if m.get('total') is not None and final_total is not None:
        T = m['total']
        p_over = 1 - phi((T - final_total) / SD_TOTAL)
        cands.append(('total', 'over', 'Over %g' % T, T, m['over_odds'], p_over, fr.get('over'), jd.get('over'), 'over'))
        cands.append(('total', 'under', 'Under %g' % T, T, m['under_odds'], 1 - p_over, inv(fr.get('over')), inv(jd.get('over')), 'under'))
    if m.get('ml_home') is not None and m.get('ml_away') is not None:
        pw = phi(final_margin / SD_MARGIN)
        cands.append(('moneyline', 'home', '%s ML' % home, None, m['ml_home'], pw, fr.get('home_win'), jd.get('home_win'), 'ml_home'))
        cands.append(('moneyline', 'away', '%s ML' % away, None, m['ml_away'], 1 - pw, inv(fr.get('home_win')), inv(jd.get('home_win')), 'ml_away'))
    # Line shopping across numbers: a book hanging a different spread or total than the consensus is priced at its own line,
    # with the consensus fair chance moved by the usual value of a point, so a stale or generous number shows up as edge.
    for b in m.get('books') or []:
        bk = 'book:' + str(b.get('b') or '?')
        L = m.get('spread_home')
        if L is not None and b.get('sp') is not None and b['sp'] != L and abs(b['sp'] - L) <= 2.5 and b.get('sph') is not None and b.get('spa') is not None:
            sp, dl = b['sp'], CLV_SLOPE['spread'] * (b['sp'] - L)
            ph = phi((final_margin + sp) / SD_MARGIN)
            fh = None if fr.get('home_cover') is None else clamp(fr['home_cover'] + dl, 0.02, 0.98)
            jh = None if jd.get('cover_home') is None else clamp(jd['cover_home'] + dl, 0.02, 0.98)
            cands.append(('spread', 'home', '%s %+g' % (home, sp), sp, b['sph'], ph, fh, jh, bk))
            cands.append(('spread', 'away', '%s %+g' % (away, -sp), -sp, b['spa'], 1 - ph, inv(fh), inv(jh), bk))
        T = m.get('total')
        if T is not None and final_total is not None and b.get('t') is not None and b['t'] != T and abs(b['t'] - T) <= 3.5 and b.get('o') is not None and b.get('u') is not None:
            t_, dl = b['t'], CLV_SLOPE['total'] * (b['t'] - T)
            po = 1 - phi((t_ - final_total) / SD_TOTAL)
            fo = None if fr.get('over') is None else clamp(fr['over'] - dl, 0.02, 0.98)
            jo = None if jd.get('over') is None else clamp(jd['over'] - dl, 0.02, 0.98)
            cands.append(('total', 'over', 'Over %g' % t_, t_, b['o'], po, fo, jo, bk))
            cands.append(('total', 'under', 'Under %g' % t_, t_, b['u'], 1 - po, inv(fo), inv(jo), bk))
    cs = []
    for c in cands:
        if c[4] is None:
            continue
        pc = cal_p(c[5], c[4], 'game', c[6])
        pj, jpu = judg_mix(pc, c[7], c[6], jd.get('scale', 1.0))
        cs.append((c[0], c[1], c[2], c[3], c[4], pj, c[5], c[6], jpu, pc, c[8]))
    scored = sorted(((c[5] - impl(c[4]), c) for c in cs), key=lambda x: -x[0])
    seen_type = set()
    for edge, c in scored:
        if edge < 0.02 or len(picks) >= 2 or c[0] in seen_type:
            continue
        seen_type.add(c[0])
        conf = game_conf(edge, min_n, inj_ok, uncertain_key, adj_abs, True)
        flag = ''
        if edge > 0.08:
            conf, flag = 'Low', 'Unusually large edge for an NBA market. Check injuries, the line and the data before trusting it.'
        units = kelly_units(c[5], c[4])
        if jd.get('concern'):
            if conf == 'High':
                conf = 'Medium'
            units = max(0.5, units - 0.5)
            flag = ((flag + ' ') if flag else '') + ('Claude flagged a concern: %s' % (jd.get('why') or 'see the written analysis'))[:240]
        elif c[8] is not None and c[7] is not None and c[8] - c[7] <= -0.03:
            conf = 'Medium' if conf == 'High' else conf
            flag = flag or 'Claude\'s own read of this side is below the market\'s fair price, so the confidence is held back.'
        picks.append({'type': c[0], 'side': c[1], 'label': c[2], 'line': c[3], 'odds': c[4], 'est': r3(c[5]), 'est_raw': r3(c[6]), 'implied': r3(impl(c[4])), 'edge': r3(edge),
                      'kind': 'priced', 'conf': conf, 'units': units, 'flag': flag, 'fair': r3(c[7]), 'jp': r3(c[8]), 'est_nj': r3(c[9]),
                      'book': c[10][5:] if str(c[10]).startswith('book:') else shop.get(c[10]), 'n_books': m.get('n_books'),
                      'alt': True if str(c[10]).startswith('book:') else None})
    return picks


def prop_pick(row, mult, injured_flag, jp_over=None, concern=False, why=''):
    """Choose the better side of a prop row from the model numbers; returns a pick dict or None."""
    pm = row['model']
    fo = row.get('fair_over')
    options = []
    for side, p, price in (('over', pm['p_over'], row['over']), ('under', pm['p_under'], row['under'])):
        kind = 'priced' if price is not None else 'assumed'
        pr = price if price is not None else -110
        fair = None if fo is None else (fo if side == 'over' else 1 - fo)
        pc = cal_p(p, pr, 'prop', fair, pm.get('n_eff', pm['n']))
        jp = None if jp_over is None else (jp_over if side == 'over' else 1 - jp_over)
        pj, jpu = judg_mix(pc, jp, fair)
        options.append((pj - impl(pr), side, pj, pr, kind, p, pc, jpu, fair))
    edge, side, p, pr, kind, p_raw, pc, jpu, fair = max(options, key=lambda o: o[0])
    if edge < 0.02:
        return None
    n, minavg = pm.get('n_eff', pm['n']), pm['min_avg'] or 0
    same_hit = (pm['hit_l10'] / pm['n_l10'] >= 0.6) if side == 'over' else (pm['hit_l10'] / pm['n_l10'] <= 0.4) if pm['n_l10'] else False
    if n >= 12 and minavg >= 18 and edge >= 0.04 and same_hit and not injured_flag and kind == 'priced':
        conf = 'High'
    elif n >= 8 and edge >= 0.025 and not injured_flag:
        conf = 'Medium'
    else:
        conf = 'Low'
    if (abs(mult - 1.0) > 0.05 or abs(row.get('rm', 1.0) - 1.0) > 0.08) and conf == 'High':
        conf = 'Medium'
    if n < 8:
        conf = 'Low'
    flag = ''
    if edge > 0.12:
        conf, flag = 'Low', 'Unusually large edge. Check the line, the player\'s role and injury news before trusting it.'
    units = kelly_units(p, pr)
    if concern:
        conf = 'Medium' if conf == 'High' else conf
        units = max(0.5, units - 0.5)
        flag = ((flag + ' ') if flag else '') + ('Claude flagged a concern: %s' % (why or 'see the written analysis'))[:240]
    elif jpu is not None and fair is not None and jpu - fair <= -0.03:
        conf = 'Medium' if conf == 'High' else conf
        flag = flag or 'Claude\'s own read of this side is below the market\'s fair price, so the confidence is held back.'
    return {'type': 'prop', 'side': side, 'label': '%s %s %g %s' % (row['player'], side, row['line'], STAT_NAMES.get(row['stat'], row['stat']).lower()), 'line': row['line'], 'odds': pr,
            'est': r3(p), 'est_raw': r3(p_raw), 'implied': r3(impl(pr)), 'edge': r3(edge), 'kind': kind, 'conf': conf, 'units': units, 'player': row['player'], 'stat': row['stat'],
            'pid': row['pid'], 'mean': pm['mean'], 'hit_l10': '%d/%d' % (pm['hit_l10'], pm['n_l10']), 'flag': flag, 'fair': r3(fair), 'jp': r3(jpu), 'est_nj': r3(pc),
            'book': row.get('over_book' if side == 'over' else 'under_book'), 'n_books': row.get('n_books'), 'sd': pm.get('sd')}


def rank_key(p):
    tier = {'priced': 2, 'assumed': 1}.get(p['kind'], 0)
    return (tier, (p['edge'] or 0) * conf_weight(p['conf']))


# ----------------------------------------------------------------------------- grading
def grade_history(history, games, today=None):
    changed = 0
    for h in history:
        if h.get('result'):
            continue
        g = games.get(h['game_id'])
        if not g or g['home'].get('pts') is None:
            try:
                old = today is not None and (today - dt.date.fromisoformat(h['date'])).days > 4
            except (ValueError, KeyError, TypeError):
                old = False
            if old:            # postponed or never found: void it so it does not sit open forever
                h['result'], h['profit'], h['final'] = 'V', 0.0, 'No result found (postponed or removed)'
                changed += 1
            continue
        res = None
        if h['type'] == 'prop':
            row = next((p for p in g.get('players') or [] if p['id'] == h.get('pid')), None)
            if not row:
                res = 'V'
            else:
                v = STAT_FN[h['stat']](row)
                h['actual'] = v
                if v == h['line']:
                    res = 'P'
                else:
                    res = 'W' if (v > h['line']) == (h['side'] == 'over') else 'L'
        else:
            hp, ap = g['home']['pts'], g['away']['pts']
            if h['type'] == 'moneyline':
                d = (hp - ap) if h['side'] == 'home' else (ap - hp)
                res = 'W' if d > 0 else 'L' if d < 0 else 'P'
            elif h['type'] == 'spread':
                d = ((hp - ap) if h['side'] == 'home' else (ap - hp)) + h['line']
                res = 'W' if d > 0 else 'L' if d < 0 else 'P'
            elif h['type'] == 'total':
                t = hp + ap
                res = 'P' if t == h['line'] else ('W' if (t > h['line']) == (h['side'] == 'over') else 'L')
        if res:
            h['result'] = res
            o = h.get('odds') or -110
            h['profit'] = round(h['units'] * (o / 100.0 if o > 0 else 100.0 / -o), 3) if res == 'W' else (-h['units'] if res == 'L' else 0.0)
            h['final'] = '%s %d - %s %d' % (g['away']['ab'], g['away']['pts'], g['home']['ab'], g['home']['pts'])
            changed += 1
    return changed


# ----------------------------------------------------------------------------- auditor
# The auditor runs right after grading. It measures how every part of the bot has actually done on graded picks
# (each market, each confidence tier, each smart signal, each edge size), rates each one with strict sample-size rules,
# asks Claude for a few short lessons, and hands the result to the picks step so the bot learns from its own record.
# Ratings never touch a probability. The only thing they can change in code is a market rated TRUST_LESS on 50+
# graded picks: its picks drop one confidence tier and risk half the units until the record recovers.
AUDIT_MIN = 20          # graded picks before a row can be rated at all
AUDIT_FULL = 50         # graded picks before a rating can change a pick in code
AUDIT_EARLY = 100       # total graded picks before the auditor stops calling every rating tentative
AUDIT = {'ratings': {}, 'text': ''}

AUDIT_SYS = """You are the EpLocks AI Auditor. You never make picks. You read the graded record of past EpLocks picks and write short, specific lessons that the EpLocks picks analyst reads before analyzing today's games.

RULES
- Use only the numbers given. Never invent a result, a record or a stat.
- Sample sizes decide everything. A row marked NOT_ENOUGH_DATA can never be the basis of a lesson. Under 100 total graded picks, call every pattern tentative.
- Win rate is judged against break-even at the prices taken (shown per row), not against 50%.
- Closing-line value (CLV) is the best early sign of real skill. A group that loses but beats the close is likely unlucky; one that wins but loses to the close is likely lucky.
- One bad or good night is noise. Keep yesterday's lessons unless the record clearly changed.
- The analyst can only do two things with a lesson: scale a player's expected output (0.90 to 1.10) or skip a prop. Write lessons it can act on through those, e.g. "Rebound overs are 14-27 on 50+ picks: lean skip unless the role change is in the data."
- Separate bad process (the read was wrong) from bad luck (blowout, foul trouble, early injury) when the box score shows it.
- Plain language, no hype, at most 5 lessons, each under 30 words.

Reply with one JSON object only:
{"summary": "two sentences on how the bot is doing", "lessons": ["up to 5 short lessons"], "yesterday": "one or two sentences on the last graded day"}"""


def audit_market(h):
    return ('prop:' + h.get('stat', '?')) if h.get('type') == 'prop' else h.get('type', '?')


def audit_edge_bucket(e):
    e = e or 0
    return '2-4%' if e < 0.04 else '4-8%' if e < 0.08 else '8-12%' if e < 0.12 else '12%+'


def audit_rate(g):
    """TRUST_MORE / NEUTRAL / TRUST_LESS / NOT_ENOUGH_DATA from wins against break-even at the prices taken."""
    n = g['w'] + g['l']
    if n < AUDIT_MIN:
        return 'NOT_ENOUGH_DATA'
    diff = g['w'] / float(n) - g['be'] / n
    z = diff / math.sqrt(max(0.25 / n, 1e-9))
    need = 0.08 if n < AUDIT_FULL else 0.04
    if abs(diff) >= need and abs(z) >= 1.64:
        if diff < 0 and g.get('clv_n', 0) >= AUDIT_MIN and g['clv_sum'] / g['clv_n'] >= 0.01:
            return 'NEUTRAL'     # losing but beating the closing line: good reads, bad luck so far
        return 'TRUST_MORE' if diff > 0 else 'TRUST_LESS'
    return 'NEUTRAL'


def audit_groups(history):
    graded = [h for h in history if h.get('result') in ('W', 'L', 'P') and h.get('kind', 'priced') == 'priced' and h.get('odds') is not None]
    groups = {}

    def add(kind, key, h):
        g = groups.setdefault((kind, key), {'kind': kind, 'key': key, 'w': 0, 'l': 0, 'p': 0, 'net': 0.0, 'risk': 0.0, 'est': 0.0, 'be': 0.0, 'clv_sum': 0.0, 'clv_n': 0})
        r = h['result']
        g['w' if r == 'W' else 'l' if r == 'L' else 'p'] += 1
        g['net'] += h.get('profit') or 0.0
        g['risk'] += h.get('units') or 0.0
        if r in ('W', 'L'):
            g['est'] += h.get('est') or 0.0
            g['be'] += impl(h['odds'])
        if h.get('clv') is not None:
            g['clv_sum'] += h['clv']
            g['clv_n'] += 1

    for h in graded:
        add('overall', 'all picks', h)
        add('market', audit_market(h), h)
        add('confidence', h.get('conf') or '?', h)
        add('edge', audit_edge_bucket(h.get('edge')), h)
        for s in h.get('sig') or []:
            add('signal', s, h)
        if not h.get('sig'):
            add('signal', 'no signal', h)
    out = []
    for g in groups.values():
        n = g['w'] + g['l']
        g['n'] = n
        g['win'] = r3(g['w'] / float(n)) if n else None
        g['breakeven'] = r3(g['be'] / n) if n else None
        g['model'] = r3(g['est'] / n) if n else None
        g['roi'] = r3(g['net'] / g['risk']) if g['risk'] else None
        g['net'] = round(g['net'], 2)
        g['rating'] = audit_rate(g)
        g['clv'] = r3(g['clv_sum'] / g['clv_n']) if g['clv_n'] else None
        for k in ('est', 'be', 'risk', 'clv_sum', 'clv_n'):
            g.pop(k, None)
        out.append(g)
    order = {'overall': 0, 'market': 1, 'confidence': 2, 'signal': 3, 'edge': 4}
    out.sort(key=lambda g: (order.get(g['kind'], 9), -g['n']))
    return out, len(graded)


def audit_last_day(history):
    days = sorted({h['date'] for h in history if h.get('result') in ('W', 'L', 'P')})
    if not days:
        return None
    d = days[-1]
    rows = [h for h in history if h['date'] == d and h.get('result') in ('W', 'L', 'P')]
    w, l, p = (sum(1 for h in rows if h['result'] == x) for x in ('W', 'L', 'P'))
    return {'date': d, 'record': '%d-%d-%d' % (w, l, p), 'units': round(sum(h.get('profit') or 0.0 for h in rows), 2), 'rows': rows}


def audit_fmt(groups, n_total, last, prev_lessons):
    L = ['GRADED PICKS SO FAR: %d%s' % (n_total, ' (early sample: every pattern is tentative)' if n_total < AUDIT_EARLY else ''),
         'Columns: group | W-L-P | win% | break-even% at the prices taken | model said% | units | ROI | avg closing-line value | rating']
    for g in groups:
        L.append('%s %s | %d-%d-%d | %s | %s | %s | %+.2f | %s | %s | %s' % (
            g['kind'], g['key'], g['w'], g['l'], g['p'], pct_s(g['win']), pct_s(g['breakeven']), pct_s(g['model']), g['net'],
            pct_s(g['roi']), 'n/a' if g['clv'] is None else '%+.1f%%' % (100 * g['clv']), g['rating']))
    if last:
        L.append('')
        L.append('LAST GRADED DAY %s: %s, %+.2f units' % (last['date'], last['record'], last['units']))
        for h in sorted(last['rows'], key=lambda h: h.get('profit') or 0.0)[:40]:
            act = (' | actual %s' % h['actual']) if h.get('actual') is not None else ''
            L.append('%s | %s %+d | model %s | %s | %s%s | signals: %s' % (
                h['result'], h['label'], h.get('odds') or -110, pct_s(h.get('est')), h.get('conf'), h.get('final', ''), act, ', '.join(h.get('sig') or []) or 'none'))
    if prev_lessons:
        L.append('')
        L.append('YESTERDAY\'S LESSONS:')
        L.extend('- ' + s for s in prev_lessons)
    return '\n'.join(L)


def pct_s(x):
    return 'n/a' if x is None else '%.1f%%' % (100 * x)


def run_audit(history, today, prev, parlays=None):
    """Builds the audit, asks Claude for lessons when the record changed, and loads AUDIT for the picks step."""
    groups, n_total = audit_groups(history)
    last = audit_last_day(history)
    prev_lessons = (prev or {}).get('lessons') or []
    audit = {'date': today.isoformat(), 'graded': n_total, 'early': n_total < AUDIT_EARLY, 'groups': groups,
             'last_day': {k: v for k, v in last.items() if k != 'rows'} if last else None,
             'summary': '', 'lessons': [], 'yesterday': '', 'by': 'code'}
    if n_total < 1:
        audit['summary'] = 'No graded picks yet. The auditor starts learning after the first games finish and are graded.'
    else:
        text = audit_fmt(groups, n_total, last, prev_lessons)
        pl_ = audit_parlay_lines(parlays or [])
        if pl_:
            text += '\n\n' + '\n'.join(pl_)
        h = hashlib.sha1((AUDIT_SYS + text).encode('utf8')).hexdigest()
        if prev and prev.get('h') == h and prev.get('by') == 'claude':
            audit.update({k: prev[k] for k in ('summary', 'lessons', 'yesterday', 'by')})
            audit['h'] = h
        else:
            js = extract_json(call_claude(text + '\n\nReturn the JSON object now.', AUDIT_SYS, max_tokens=1200)) if API_KEY else None
            if isinstance(js, dict) and isinstance(js.get('lessons'), list):
                audit.update({'summary': str(js.get('summary') or '')[:400], 'lessons': [str(s)[:240] for s in js['lessons']][:5],
                              'yesterday': str(js.get('yesterday') or '')[:300], 'by': 'claude', 'h': h})
            else:
                ov = next((g for g in groups if g['kind'] == 'overall'), None)
                audit['summary'] = 'Graded so far: %d-%d-%d, %+.2f units.' % (ov['w'], ov['l'], ov['p'], ov['net']) if ov else ''
                audit['lessons'] = prev_lessons
        if n_total < AUDIT_EARLY:
            early = 'Early sample (%d graded picks). Treat every rating as tentative.' % n_total
            audit['lessons'] = [early] + [s for s in audit['lessons'] if 'early sample' not in s.lower()][:4]
    AUDIT['ratings'] = {(g['kind'], g['key']): g for g in groups}
    AUDIT['text'] = audit_prompt_text(audit)
    return audit


def audit_prompt_text(audit):
    """The short block added to every game packet so the analyst sees its own record."""
    if not audit.get('graded'):
        return ''
    L = ['TRACK RECORD (from the EpLocks auditor, %d graded picks%s):' % (audit['graded'], ', early sample' if audit.get('early') else '')]
    if audit.get('summary'):
        L.append(audit['summary'])
    for s in audit.get('lessons') or []:
        L.append('- ' + s)
    rated = [g for g in audit['groups'] if g['rating'] in ('TRUST_MORE', 'TRUST_LESS') and g['kind'] in ('market', 'signal')]
    for g in rated[:10]:
        L.append('%s %s: %d-%d, win %s vs break-even %s -> %s' % (g['kind'], g['key'], g['w'], g['l'], pct_s(g['win']), pct_s(g['breakeven']), g['rating']))
    return '\n'.join(L)


def audit_adjust(p):
    """A market rated TRUST_LESS on 50+ graded picks: drop one confidence tier and halve the stake."""
    g = AUDIT['ratings'].get(('market', audit_market(p)))
    if not g or g['rating'] != 'TRUST_LESS' or g['n'] < AUDIT_FULL:
        return
    p['conf'] = {'High': 'Medium', 'Medium': 'Low'}.get(p['conf'], 'Low')
    p['units'] = max(0.5, round(p['units'] / 2.0 * 2) / 2)
    msg = 'Auditor: this market is %d-%d over %d graded picks (win %s vs break-even %s), so confidence and stake are cut.' % (
        g['w'], g['l'], g['n'], pct_s(g['win']), pct_s(g['breakeven']))
    p['flag'] = (p.get('flag') + ' ' if p.get('flag') else '') + msg


# ----------------------------------------------------------------------------- the daily card
UNIT_CAP = {'High': 2.0, 'Medium': 1.0, 'Low': 0.5}            # the most a pick can risk at each confidence level
MAX_DAILY_PICKS = int(os.environ.get('EPLOCKS_MAX_PICKS') or 25)
MAX_PROPS_PER_GAME = int(os.environ.get('EPLOCKS_MAX_PROPS_PER_GAME') or 4)
MAX_PER_PLAYER = 1


def finalize_picks(all_picks, out_games):
    """Turns every bet that cleared the edge bar into the day's card: holds back edges too big to be real, sizes stakes
    by confidence, keeps the best few per game and per player, and caps the day. Returns the card, best first."""
    held = [p for p in all_picks if str(p.get('flag') or '').startswith('Unusually large edge')]
    pool = [p for p in all_picks if p not in held]
    for p in pool:
        p['units'] = min(p['units'], UNIT_CAP.get(p['conf'], 0.5))
    pool.sort(key=lambda p: rank_key(p), reverse=True)
    card, per_game, per_player = [], {}, {}
    for p in pool:
        if len(card) >= MAX_DAILY_PICKS:
            break
        if p['type'] == 'prop':
            if per_game.get(p['game_id'], 0) >= MAX_PROPS_PER_GAME or per_player.get(p.get('pid'), 0) >= MAX_PER_PLAYER:
                continue
            per_game[p['game_id']] = per_game.get(p['game_id'], 0) + 1
            per_player[p.get('pid')] = per_player.get(p.get('pid'), 0) + 1
        card.append(p)
    keep = {id(p) for p in card}
    for g in out_games:
        g['pick_objs'] = [p for p in g.get('pick_objs') or [] if id(p) in keep]
        g['picks'] = [p['id'] for p in g['pick_objs']]
    cut = len(pool) - len(card)
    if held or cut:
        note('Card: %d picks posted%s%s' % (len(card), '; %d held back because the edge looked too big to be real (usually a stale line or a role change)' % len(held) if held else '',
                                           '; %d more cleared the bar but missed the daily cap (%d a day, %d props a game, 1 per player)' % (cut, MAX_DAILY_PICKS, MAX_PROPS_PER_GAME) if cut else ''))
    return card


# ----------------------------------------------------------------------------- who is playing, and the schedule (sides)
# Team ratings come from the games a team has played with its usual players. When one of them sits, the rating alone does not know.
# Each player gets a value: his production per minute (Hollinger game score) above a bench-level replacement, times his usual minutes.
# A team's "missing value" tonight is the value of the players it usually has but will not have (Out, Doubtful, or partly Questionable).
# A player who has already been out for most of the last ten games is already in the rating, so he counts for little or nothing.
# Schedule spots (rest, three games in four nights, time zones crossed, long road trips, altitude) are measured the same way.
# How many points each of these is worth is learned from last season by the replay (python eplocks.py --replay), not assumed:
# until then only the player value is used, at a cautious starting size, and the schedule spots are shown but count for nothing.
TEAM_TZ = {'ATL': 0, 'BOS': 0, 'BKN': 0, 'CHA': 0, 'CLE': 0, 'DET': 0, 'IND': 0, 'MIA': 0, 'NYK': 0, 'ORL': 0, 'PHI': 0, 'TOR': 0, 'WAS': 0,
           'CHI': -1, 'DAL': -1, 'HOU': -1, 'MEM': -1, 'MIL': -1, 'MIN': -1, 'NOP': -1, 'OKC': -1, 'SAS': -1,
           'DEN': -2, 'PHX': -2, 'UTA': -2, 'GSW': -3, 'LAC': -3, 'LAL': -3, 'POR': -3, 'SAC': -3}
ALTITUDE = ('DEN', 'UTA')
SIDE_FEATS = ('impact', 'rest', 'b34', 'tz', 'trip', 'alt')
SIDE_TFEATS = ('impact', 'b34', 'alt')
SIDE_PRIOR = {'impact': 0.30, 'rest': 0.0, 'b34': 0.0, 'tz': 0.0, 'trip': 0.0, 'alt': 0.0}
SIDE_TPRIOR = {'impact': 0.0, 'b34': 0.0, 'alt': 0.0}
SIDE = {'coef': dict(SIDE_PRIOR), 'tcoef': dict(SIDE_TPRIOR), 'fitted': False, 'msg': ''}
SIDE_CAP = {'margin': 8.0, 'total': 6.0}       # the most these adjustments can ever move a projection, in points
SIDE_LABEL = {'impact': 'Who is playing', 'rest': 'Rest', 'b34': 'Third game in four nights', 'tz': 'Time zones crossed', 'trip': 'Long road trip', 'alt': 'Altitude'}
RATE_K = 200.0          # minutes of prior each player's per-minute value starts with


def gms(p):
    """Hollinger game score for one box-score line; a close stand-in when shooting splits are missing (older stored games)."""
    pts, reb, ast = p.get('pts') or 0, p.get('reb') or 0, p.get('ast') or 0
    stl, blk, tov = p.get('stl') or 0, p.get('blk') or 0, p.get('tov') or 0
    if p.get('fga') is not None:
        orb = p.get('orb') or 0
        return (pts + 0.4 * (p.get('fgm') or 0) - 0.7 * p['fga'] - 0.4 * ((p.get('fta') or 0) - (p.get('ftm') or 0)) + 0.7 * orb + 0.3 * max(0, reb - orb)
                + stl + 0.7 * ast + 0.7 * blk - 0.4 * (p.get('pf') or 0) - tov)
    return 0.55 * pts + 0.35 * reb + 0.7 * ast + stl + 0.7 * blk - tov


class SideTracker:
    """Walks finished games in date order. Before any game it can say what each team is missing and what spot it is in."""

    def __init__(self, prior_rates=None, repl=0.30):
        self.pl, self.team, self.prior, self.repl = {}, {}, prior_rates or {}, repl

    def add(self, g):
        host = g['home']['ab']
        for s in ('home', 'away'):
            t = g[s]['ab']
            played = {p['id']: p['min'] for p in g.get('players') or [] if p.get('t') == t and (p.get('min') or 0) > 0}
            self.team.setdefault(t, []).append((g['date'], g['id'], host, played))
        for p in g.get('players') or []:
            if (p.get('min') or 0) > 0:
                s_ = self.pl.setdefault(p['id'], [0.0, 0.0])
                s_[0] += gms(p)
                s_[1] += p['min']

    def rate(self, pid):
        gs, mn = self.pl.get(pid, (0.0, 0.0))
        return (gs + RATE_K * self.prior.get(pid, self.repl + 0.05)) / (mn + RATE_K)

    def missing(self, t, avail, listed=lambda pid: False):
        """avail(pid) -> chance he plays tonight (0 to 1). Returns (missing value, [(pid, value lost, usual share, tonight)])."""
        hist = self.team.get(t, [])[-10:]
        if len(hist) < 3:
            return 0.0, []
        mins, last3 = {}, set()
        for i, (_, _, _, played) in enumerate(hist):
            for pid, mn in played.items():
                mins.setdefault(pid, []).append(mn)
                if i >= len(hist) - 3:
                    last3.add(pid)
        M, items = 0.0, []
        for pid, ms in mins.items():
            exp_min = mean(ms[-5:])
            if exp_min < 10:
                continue
            usual = (len(ms) + 2.0) / (len(hist) + 2.0)   # shrunk toward 'he plays': a few missed games say little, and early ratings lean on last season
            if pid not in last3 and not listed(pid):
                continue                   # gone from the rotation lately and not on the report: maybe traded or waived, so no guess
            a = avail(pid)
            d = (usual - a) * clamp((self.rate(pid) - self.repl) * exp_min, -3.0, 15.0)
            M += d
            if abs(d) >= 0.5:
                items.append((pid, d, usual, a))
        items.sort(key=lambda x: -abs(x[1]))
        return M, items

    def spot(self, t, date, host):
        prev = [h for h in self.team.get(t, []) if h[0] < date]
        d = dt.date.fromisoformat(date)
        rest = 3 if not prev else min(3, (d - dt.date.fromisoformat(prev[-1][0])).days - 1)
        b34 = 1 if sum(1 for h in prev[-3:] if (d - dt.date.fromisoformat(h[0])).days <= 3) >= 2 else 0
        tz = abs(TEAM_TZ.get(host, 0) - TEAM_TZ.get(prev[-1][2], 0)) if prev and rest <= 1 else 0
        trip = 0
        if host != t:
            trip = 1
            for h in reversed(prev):
                if h[2] == t:
                    break
                trip += 1
        return {'rest': rest, 'b34': b34, 'tz': tz, 'trip': trip}


def side_features(tr, home, away, date, avail_h=None, avail_a=None, listed_h=None, listed_a=None):
    """Features from the home team's side (positive favors home) and for the total. Without avail functions, who played is read from the box (fitting)."""
    mh, ih = tr.missing(home, avail_h or (lambda pid: 1.0), listed_h or (lambda pid: False))
    ma, ia = tr.missing(away, avail_a or (lambda pid: 1.0), listed_a or (lambda pid: False))
    sh, sa = tr.spot(home, date, home), tr.spot(away, date, home)
    x = {'impact': ma - mh, 'rest': sh['rest'] - sa['rest'], 'b34': sa['b34'] - sh['b34'], 'tz': sa['tz'] - sh['tz'],
         'trip': clamp(sa['trip'] - 2, 0, 4), 'alt': 1.0 if home in ALTITUDE else 0.0}
    xt = {'impact': mh + ma, 'b34': sh['b34'] + sa['b34'], 'alt': x['alt']}
    return x, xt, {'home': {'missing': mh, 'items': ih, 'spot': sh}, 'away': {'missing': ma, 'items': ia, 'spot': sa}}


def side_adjust(x, xt):
    am = sum(SIDE['coef'].get(k, 0.0) * x[k] for k in SIDE_FEATS)
    at = sum(SIDE['tcoef'].get(k, 0.0) * xt[k] for k in SIDE_TFEATS)
    return clamp(am, -SIDE_CAP['margin'], SIDE_CAP['margin']), clamp(at, -SIDE_CAP['total'], SIDE_CAP['total'])


def side_explain(x, det, home, away, names):
    """Plain-language lines for the site and for Claude's packet."""
    L = []
    for t, s in ((away, 'away'), (home, 'home')):
        for pid, d, usual, a in det[s]['items'][:4]:
            nm = names.get(pid, pid)
            if d > 0:
                L.append('%s without %s (plays %d%% of games, %s tonight): %s %.1f points of value missing' % (t, nm, round(100 * usual), 'out' if a <= 0.05 else '%d%% to play' % round(100 * a), t, d))
            else:
                L.append('%s has %s back (played %d%% of recent games): %s %+.1f points of value' % (t, nm, round(100 * usual), t, -d))
    for k in SIDE_FEATS[1:]:
        c = SIDE['coef'].get(k, 0.0)
        if x[k] and (abs(c * x[k]) >= 0.3 or not SIDE['fitted']):
            eff = c * x[k]
            desc = {'rest': 'rest days %s %d, %s %d' % (home, det['home']['spot']['rest'], away, det['away']['spot']['rest']),
                    'b34': '%s on the third game in four nights' % (away if x[k] > 0 else home),
                    'tz': '%s crossed %d time zone%s since its last game' % (away if x[k] > 0 else home, abs(det['away' if x[k] > 0 else 'home']['spot']['tz']), '' if abs(x[k]) == 1 else 's'),
                    'trip': '%s is %d games into a road trip' % (away, det['away']['spot']['trip']),
                    'alt': 'game at altitude in %s' % home}[k]
            L.append('%s: %s (%s)' % (SIDE_LABEL[k], desc, ('%+.1f to %s' % (abs(eff), home if eff > 0 else away)) if SIDE['fitted'] and abs(eff) >= 0.05 else 'not proven yet, counts for nothing'))
    return L


def prior_rates_from(box=None, prior_state=None):
    """Last season's per-minute values: from the replay's box scores when loaded (full shooting splits), else from the season summary."""
    rates, bench = {}, []
    if box and box.get('games'):
        acc = {}
        for g in box['games'].values():
            for r in g['p']:
                if r[2] <= 0:
                    continue
                p = row_to_player(r)
                a = acc.setdefault(r[0], [0.0, 0.0, 0])
                a[0] += gms(p)
                a[1] += r[2]
                a[2] += 1
        for pid, (gs, mn, n) in acc.items():
            if mn >= 150:
                rates[pid] = gs / mn
            if n >= 15 and 8 <= mn / n <= 18:
                bench.append(gs / mn)
    elif prior_state and prior_state.get('players'):
        for pid, p in prior_state['players'].items():
            if p.get('min', 0) >= 150:
                rates[pid] = gms(p) / p['min']
                if p.get('n', 0) >= 15 and 8 <= p['min'] / p['n'] <= 18:
                    bench.append(gms(p) / p['min'])
    bench.sort()
    repl = bench[len(bench) // 2] if len(bench) >= 30 else 0.30
    return rates, repl


ROW_KEYS = ('pts', 'reb', 'ast', 'tpm', 'stl', 'blk', 'tov', 'fgm', 'fga', 'ftm', 'fta', 'orb', 'pf', 'pm')


def row_to_player(r):
    p = dict(zip(ROW_KEYS, r[3:3 + len(ROW_KEYS)]), id=r[0], t=r[1], min=r[2])
    if len(r) < 3 + len(ROW_KEYS):
        p['fga'] = None
    return p


def solve(A, b):
    """Gaussian elimination for the small ridge systems below."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def ridge(rows, feats, prior, strength=0.15):
    """Least squares of residual on features, pulled toward the starting sizes. strength: share of the data's own weight the prior gets."""
    k = len(feats)
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for x, r in rows:
        v = [x[f] for f in feats]
        for i in range(k):
            b[i] += v[i] * r
            for j in range(k):
                A[i][j] += v[i] * v[j]
    for i, f in enumerate(feats):
        lam = strength * A[i][i] + 1.0
        A[i][i] += lam
        b[i] += lam * prior[f]
    sol = solve(A, b)
    if not sol:
        return dict(prior)
    out = dict(zip(feats, sol))
    n = len(rows)
    if n > len(feats) + 10:
        # a size only counts when it clearly stands out from the noise of one season (about 2 standard errors); otherwise it is 0
        res = [r - sum(out[f] * x[f] for f in feats) for x, r in rows]
        sig = math.sqrt(sum(e * e for e in res) / (n - len(feats)))
        for i, f in enumerate(feats):
            sxx = sum(x[f] ** 2 for x, r in rows)
            if sxx <= 0 or abs(out[f]) < 2.0 * sig / math.sqrt(sxx):
                out[f] = 0.0
    return out


def side_rows(games, repl):
    """One season walked in date order: ratings built only from earlier games, features before each game, and how far off the plain projection was."""
    tr = SideTracker({}, repl)          # no head start: the season being replayed is the only evidence
    by_day = {}
    for g in games:
        by_day.setdefault(g['date'], []).append(g)
    seen, rows = {}, []
    for d in sorted(by_day):
        if len(seen) >= 150:
            ratings, lo, lp = build_ratings(seen)
            today_ = dt.date.fromisoformat(d)
            for g in by_day[d]:
                h, a = g['home']['ab'], g['away']['ab']
                if min((ratings.get(h) or {}).get('n', 0), (ratings.get(a) or {}).get('n', 0)) < 10:
                    continue
                mo = project(h, a, ratings, lo, lp, today_)
                if not mo:
                    continue
                played = {p['id'] for p in g['players'] if (p.get('min') or 0) > 0}
                av = lambda pid: 1.0 if pid in played else 0.0  # noqa
                lst = lambda pid: pid in played  # noqa
                x, xt, _ = side_features(tr, h, a, d, av, av, lst, lst)
                rows.append((d, x, xt, g['home']['pts'] - g['away']['pts'] - mo['margin'], g['home']['pts'] + g['away']['pts'] - mo['total']))
        for g in by_day[d]:
            seen[g['id']] = g
            tr.add(g)
    return rows


def run_side_fit(boxes):
    """Replays every stored season with ratings built only from earlier games and learns what who-is-playing and each schedule spot
    is worth, in points of margin and total. The first 60% of each season fits, the rest checks, then all of it gives the final sizes."""
    rows, test = [], []
    for box in boxes:
        games = box_games(box)
        if len(games) < 500:
            continue
        rs = side_rows(games, prior_rates_from(box)[1])
        k = int(len(rs) * 0.6)
        rows += [(True,) + r for r in rs[:k]] + [(False,) + r for r in rs[k:]]
    if len(rows) < 300:
        print('Side model: not enough replayed games (%d)' % len(rows), flush=True)
        return None
    fit_m = ridge([(x, r) for f, d, x, xt, r, rt in rows if f], SIDE_FEATS, SIDE_PRIOR)
    fit_t = ridge([(xt, rt) for f, d, x, xt, r, rt in rows if f], SIDE_TFEATS, SIDE_TPRIOR)
    test = [(x, xt, r, rt) for f, d, x, xt, r, rt in rows if not f]
    rows = [r[1:] for r in rows]

    def mae(fn):
        return round(mean(abs(fn(x, xt, r, rt)) for x, xt, r, rt in test), 3)
    check = {'games': len(test),
             'margin_error_before': mae(lambda x, xt, r, rt: r),
             'margin_error_after': mae(lambda x, xt, r, rt: r - clamp(sum(fit_m[k] * x[k] for k in SIDE_FEATS), -SIDE_CAP['margin'], SIDE_CAP['margin'])),
             'total_error_before': mae(lambda x, xt, r, rt: rt),
             'total_error_after': mae(lambda x, xt, r, rt: rt - clamp(sum(fit_t[k] * xt[k] for k in SIDE_TFEATS), -SIDE_CAP['total'], SIDE_CAP['total']))}
    coef = ridge([(x, r) for d, x, xt, r, rt in rows], SIDE_FEATS, SIDE_PRIOR)
    tcoef = ridge([(xt, rt) for d, x, xt, r, rt in rows], SIDE_TFEATS, SIDE_TPRIOR)
    keep_m = check['margin_error_after'] < check['margin_error_before']      # it has to actually help on unseen games, not tie
    keep_t = check['total_error_after'] < check['total_error_before']
    out = {'made': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'games': len(rows), 'seasons': len(boxes),
           'coef': {k: round(v, 3) for k, v in coef.items()} if keep_m else {k: 0.0 for k in SIDE_FEATS},
           'tcoef': {k: round(v, 3) for k, v in tcoef.items()} if keep_t else {k: 0.0 for k in SIDE_TFEATS},
           'check': check, 'used': {'margin': keep_m, 'total': keep_t}}
    jsave(os.path.join(DATA, 'side_model.json'), out)
    print('Side model: %s' % json.dumps({k: out[k] for k in ('coef', 'tcoef', 'check')}), flush=True)
    return out


def load_side_model():
    js = jload(os.path.join(DATA, 'side_model.json'), {})
    if js.get('coef'):
        SIDE['coef'] = {k: float(js['coef'].get(k, 0.0)) for k in SIDE_FEATS}
        SIDE['tcoef'] = {k: float((js.get('tcoef') or {}).get(k, 0.0)) for k in SIDE_TFEATS}
        SIDE['fitted'] = True
        ck = js.get('check') or {}
        src('side_model', True, 'who-is-playing and schedule sizes learned from %d replayed games; on games the fit had not seen, margin error %.2f -> %.2f points, total error %.2f -> %.2f' % (
            js.get('games', 0), ck.get('margin_error_before', 0), ck.get('margin_error_after', 0), ck.get('total_error_before', 0), ck.get('total_error_after', 0)))
    else:
        src('side_model', True, 'who-is-playing adjustment at a cautious starting size (%.2f points per point of player value); schedule spots shown but not counted until the replay measures them' % SIDE_PRIOR['impact'])


# ----------------------------------------------------------------------------- parlay tracker
# The website's parlay builder runs in the browser, so on its own nothing it suggests is ever recorded. Here the daily run builds the
# same tickets with the builder's default settings (balanced, any confidence, games and props, priced legs only, one leg per game),
# logs them in data/parlays.json before tip-off, and grades every leg from the real box scores, exactly like single picks.
# A ticket is rebuilt on each run until the first of its games tips off, then it is locked. Results are never edited by hand.
PARLAY_CW = {'High': 1.0, 'Medium': 0.75, 'Low': 0.4}
PARLAY_MAX = 5


def american_to_dec(o):
    return 1 + o / 100.0 if o > 0 else 1 + 100.0 / -o


def dec_to_american(d):
    return int(round((d - 1) * 100)) if d >= 2 else -int(round(100 / (d - 1)))


def parlay_ticket(legs):
    """Same math as the site's builder: combined price, chance if legs are independent, edge, and a quarter-Kelly stake on half the edge."""
    D, P, Pc = 1.0, 1.0, 1.0
    for p in legs:
        D *= american_to_dec(p['odds'])
        P *= p['est']
        Pc *= max(0.01, p['est'] - 0.5 * p['edge'])
    b = D - 1
    ev, evc = P * D - 1, Pc * D - 1
    f = evc / b if evc > 0 else 0.0
    g = Pc * math.log(1 + f * b) + (1 - Pc) * math.log(1 - f) if evc > 0 else evc
    stake = max(0.25, min(1.0, round(f * 100 * 0.25 * 4) / 4)) if f > 0 else 0.0
    conf = 'High' if all(p['conf'] == 'High' for p in legs) else 'Low' if any(p['conf'] == 'Low' for p in legs) else 'Medium'
    return {'n': len(legs), 'dec': round(D, 4), 'odds': dec_to_american(D), 'P': r3(P), 'ev': r3(ev), 'evc': r3(evc), 'g': g, 'stake': stake, 'conf': conf}


def build_parlays(picks, day, posted):
    """Today's 2 to 5 leg tickets from the day's card, the way the builder makes them, plus which one is the Ultimate Pick."""
    pool = [p for p in picks if (p.get('edge') or 0) > 0 and p.get('odds') is not None and p.get('est') is not None and p.get('kind', 'priced') == 'priced']
    pool.sort(key=lambda p: -(p['edge'] * PARLAY_CW.get(p['conf'], 0.4)))
    used, legs = set(), []
    for p in pool:
        if p['game_id'] in used:
            continue
        used.add(p['game_id'])
        legs.append(p)
    out = []
    for n in range(2, min(PARLAY_MAX, len(legs)) + 1):
        t = parlay_ticket(legs[:n])
        t['legs'] = [{k: p.get(k) for k in ('id', 'label', 'type', 'side', 'line', 'odds', 'est', 'edge', 'conf', 'game_id', 'away', 'home', 'player', 'stat', 'pid', 'utc')} for p in legs[:n]]
        out.append(t)
    if not out:
        return []
    ult = max(out, key=lambda t: (t['g'], -t['n']))
    for t in out:
        t['ult'] = t is ult
        t['id'] = '%s-P%d' % (day, t['n'])
        t['date'], t['posted'], t['v'] = day, posted, ENGINE_VERSION
        t.pop('g', None)
    return out


def grade_parlays(parlays, games, today=None):
    """Grades each leg from the box scores (same rules as single picks), then the ticket: one losing leg loses it; pushed or voided legs
    drop out and the price is recomputed from the rest; a ticket left with no live legs is void."""
    changed = 0
    for t in parlays:
        if t.get('result'):
            continue
        for lg in t['legs']:
            if lg.get('result'):
                continue
            h = dict(lg, date=t['date'], units=1.0)
            grade_history([h], games, today)
            if h.get('result'):
                lg['result'] = h['result']
                if h.get('actual') is not None:
                    lg['actual'] = h['actual']
        rs = [lg.get('result') for lg in t['legs']]
        if 'L' in rs:
            t['result'] = 'L'
        elif all(rs):
            live = [lg for lg in t['legs'] if lg['result'] == 'W']
            if not live:
                t['result'] = 'V'
            else:
                t['result'] = 'W'
                t['paid_dec'] = round(math.prod(american_to_dec(lg['odds']) for lg in live), 4)
        else:
            continue
        d = t.get('paid_dec', t['dec'])
        t['profit_flat'] = round(d - 1, 3) if t['result'] == 'W' else (-1.0 if t['result'] == 'L' else 0.0)
        t['profit'] = round(t['stake'] * (d - 1), 3) if t['result'] == 'W' else (-t['stake'] if t['result'] == 'L' else 0.0)
        t['graded'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        changed += 1
    return changed


def update_parlays(all_picks, out_games, games, today, posted):
    """Grades old tickets and (re)builds today's unless one of today's ticket games has already tipped off."""
    path = os.path.join(OUT, 'parlays.json')
    parlays = jload(path, [])
    n_graded = grade_parlays(parlays, games, today)
    day = today.isoformat()
    state = {g['id']: g.get('state') for g in out_games}
    todays = [t for t in parlays if t['date'] == day]
    locked = any(state.get(lg['game_id']) not in (None, 'pre') for t in todays for lg in t['legs'])
    if not locked:
        parlays = [t for t in parlays if t['date'] != day] + build_parlays([p for p in all_picks if state.get(p['game_id']) == 'pre'], day, posted)
    jsave(path, parlays)
    done = [t for t in parlays if t.get('result') in ('W', 'L')]
    w = sum(1 for t in done if t['result'] == 'W')
    src('parlays', True, '%d tickets logged today%s; %d graded so far (%d-%d)%s' % (
        sum(1 for t in parlays if t['date'] == day), ' (locked: a game has started)' if locked else '', len(done), w, len(done) - w,
        ', %d graded this run' % n_graded if n_graded else ''))
    return parlays


def audit_parlay_lines(parlays):
    """A few lines for the auditor's prompt: how the logged parlays have done against what the model expected."""
    done = [t for t in parlays if t.get('result') in ('W', 'L')]
    if not done:
        return []
    L = ['PARLAYS (logged daily from the builder, graded): %d-%d, flat 1u each %+.2f units; model expected a %.0f%% hit rate, actual %.0f%%' % (
        sum(1 for t in done if t['result'] == 'W'), sum(1 for t in done if t['result'] == 'L'), sum(t['profit_flat'] for t in done),
        100 * mean(t['P'] for t in done), 100.0 * sum(1 for t in done if t['result'] == 'W') / len(done))]
    for n in range(2, PARLAY_MAX + 1):
        g = [t for t in done if t['n'] == n]
        if g:
            L.append('  %d legs: %d-%d, expected hit %.0f%%, flat units %+.2f' % (n, sum(1 for t in g if t['result'] == 'W'), sum(1 for t in g if t['result'] == 'L'),
                                                                                    100 * mean(t['P'] for t in g), sum(t['profit_flat'] for t in g)))
    brk, seen = {}, set()
    for t in done:
        for lg in t['legs']:
            key = (t['date'], lg.get('id') or lg.get('label'))
            if lg.get('result') == 'L' and key not in seen:     # the bigger tickets reuse the smaller ones' legs; count each leg once
                seen.add(key)
                k = ('prop:' + str(lg.get('stat'))) if lg.get('type') == 'prop' else lg.get('type')
                brk[k] = brk.get(k, 0) + 1
    if brk:
        L.append('  legs that lost parlays most often: ' + ', '.join('%s %d' % kv for kv in sorted(brk.items(), key=lambda kv: -kv[1])[:4]))
    return L


# ----------------------------------------------------------------------------- injury news sniper and phone alerts
# Prop and side lines move most in the half hour after a rotation player is ruled out. The closing job already runs every 20 minutes
# in the evening, so each time it also reads ESPN's injury report (free, no Odds API credits). When a player who plays real minutes is
# newly ruled out (or is back) for a game that has not tipped, it reruns the full card at once and pushes any new pick to your phone.
NEWS_MIN_MINUTES = 20.0       # a player must average this many minutes for his news to matter
NEWS_WINDOW = (15, 300)       # only games tipping off between 15 minutes and 5 hours from now
NEWS_MAX_RERUNS = int(os.environ.get('EPLOCKS_NEWS_RERUNS') or 4)   # most news reruns per day (each one spends Odds API credits)
NEWS = {'reason': None}
NTFY_TOPIC = os.environ.get('NTFY_TOPIC', '').strip()
SITE_URL = os.environ.get('EPLOCKS_SITE_URL') or 'https://eplocks.github.io/eplocks/'
WATCH_PATH = os.path.join(DATA, 'injury_watch.json')


def watch_save(injuries, day, reruns=None):
    """Remembers the injury report as last seen, so the next check only reacts to what changed."""
    old = jload(WATCH_PATH, {})
    if reruns is None:
        reruns = old.get('reruns', 0) if old.get('date') == day else 0
    teams = {t: {norm_name(i['name']): i['status'] for i in lst} for t, lst in (injuries or {}).items()}
    jsave(WATCH_PATH, {'date': day, 'teams': teams, 'reruns': reruns, 'seen_at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})


def news_check(now=None):
    """Returns a short description of new injury news worth a rerun, or None."""
    now = now or dt.datetime.now(dt.timezone.utc)
    latest = jload(os.path.join(OUT, 'latest.json'), {})
    day = today_ct().isoformat()
    if latest.get('for_date') != day or not latest.get('games'):
        return None
    name_to_abbr = {g[s]['name'].lower(): g[s]['abbr'] for g in latest['games'] for s in ('home', 'away')}
    inj = fetch_injuries(name_to_abbr)
    if inj is None:
        return None
    watch = jload(WATCH_PATH, {})
    if watch.get('date') != day or not watch.get('teams'):
        watch_save(inj, day, 0)          # no baseline yet today: start one, react from the next check on
        return None
    old, items = watch['teams'], []
    for g in latest['games']:
        t = parse_utc(g.get('utc'))
        if g.get('state') != 'pre' or not t:
            continue
        mins_to = (t - now).total_seconds() / 60.0
        if not NEWS_WINDOW[0] <= mins_to <= NEWS_WINDOW[1]:
            continue
        for s in ('away', 'home'):
            team = g[s]['abbr']
            new_map = {norm_name(i['name']): i['status'] for i in inj.get(team, [])}
            old_map = old.get(team, {})
            for p in g[s].get('rotation') or []:
                if (p.get('min') or 0) < NEWS_MIN_MINUTES or (p.get('gp') or 0) < 2:
                    continue
                k = norm_name(p['name'])
                ow, nw = inj_weight(old_map.get(k, '')), inj_weight(new_map.get(k, ''))
                if nw >= 0.8 and ow < 0.8:
                    items.append('%s (%s, %.0f min) now %s' % (p['name'], team, p['min'], new_map[k]))
                elif ow >= 0.8 and nw < 0.3:
                    items.append('%s (%s, %.0f min) is back%s' % (p['name'], team, p['min'], (': ' + new_map[k]) if new_map.get(k) else ''))
    reruns = watch.get('reruns', 0)
    if items and reruns >= NEWS_MAX_RERUNS:
        print('Injury news (%s), but today\'s %d reruns are used up' % ('; '.join(items), NEWS_MAX_RERUNS), flush=True)
        items = []
    watch_save(inj, day, reruns + (1 if items else 0))
    return '; '.join(items[:6]) if items else None


def push_note(title, body, prio='default', tags='basketball'):
    """Sends a phone notification through ntfy.sh (free app). Does nothing without the NTFY_TOPIC secret."""
    if not NTFY_TOPIC:
        return False
    try:
        safe = lambda x: x.encode('latin-1', 'ignore').decode('latin-1')  # noqa
        req = urllib.request.Request('https://ntfy.sh/' + urllib.parse.quote(NTFY_TOPIC), data=body.encode('utf-8'), method='POST',
                                     headers={'Title': safe(title)[:200], 'Priority': prio, 'Tags': tags, 'Click': SITE_URL})
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
        return True
    except Exception as e:  # noqa
        note('Phone alert failed: %s' % str(e)[:120])
        return False


def alert_line(p):
    odds = '%+d' % p['odds'] if p.get('odds') is not None else 'no price'
    book = (' at %s' % p['book']) if p.get('book') else ''
    return '%s %s%s | %s %gu | edge %.1f%% | %s @ %s' % (p['label'], odds, book, p['conf'], p['units'], 100 * (p.get('edge') or 0), p['away'], p['home'])


def send_alerts(prev, result):
    """First card of the day: a summary. Later runs: only picks that were not on the card before, with the news that caused them."""
    if not NTFY_TOPIC:
        return
    picks = result.get('picks') or []
    reason = NEWS.get('reason')
    if prev.get('for_date') != result.get('for_date'):
        if not picks:
            return
        body = '\n'.join(alert_line(p) for p in picks[:5]) + ('\n+%d more on the site' % (len(picks) - 5) if len(picks) > 5 else '')
        ok = push_note("Today's EpLocks card: %d picks" % len(picks), body)
    else:
        before = {(p.get('game_id'), p.get('label')) for p in prev.get('picks') or []}
        live = {g['id'] for g in result.get('games') or [] if g.get('state') == 'pre'}
        new = [p for p in picks if (p.get('game_id'), p.get('label')) not in before and p.get('game_id') in live]
        if new:
            body = ((reason + '\n\n') if reason else '') + '\n'.join(alert_line(p) for p in new[:6])
            ok = push_note('%d new pick%s%s' % (len(new), '' if len(new) == 1 else 's', ' after injury news' if reason else ''), body, prio='high' if reason else 'default')
        elif reason:
            ok = push_note('Injury news: card rechecked', reason + '\n\nNo new bet cleared the bar.', prio='low')
        else:
            return
    src('alerts', ok, 'phone alert sent' if ok else 'phone alert failed')


# ----------------------------------------------------------------------------- prop model replay (no betting lines needed)
# Replays last season game by game. Before each game, the same prop model predicts every rotation player's range for each stat
# from his earlier games only; then the real box score shows whether the predicted chances held up ("said 60% over, happened 60%?").
# From tens of thousands of these it learns two small corrections per stat, one for the average and one for the width of the range,
# and the daily run applies them. Run it from the Actions tab (mode: replay); it resumes on its own if one run is not enough.
REPLAY_STATS = ('points', 'rebounds', 'assists', 'threes', 'pra', 'pts_reb', 'pts_ast', 'reb_ast')
REPLAY_SECONDS = float(os.environ.get('EPLOCKS_REPLAY_SECONDS') or 3000)
REPLAY_MIN_GAMES = 5          # games a player needs before he is predicted
REPLAY_MIN_MIN = 15.0         # and minutes per game over his last five
REPLAY_MIN_N = 1000           # predictions a stat needs before its correction is used
REPLAY_V = 3
REPLAY_FIX = {}               # stat -> {'b': average multiplier, 's': range multiplier}; filled from data/prop_replay.json
REPLAY_OFF = {'off': False}   # the replay itself must judge the uncorrected model
REPLAY_SUM = {}               # short summary for the website


REPLAY_SEASONS = [tuple(x.split(':')) for x in (os.environ.get('EPLOCKS_REPLAY_SEASONS') or '2024-10-22:2025-04-13,%s:%s' % (PRIOR_START, PRIOR_END)).split(',')]
TEAM_KEYS = ('pts', 'fga', 'fgm', 'tpm', 'tpa', 'fta', 'ftm', 'orb', 'drb', 'reb', 'tov', 'ast', 'pf')
PROP_CANDIDATES = [   # ways to turn a player's game log into tonight's expectation, compared on games the fit never saw
    {'name': 'current: 50/30/20 season, last 10, last 5', 'w': (0.5, 0.3, 0.2), 'permin': False},
    {'name': 'more recent: 35/35/30', 'w': (0.35, 0.35, 0.3), 'permin': False},
    {'name': 'more season: 65/25/10', 'w': (0.65, 0.25, 0.1), 'permin': False},
    {'name': 'even: 40/40/20', 'w': (0.4, 0.4, 0.2), 'permin': False},
    {'name': 'per minute x projected minutes, 50/30/20', 'w': (0.5, 0.3, 0.2), 'permin': True},
    {'name': 'per minute x projected minutes, 35/35/30', 'w': (0.35, 0.35, 0.3), 'permin': True}]
PROP_CFG = {'w': (0.5, 0.3, 0.2), 'permin': False}


def box_path(rng):
    return os.path.join(DATA, 'replay_box_%s.json' % rng[1][:4])


def replay_box_update(rng, t0):
    """Stores one season's box scores compactly (players and team totals), resuming where the previous run stopped."""
    path = box_path(rng)
    st = jload(path, None)
    if not st or st.get('range') != list(rng) or st.get('v') != REPLAY_V:
        st = {'v': REPLAY_V, 'range': list(rng), 'days_done': [], 'games': {}, 'complete': False}
    if st.get('complete'):
        return st
    d0, d1 = dt.date.fromisoformat(rng[0]), dt.date.fromisoformat(rng[1])
    done, d = set(st['days_done']), d0
    while d <= d1 and time.time() - t0 < REPLAY_SECONDS * 0.8:
        ds = d.isoformat()
        if ds not in done:
            try:
                js = http_json('%s/scoreboard?dates=%s&limit=100' % (ESPN, d.strftime('%Y%m%d')))
                evs = [parse_event(e) for e in js.get('events') or []]
                evs = [e for e in evs if 'home' in e and 'away' in e and e['completed'] and e['season_type'] in (None, 2) and e['id'] not in st['games']]
                res = fetch_summaries(evs)
                ok_day = True
                for e in evs:
                    r = res.get(e['id'])
                    if not isinstance(r, dict):
                        ok_day = False
                        continue
                    st['games'][e['id']] = {'d': r.get('date') or ds,
                                            'h': [r['home'].get('ab')] + [r['home'].get(k) for k in TEAM_KEYS],
                                            'a': [r['away'].get('ab')] + [r['away'].get(k) for k in TEAM_KEYS],
                                            'p': [[p['id'], p['t'], p.get('min') or 0] + [p.get(k) or 0 for k in ROW_KEYS] for p in r.get('players') or []]}
                if ok_day:
                    done.add(ds)
                    if len(done) % 20 == 0:        # save progress as it goes, so a run that is cut off still keeps what it loaded
                        st['days_done'] = sorted(done)
                        jsave(path, st)
            except Exception as ex:  # noqa
                note('Replay: %s could not be loaded: %s' % (ds, str(ex)[:100]))
        d += dt.timedelta(days=1)
    st['days_done'] = sorted(done)
    st['complete'] = len(done) >= (d1 - d0).days + 1
    jsave(path, st)
    return st


def box_games(box):
    """The stored season as game dicts the ratings and trackers understand, in date order."""
    out = []
    for gid, b in box['games'].items():
        if not b.get('h') or not b.get('a'):
            continue
        g = {'id': gid, 'date': b['d'], 'players': [row_to_player(r) for r in b['p']]}
        for side, k in (('home', 'h'), ('away', 'a')):
            g[side] = dict(zip(TEAM_KEYS, b[k][1:]), ab=b[k][0])
        if g['home'].get('pts') is None or g['away'].get('pts') is None:
            continue
        out.append(g)
    out.sort(key=lambda g: (g['date'], g['id']))
    return out


def replay_predictions(box, cfg=None):
    """Walks one season in order; returns {stat: [(mean, sd, actual, date), ...]} from predictions made before each game with cfg."""
    keys = ('pts', 'reb', 'ast', 'tpm', 'stl', 'blk', 'tov')
    logs, rows = {}, {s: [] for s in REPLAY_STATS}
    saved = dict(PROP_CFG)
    REPLAY_OFF['off'] = True
    if cfg:
        PROP_CFG.update({'w': tuple(cfg['w']), 'permin': cfg['permin']})
    try:
        for gid, g in sorted(box['games'].items(), key=lambda kv: (kv[1]['d'], kv[0])):
            for r in g['p']:
                pid, team, mins = r[0], r[1], r[2]
                p = dict(zip(keys, r[3:10]), t=team, min=mins)
                L = logs.setdefault(pid, {'id': pid, 'team': team, 'games': []})
                L['team'] = team
                played = [x for x in L['games'] if x['min'] > 0]
                if mins > 0 and len(played) >= REPLAY_MIN_GAMES and mean(x['min'] for x in played[-5:]) >= REPLAY_MIN_MIN:
                    for s in REPLAY_STATS:
                        m = prop_model(L, s, 0.5)
                        if m and m['sd']:
                            rows[s].append((m['mean'], m['sd'], STAT_FN[s](p), g['d']))
                L['games'].append(p)
    finally:
        REPLAY_OFF['off'] = False
        PROP_CFG.clear()
        PROP_CFG.update(saved)
    return rows


def replay_events(rs, b=1.0, s=1.0, step=1, lines_from=None):
    """Over/under questions at the half-point lines around each prediction: (model chance of over, 1 if over happened).
    lines_from: another set of predictions (same games, same order) whose means set the lines, so candidates face identical questions."""
    out = []
    ref = lines_from or rs
    for i in range(0, len(rs), step):
        mu, sd, act = rs[i][0], rs[i][1], rs[i][2]
        mu2, sd2 = mu * b, sd * s
        base = math.floor(ref[i][0]) + 0.5
        for j in (-2, -1, 0, 1, 2):
            line = base + j
            if line <= 0:
                continue
            out.append((1 - phi((line - mu2) / sd2), 1.0 if act > line else 0.0))
    return out


def logloss(ev):
    return -sum(y * math.log(max(p, 1e-6)) + (1 - y) * math.log(max(1 - p, 1e-6)) for p, y in ev) / max(len(ev), 1)


def replay_fit(rs):
    """Centre and range multipliers chosen together by the best log loss on the over/under questions.
    The centre is not the plain average: stats like points are skewed (a few big nights pull the average up), so the line he
    beats half the time sits a little below his average. Fitting on over/under outcomes finds that point directly."""
    step = max(1, len(rs) // 6000)
    best = None
    for i in range(0, 13):
        b = 0.92 + 0.01 * i
        for k in range(0, 31):
            s = 0.85 + 0.02 * k
            ll = logloss(replay_events(rs, b, s, step))
            if best is None or ll < best[0]:
                best = (ll, b, s)
    return best[1], best[2]


def replay_table(rs, b=1.0, s=1.0):
    """Reliability: model chance of over in 10-point buckets against how often the over happened."""
    bk = {}
    for p, y in replay_events(rs, b, s, max(1, len(rs) // 20000)):
        k = min(9, int(p * 10))
        c = bk.setdefault(k, [0, 0.0, 0.0])
        c[0] += 1
        c[1] += p
        c[2] += y
    return [{'bucket': '%d-%d%%' % (k * 10, k * 10 + 10), 'n': c[0], 'model': r3(c[1] / c[0]), 'actual': r3(c[2] / c[0])} for k, c in sorted(bk.items()) if c[0] >= 50]


def split_rows(rows_by_season, frac=0.6):
    """Chronological split inside each season: the first part fits, the rest checks."""
    tr, te = [], []
    for rs in rows_by_season:
        k = int(len(rs) * frac)
        tr += rs[:k]
        te += rs[k:]
    return tr, te


def probe_sources():
    """Checks whether NBA.com's advanced stats answer from this machine (they block many cloud servers). Recorded, not used, until proven."""
    url = 'https://stats.nba.com/stats/leaguedashteamstats?Season=2025-26&SeasonType=Regular%20Season&MeasureType=Advanced&PerMode=PerGame&LeagueID=00'
    hdr = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
           'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com', 'Accept': 'application/json, text/plain, */*',
           'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true'}
    out = {'checked': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=25) as r:
            js = json.loads(r.read().decode('utf-8'))
        rows = ((js.get('resultSets') or [{}])[0].get('rowSet')) or []
        out['nba_stats'] = {'ok': bool(rows), 'msg': '%d team rows returned' % len(rows)}
    except Exception as e:  # noqa
        out['nba_stats'] = {'ok': False, 'msg': str(e)[:160]}
    jsave(os.path.join(DATA, 'source_probe.json'), out)
    print('NBA.com stats probe: %s' % out['nba_stats'], flush=True)
    return out


def run_replay():
    """--replay: loads the seasons in REPLAY_SEASONS (resumable), then:
    1. compares ways of building the prop expectation on games the comparison never fit on, and keeps the best;
    2. learns per-stat corrections for the average and the range, kept only where they help on unseen games;
    3. learns what who-is-playing and each schedule spot is worth (run_side_fit);
    4. checks whether NBA.com's advanced stats can be reached from here."""
    t0 = time.time()
    boxes = [replay_box_update(rng, t0) for rng in REPLAY_SEASONS]
    try:
        probe_sources()
    except Exception as e:  # noqa
        print('Probe failed: %s' % str(e)[:120], flush=True)
    if not all(b.get('complete') for b in boxes):
        msg = 'Replay: %s. Run it again to finish loading.' % ', '.join('%s %d of %d days' % (b['range'][1][:4], len(b['days_done']),
                                                                                         (dt.date.fromisoformat(b['range'][1]) - dt.date.fromisoformat(b['range'][0])).days + 1) for b in boxes)
        print(msg, flush=True)
        jsave(os.path.join(DATA, 'prop_replay.json'), {'complete': False, 'days': sum(len(b['days_done']) for b in boxes), 'msg': msg})
        return 0
    # 1. which way of reading a game log predicts best
    cand = []
    base_rows = [replay_predictions(b, PROP_CANDIDATES[0]) for b in boxes]
    for c in PROP_CANDIDATES:
        rows = base_rows if c is PROP_CANDIDATES[0] else [replay_predictions(b, c) for b in boxes]
        score, n = 0.0, 0
        for stat in REPLAY_STATS:
            _, te = split_rows([r[stat] for r in rows])
            _, te0 = split_rows([r[stat] for r in base_rows])
            ev = replay_events(te, step=max(1, len(te) // 4000), lines_from=te0)
            score += logloss(ev) * len(ev)
            n += len(ev)
        cand.append({'name': c['name'], 'w': list(c['w']), 'permin': c['permin'], 'score': round(score / max(n, 1), 5), 'rows': rows})
        print('Prop method "%s": held-out log loss %.5f' % (c['name'], score / max(n, 1)), flush=True)
    base_score = cand[0]['score']
    best = min(cand, key=lambda c: c['score'])
    if best['score'] > base_score - 0.0005:       # a change has to clearly beat the current method, not tie it
        best = cand[0]
    rows = best['rows']
    out = {'complete': True, 'v': REPLAY_V, 'seasons': [b['range'] for b in boxes], 'games': sum(len(b['games']) for b in boxes),
           'made': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'cfg': {'w': best['w'], 'permin': best['permin'], 'name': best['name']},
           'candidates': [{k: c[k] for k in ('name', 'score')} for c in cand], 'stats': {}}
    # 2. per-stat corrections, judged on the held-out part first
    for stat in REPLAY_STATS:
        allr = []
        for r in rows:
            allr += r[stat]
        if len(allr) < 200:
            continue
        tr, te = split_rows([r[stat] for r in rows])
        bt, stt = replay_fit(tr)
        hold_before = logloss(replay_events(te, 1.0, 1.0, max(1, len(te) // 6000)))
        hold_after = logloss(replay_events(te, bt, stt, max(1, len(te) // 6000)))
        helps = hold_after < hold_before
        b, s = replay_fit(allr) if helps else (1.0, 1.0)
        z = [(a - m) / sd for m, sd, a, _ in allr]
        out['stats'][stat] = {'n': len(allr), 'b': r3(b), 's': r3(s), 'use': helps and len(allr) >= REPLAY_MIN_N,
                              'avg_pred': r1(mean(r[0] for r in allr)), 'avg_actual': r1(mean(r[2] for r in allr)),
                              'in50': r3(sum(1 for x in z if abs(x) <= 0.674) / len(z)), 'in80': r3(sum(1 for x in z if abs(x) <= 1.282) / len(z)),
                              'holdout_before': round(hold_before, 4), 'holdout_after': round(hold_after, 4),
                              'table_before': replay_table(allr), 'table_after': replay_table(allr, b, s)}
        print('%s: %d predictions, average x%.3f, range x%.2f, held-out log loss %.4f -> %.4f%s' % (
            stat, len(allr), b, s, hold_before, hold_after, '' if helps else ' (no gain, not used)'), flush=True)
    for c in cand:
        c.pop('rows', None)
    jsave(os.path.join(DATA, 'prop_replay.json'), out)
    # 3. who is playing and schedule spots, on every replayed season
    try:
        run_side_fit(boxes)
    except Exception as e:  # noqa
        print('Side model fit failed: %s' % str(e)[:200], flush=True)
    return 0


def load_replay():
    js = jload(os.path.join(DATA, 'prop_replay.json'), {})
    REPLAY_FIX.clear()
    if js.get('complete') and js.get('cfg') and js.get('v') == REPLAY_V:
        PROP_CFG.update({'w': tuple(js['cfg']['w']), 'permin': bool(js['cfg']['permin'])})
    REPLAY_SUM.clear()
    if js.get('complete'):
        REPLAY_SUM.update({'games': js.get('games'), 'made': js.get('made'), 'cfg': js.get('cfg'), 'candidates': js.get('candidates'), 'stats': {k: {x: v.get(x) for x in ('n', 'b', 's', 'use', 'in50', 'in80', 'holdout_before', 'holdout_after', 'table_before', 'table_after')}
                                                                                     for k, v in (js.get('stats') or {}).items()}})
    if not js.get('complete'):
        if js.get('days'):
            src('prop_replay', False, js.get('msg') or 'last-season replay not finished yet')
        return
    for stat, v in (js.get('stats') or {}).items():
        if v.get('use'):
            REPLAY_FIX[stat] = {'b': v['b'], 's': v['s']}
    if REPLAY_FIX:
        src('prop_replay', True, 'prop ranges tuned from %s replayed predictions: %s' % (
            '{:,}'.format(sum(v['n'] for v in js['stats'].values())), ', '.join('%s x%.2f' % (STAT_NAMES.get(k, k).lower(), v['s']) for k, v in REPLAY_FIX.items())))


# ----------------------------------------------------------------------------- main
def team_block(ev_team, ratings, logs, injuries, inj_ok, today, name_to_abbr):
    a = ev_team['abbr']
    rt = ratings.get(a)
    rest_txt, b2b, _ = rest_info(rt, today)
    roster = [L for L in logs.values() if L['team'] == a]
    roster.sort(key=lambda L: -mean(g['min'] for g in L['games']))
    rot = []
    for L in roster[:8]:
        gs = L['games']
        rot.append({'id': L['id'], 'name': L['name'], 'min': r1(mean(g['min'] for g in gs)), 'pts': r1(mean(g['pts'] for g in gs)), 'reb': r1(mean(g['reb'] for g in gs)),
                    'ast': r1(mean(g['ast'] for g in gs)), 'tpm': r1(mean(g['tpm'] for g in gs)), 'pts_l5': r1(mean(g['pts'] for g in gs[-5:])), 'gp': len(gs)})
    inj = []
    idx = {norm_name(L['name']): L for L in roster}
    for i in (injuries or {}).get(a, []):
        e = dict(i)
        L = idx.get(norm_name(i['name']))
        if L:
            gs = L['games']
            e['id'] = L['id']
            e['avg'] = {'min': r1(mean(g['min'] for g in gs)), 'pts': r1(mean(g['pts'] for g in gs)), 'reb': r1(mean(g['reb'] for g in gs)), 'ast': r1(mean(g['ast'] for g in gs))}
        inj.append(e)
    return {'abbr': a, 'name': ev_team['name'], 'record': ev_team['record'] or ('%d-%d' % (rt['w'], rt['l']) if rt else ''), 'rest': rest_txt, 'b2b': b2b,
            'ratings': rt, 'rotation': rot, 'injuries': inj, 'injuries_ok': inj_ok}


# ----------------------------------------------------------------------------- closing line value (CLV)
# The closing line is the market's last and sharpest opinion. A pick that beat the close was a better number than the market ended up agreeing on,
# which is the quickest honest signal of skill: it shows up in dozens of picks, while win-loss records need hundreds.
CLV_SLOPE = {'spread': 0.033, 'total': 0.023}     # probability per point near the middle (normal curve with the typical NBA error)


def parse_utc(s):
    try:
        return dt.datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(dt.timezone.utc)
    except Exception:  # noqa
        return None


def close_for(h, m):
    """The closing line, price and no-vig probability on the SAME side the pick took, from a market dict. None when the market lacks it."""
    f = market_fair(m)
    t, s = h.get('type'), h.get('side')
    if t == 'spread':
        if m.get('spread_home') is None:
            return None
        home = s == 'home'
        L = m['spread_home'] if home else -m['spread_home']
        od = m.get('spread_odds_home') if home else m.get('spread_odds_away')
        fp = f.get('home_cover')
        fp = fp if home else (None if fp is None else 1 - fp)
    elif t == 'total':
        if m.get('total') is None:
            return None
        over = s == 'over'
        L, od = m['total'], (m.get('over_odds') if over else m.get('under_odds'))
        fp = f.get('over')
        fp = fp if over else (None if fp is None else 1 - fp)
    elif t == 'moneyline':
        home = s == 'home'
        L, od = None, (m.get('ml_home') if home else m.get('ml_away'))
        fp = f.get('home_win')
        fp = fp if home else (None if fp is None else 1 - fp)
        if od is None:
            return None
    else:
        return None
    return {'line': L, 'odds': od, 'fair': r3(fp), 'n': m.get('n_books') or 1}


def clv_calc(h, c):
    """How much better the pick's number was than the close. clv_pts: points of line (positive = better than the close). clv: the pick's edge over the
    closing fair price, in probability, after allowing for any line move."""
    pts, slope = 0.0, 0.0
    if h['type'] == 'spread' and c.get('line') is not None:
        pts, slope = h['line'] - c['line'], CLV_SLOPE['spread']
    elif h['type'] == 'total' and c.get('line') is not None:
        pts, slope = ((c['line'] - h['line']) if h['side'] == 'over' else (h['line'] - c['line'])), CLV_SLOPE['total']
    elif h['type'] == 'prop' and c.get('line') is not None:
        pts = (c['line'] - h['line']) if h['side'] == 'over' else (h['line'] - c['line'])
        slope = 0.4 / max(h.get('sd') or 0.3 * abs(h['line']), 1.0)
    out = {'clv_pts': round(pts, 2) if h['type'] != 'moneyline' else None, 'clv': None}
    if c.get('fair') is not None and h.get('odds') is not None:
        out['clv'] = round(c['fair'] + slope * pts - impl(h['odds']), 4)
    return out


def run_closing():
    """--closing: runs every 20 minutes in the evening. Does nothing (and spends no credits) unless a logged pick tips off within CLOSE_WINDOW_MIN minutes.
    Then it reads the market once more and stores the closing line beside each such pick."""
    now = dt.datetime.now(dt.timezone.utc)
    o = os.environ.get('EPLOCKS_NOW')
    if o:
        now = parse_utc(o)
    hist_path = os.path.join(OUT, 'history.json')
    history = jload(hist_path, [])
    latest = jload(os.path.join(OUT, 'latest.json'), {})
    gmap = {g['id']: g for g in latest.get('games', [])}
    due = []
    for h in history:
        if h.get('result'):
            continue
        t = parse_utc(h.get('utc') or (gmap.get(h.get('game_id')) or {}).get('utc'))
        if t and 0.5 <= (t - now).total_seconds() / 60.0 <= CLOSE_WINDOW_MIN:
            due.append(h)
    if not due:
        print('Closing mode: no logged pick tips off within %d minutes. Nothing to do.' % CLOSE_WINDOW_MIN, flush=True)
        return 0
    print('Closing mode: %d picks tip off within %d minutes' % (len(due), CLOSE_WINDOW_MIN), flush=True)
    gids = sorted({h['game_id'] for h in due})
    espn_m = {}
    for ds in sorted({h['date'] for h in due}):
        try:
            js = http_json('%s/scoreboard?dates=%s&limit=100' % (ESPN, ds.replace('-', '')))
            for e in js.get('events') or []:
                ev = parse_event(e)
                if ev.get('market') and ev['id'] in gids:
                    espn_m[ev['id']] = ev['market']
        except Exception as ex:  # noqa
            note('Closing: scoreboard for %s failed: %s' % (ds, str(ex)[:100]))
    ev_like = [{'id': gid, 'state': 'pre', 'home': {'name': gmap[gid]['home']['name']}, 'away': {'name': gmap[gid]['away']['name']}} for gid in gids if gid in gmap]
    game_due = [h for h in due if h['type'] != 'prop']
    books_m = fetch_game_books([e for e in ev_like if e['id'] in {h['game_id'] for h in game_due}]) if (ODDS_KEY and game_due) else {}
    prop_by = {}
    prop_due = [h for h in due if h['type'] == 'prop']
    if ODDS_KEY and prop_due:
        try:
            evs = odds_get('events', {})            # free call; it also tells us how many credits are left
            if ODDS['left'] is not None and ODDS['left'] < PROPS_CLOSE_MIN_CREDITS:
                note('Closing: prop closing lines skipped to save Odds API credits (%s left)' % ODDS['left'])
                evs = []
            oid = {(nick(e.get('home_team')), nick(e.get('away_team'))): e['id'] for e in evs if isinstance(e, dict)} if isinstance(evs, list) else {}
            for gid in sorted({h['game_id'] for h in prop_due}):
                g = gmap.get(gid)
                if not g:
                    continue
                eid = oid.get((nick(g['home']['name']), nick(g['away']['name'])))
                mk = sorted({STAT_MARKET[h['stat']] for h in prop_due if h['game_id'] == gid and h.get('stat') in STAT_MARKET})
                if not eid or not mk:
                    continue
                if not odds_ok(len(mk) * len(ODDS_REGIONS.split(','))):
                    note('Closing: Odds API credits are nearly used up (%s left), so prop closing lines are skipped' % ODDS['left'])
                    break
                js = odds_get('events/%s/odds' % eid, {'regions': ODDS_REGIONS, 'markets': ','.join(mk), 'oddsFormat': 'american'})
                for p in agg_props(js):
                    prop_by[(gid, norm_name(p['player']), p['stat'])] = p
        except Exception as ex:  # noqa
            note('Closing: prop lines failed: %s' % str(ex)[:120])
    stamp, n_ok = now.strftime('%Y-%m-%dT%H:%M:%SZ'), 0
    for h in due:
        c = None
        if h['type'] == 'prop':
            p = prop_by.get((h['game_id'], norm_name(h.get('player', '')), h.get('stat')))
            if p:
                over, fo = h['side'] == 'over', p.get('fair_over')
                c = {'line': p['line'], 'odds': p['over'] if over else p['under'], 'fair': fo if over else (None if fo is None else r3(1 - fo)), 'n': p['n_books']}
        else:
            m = merge_market(espn_m.get(h['game_id']), books_m.get(h['game_id']))
            c = close_for(h, m) if m else None
        if not c:
            continue
        h['close'] = dict(c, t=stamp)
        h.update(clv_calc(h, c))
        if not h.get('utc') and gmap.get(h['game_id']):
            h['utc'] = gmap[h['game_id']].get('utc')
        n_ok += 1
    jsave(hist_path, history)
    if latest:
        stt = latest.setdefault('status', {}).setdefault('sources', {})
        stt['closing'] = {'ok': n_ok > 0, 'msg': 'closing lines saved for %d of %d picks near tip-off (last check %s UTC)' % (n_ok, len(due), now.strftime('%H:%M'))}
        latest['odds'] = {'left': ODDS['left'], 'used_closing': ODDS['used']}
        jsave(os.path.join(OUT, 'latest.json'), latest)
    print('Closing lines saved for %d of %d picks' % (n_ok, len(due)), flush=True)
    return 0


# ----------------------------------------------------------------------------- backtest: replay last season through the same ratings and pricing
BT_SECONDS = float(os.environ.get('EPLOCKS_BACKTEST_SECONDS') or 600)
LINE_KEYS = ('provider', 'spread_home', 'total', 'ml_home', 'ml_away', 'spread_odds_home', 'spread_odds_away', 'over_odds', 'under_odds')


def bt_fetch_lines(days, lc):
    """Closing lines ESPN still shows for last season's games (one scoreboard call per day, saved so it only ever happens once)."""
    done = set(lc['days_done'])
    todo = [d for d in days if d not in done]
    t0 = time.time()

    def one(d):
        try:
            js = http_json('%s/scoreboard?dates=%s&limit=100' % (ESPN, d.replace('-', '')), tries=2)
            out = {}
            for e in js.get('events') or []:
                ev = parse_event(e)
                if ev.get('market') and 'home' in ev:
                    out[ev['id']] = {k: ev['market'].get(k) for k in LINE_KEYS}
            return d, out
        except Exception:  # noqa
            return d, None
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i in range(0, len(todo), 30):
            if time.time() - t0 > BT_SECONDS:
                break
            for d, out in ex.map(one, todo[i:i + 30]):
                if out is not None:
                    lc['lines'].update(out)
                    lc['days_done'].append(d)
    return len(todo)


def bt_bucket(rows):
    n = len(rows)
    w = sum(1 for r in rows if r['result'] == 'W')
    l_ = sum(1 for r in rows if r['result'] == 'L')
    net = sum(r['profit'] for r in rows)
    return {'n': n, 'w': w, 'l': l_, 'hit': r3(w / float(w + l_)) if (w + l_) else None, 'roi': r3(net / n) if n else None, 'est': r3(mean(r['est'] for r in rows)) if rows else None}


def run_backtest(prior_state=None, quiet=False):
    """Replays last season day by day: ratings built only from games before each day, the same projection, blend and pricing as the live engine (no Claude, no
    injuries), priced against ESPN's closing lines and graded on the real final scores. Returns the summary dict and saves data/backtest.json."""
    st = prior_state or jload(os.path.join(DATA, 'prior.json'), None)
    games = {k: v for k, v in ((st or {}).get('games') or {}).items() if v.get('date') and (v.get('home') or {}).get('pts') is not None and (v.get('away') or {}).get('pts') is not None}
    if len(games) < 400:
        if not quiet:
            src('backtest', False, 'needs last season loaded first (%d games so far, needs 400)' % len(games))
        return None
    path = os.path.join(DATA, 'backtest_lines.json')
    lc = jload(path, {'v': 1, 'days_done': [], 'lines': {}})
    by_day = {}
    for g in games.values():
        by_day.setdefault(g['date'], []).append(g)
    days = sorted(by_day)
    bt_fetch_lines(days, lc)
    jsave(path, lc)
    n_lined = sum(1 for g in games.values() if (lc['lines'].get(g['id']) or {}).get('spread_home') is not None or (lc['lines'].get(g['id']) or {}).get('total') is not None)
    if len(lc['days_done']) < 0.9 * len(days):
        src('backtest', False, 'still collecting last season\'s lines: %d of %d days. It continues on the next run.' % (len(lc['days_done']), len(days)))
        return None
    if n_lined < 300:
        src('backtest', True, 'not available: ESPN no longer keeps last season\'s closing lines (%d of %d games had one). The replay mode checks the model on last season without lines instead.' % (n_lined, len(games)))
        return None
    saved = dict(CAL)
    CAL.update({'game': 1.0, 'prop': 1.0})
    seen, rows, mr, tr = {}, [], [], []
    try:
        for d in days:
            if len(seen) >= 150:
                ratings, lo, lp = build_ratings(seen)
                today_ = dt.date.fromisoformat(d)
                for g in by_day[d]:
                    mk = lc['lines'].get(g['id'])
                    if not mk or (mk.get('spread_home') is None and mk.get('total') is None):
                        continue
                    hab, aab = g['home']['ab'], g['away']['ab']
                    rh, ra = ratings.get(hab), ratings.get(aab)
                    if not rh or not ra or min(rh['n'], ra['n']) < 10:
                        continue
                    mo = project(hab, aab, ratings, lo, lp, today_)
                    if not mo:
                        continue
                    n_eff = min(rh['n_eff'], ra['n_eff'])
                    w = BT_BLEND * n_eff / (n_eff + 12.0)
                    mkm = -mk['spread_home'] if mk.get('spread_home') is not None else None
                    bm = mo['margin'] if mkm is None else mkm + w * (mo['margin'] - mkm)
                    bt_ = mo['total'] if mk.get('total') is None else mk['total'] + w * (mo['total'] - mk['total'])
                    act_m, act_t = g['home']['pts'] - g['away']['pts'], g['home']['pts'] + g['away']['pts']
                    if mkm is not None and n_eff >= 25:
                        mr.append((mkm, mo['margin'], act_m))
                    if mk.get('total') is not None and n_eff >= 25:
                        tr.append((mk['total'], mo['total'], act_t))
                    mk2 = dict(mk, provider='ESPN')
                    for k_ in ('spread_odds_home', 'spread_odds_away', 'over_odds', 'under_odds'):
                        mk2[k_] = mk2.get(k_) or -110
                    pk = game_picks({'market': mk2, 'away': {'abbr': aab}, 'home': {'abbr': hab}}, bm, bt_, n_eff, True, False, 0.0)
                    for p in pk:
                        h = dict(p, game_id=g['id'], date=d, units=1.0)
                        grade_history([h], {g['id']: g})
                        if h.get('result') in ('W', 'L', 'P'):
                            rows.append(h)
            for g in by_day[d]:
                seen[g['id']] = g
    finally:
        CAL.update(saved)
    dec = [r for r in rows if r['result'] in ('W', 'L')]
    sxy = sum((r['est_raw'] - impl(r['odds'])) * ((1.0 if r['result'] == 'W' else 0.0) - impl(r['odds'])) for r in dec)
    sxx = sum((r['est_raw'] - impl(r['odds'])) ** 2 for r in dec)
    lam = clamp(sxy / sxx, 0.0, 1.0) if sxx > 0 else None

    def best_w(trip):
        num = sum((mo - mk) * (a - mk) for mk, mo, a in trip)
        den = sum((mo - mk) ** 2 for mk, mo, a in trip)
        return clamp(num / den, 0.0, 1.0) if den > 0 else None
    wm, wt = best_w(mr), best_w(tr)

    def mae(trip, f):
        return r3(mean(abs(f(mk, mo) - a) for mk, mo, a in trip)) if trip else None

    def rms(trip, f):
        return r3(math.sqrt(mean((f(mk, mo) - a) ** 2 for mk, mo, a in trip))) if trip else None
    edges = (('2 to 4 points', 0.02, 0.04), ('4 to 6 points', 0.04, 0.06), ('6 or more', 0.06, 9.0))
    ests = (('Model chance 50 to 55%', 0.0, 0.55), ('55 to 60%', 0.55, 0.60), ('60% or more', 0.60, 1.01))
    types = {}
    for t in ('spread', 'total', 'moneyline'):
        types[t] = bt_bucket([r for r in rows if r['type'] == t])
    months = {}
    for r in rows:
        months.setdefault(r['date'][:7], []).append(r)
    bt = {'generated_at': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'season': '%s to %s' % (days[0], days[-1]), 'engine': ENGINE_VERSION,
          'games': len(games), 'lined': n_lined, 'bets': len(rows), 'all': bt_bucket(rows), 'by_type': types,
          'by_edge': [dict(label=lb, **bt_bucket([r for r in rows if lo_ <= r['edge'] < hi_])) for lb, lo_, hi_ in edges],
          'by_est': [dict(label=lb, **bt_bucket([r for r in rows if lo_ <= r['est'] < hi_])) for lb, lo_, hi_ in ests],
          'months': [dict(m=k, **bt_bucket(v)) for k, v in sorted(months.items())],
          'lambda': r3(lam), 'lambda_n': len(dec),
          'mae_margin': {'market': mae(mr, lambda mk, mo: mk), 'model': mae(mr, lambda mk, mo: mo), 'blend': mae(mr, lambda mk, mo: mk + (wm or 0) * (mo - mk)), 'n': len(mr)},
          'mae_total': {'market': mae(tr, lambda mk, mo: mk), 'model': mae(tr, lambda mk, mo: mo), 'blend': mae(tr, lambda mk, mo: mk + (wt or 0) * (mo - mk)), 'n': len(tr)},
          'best_w_margin': r3(wm), 'best_w_total': r3(wt),
          'sd_margin': rms(mr, lambda mk, mo: mk + (wm or 0) * (mo - mk)), 'sd_total': rms(tr, lambda mk, mo: mk + (wt or 0) * (mo - mk)),
          'note': 'Replay of last season with team ratings only: no Claude, no injuries, no props (past prop lines are not available). Each day uses only games played before it.'}
    jsave(os.path.join(OUT, 'backtest.json'), bt)
    src('backtest', True, '%d games replayed, %d bets, ROI %s%%; model keeps about %s%% of its raw edge' % (len(games), len(rows), r1(100 * (bt['all']['roi'] or 0)), r1(100 * lam) if lam is not None else '?'))
    return bt


BT_BLEND = 0.5


def apply_backtest():
    """Seeds the starting beliefs from last season's replay when it has enough bets. Live results still take over as they accumulate."""
    global BLEND_MAX
    bt = jload(os.path.join(OUT, 'backtest.json'), None)
    if not bt or (bt.get('bets') or 0) < 150:
        return None
    msg = []
    if bt.get('lambda') is not None and (bt.get('lambda_n') or 0) >= 150:
        CAL_PRIOR['game'] = clamp(bt['lambda'], 0.3, 1.0)
        CAL['game'] = CAL_PRIOR['game']
        msg.append('game picks start at x%.2f of the raw edge' % CAL_PRIOR['game'])
    wm = bt.get('best_w_margin')
    if wm is not None and (bt.get('mae_margin') or {}).get('n', 0) >= 300:
        BLEND_MAX = clamp(wm / 0.75, 0.05, 0.6)
        msg.append('model weight against the market tops out at %.2f' % BLEND_MAX)
    if msg:
        note('Last-season backtest set the starting point: ' + '; '.join(msg))
    return bt


def maybe_backtest(prior_state):
    """Runs the replay once last season has fully loaded, then again about once a month."""
    if not prior_state or not prior_state.get('complete'):
        return
    bt = jload(os.path.join(OUT, 'backtest.json'), None)
    if bt:
        try:
            age = (dt.datetime.now(dt.timezone.utc) - parse_utc(bt['generated_at'])).days
        except Exception:  # noqa
            age = 99
        if age < 30:
            return
    try:
        run_backtest(prior_state)
    except Exception as e:  # noqa
        src('backtest', False, 'replay failed: %s' % str(e)[:140])


def main(argv):
    only_data = '--no-llm' in argv
    global API_KEY
    if only_data:
        API_KEY = ''
    if '--closing' in argv:
        rc = run_closing()
        try:
            reason = news_check()
        except Exception as e:  # noqa
            print('Injury check failed: %s' % str(e)[:160], flush=True)
            reason = None
        if not reason:
            return rc
        print('Injury news: %s. Rerunning the card.' % reason, flush=True)
        STATUS['sources'].clear()
        STATUS['notes'].clear()
        NEWS['reason'] = reason
    if '--replay' in argv:
        return run_replay()
    if '--backtest' in argv:
        bt = run_backtest(None)
        print(json.dumps({'backtest': bool(bt), 'status': STATUS['sources'].get('backtest'), 'notes': STATUS['notes'][-3:]}), flush=True)
        return 0
    today = today_ct()
    started = dt.datetime.now(dt.timezone.utc)
    print('EpLocks run for', today, flush=True)

    try:
        load_replay()
    except Exception as e:  # noqa
        src('prop_replay', False, 'replay corrections not loaded: %s' % str(e)[:100])
    try:
        load_side_model()
    except Exception as e:  # noqa
        src('side_model', False, 'side model not loaded: %s' % str(e)[:100])
    cache, slate_events = update_games(today)
    try:
        books_ = fetch_game_books(slate_events)
        for ev_ in slate_events:
            ev_['market'] = merge_market(ev_.get('market'), books_.get(ev_['id']))
    except Exception as e:  # noqa
        note('Multi-book prices skipped: %s' % str(e)[:100])
    games = {k: v for k, v in cache['games'].items() if v.get('date') and v['date'] <= today.isoformat()}
    prior_state = None
    try:
        prior_state = update_prior(today)
    except Exception as e:  # noqa
        src('prior_season', False, 'last season could not be loaded: %s' % str(e)[:140])
    global CL_T0
    CL_T0 = time.time()
    pr_pack = None
    try:
        pr_pack = prior_ratings(prior_state)
    except Exception as e:  # noqa
        src('prior_season', False, 'last season could not be used: %s' % str(e)[:140])
    if 'prior_season' not in STATUS['sources']:
        if not PRIOR_ON:
            src('prior_season', True, 'switched off')
        elif pr_pack:
            src('prior_season', True, 'last season (%d games) is the starting point for team ratings and player averages' % len(prior_state['games']))
        elif prior_state and prior_state.get('complete'):
            src('prior_season', False, 'last season loaded but had too few games to use')
        elif prior_state is not None:
            src('prior_season', True, 'still loading last season: %d of %d days done. It finishes over the next run or two; until then picks use this season only.' % (
                len(prior_state['days_done']), (dt.date.fromisoformat(PRIOR_END) - dt.date.fromisoformat(PRIOR_START)).days + 1))
    try:
        maybe_backtest(prior_state)
        apply_backtest()
    except Exception as e:  # noqa
        note('Backtest skipped: %s' % str(e)[:100])
    CL_T0 = time.time()
    ratings, lg_ortg, lg_pace = build_ratings(games, pr_pack) if games else ({}, 112.0, 99.0)
    sane = (not games) or (95.0 <= lg_ortg <= 135.0 and 85.0 <= lg_pace <= 112.0)
    if not sane:
        note('League averages look wrong (%.1f points per 100, pace %.1f). The box-score data may be damaged, so every pick is capped at Low.' % (lg_ortg, lg_pace))
    logs, name_index = build_player_logs(games, prior_player_means(prior_state) if pr_pack else None)
    side_tr = None
    try:
        box_ = jload(box_path(REPLAY_SEASONS[-1]), None)
        box_ = box_ if box_ and box_.get('complete') and box_.get('v') == REPLAY_V else None
        rates_, repl_ = prior_rates_from(box_, prior_state if pr_pack else None)
        side_tr = SideTracker(rates_, repl_)
        for g_ in sorted(games.values(), key=lambda x_: (x_.get('date') or '', x_['id'])):
            if (g_.get('home') or {}).get('pts') is not None:
                side_tr.add(g_)
        box_ = None
    except Exception as e:  # noqa
        side_tr = None
        src('side_model', False, 'who-is-playing values could not be built: %s' % str(e)[:120])
    src('ratings', bool(ratings) and sane, ('%d teams rated from %d games%s' % (len(ratings), len(games), ' (with last season as a starting point)' if pr_pack else '')) if sane else 'league averages look wrong, so picks are capped at Low')
    shot_data = None
    try:
        shot_data = build_shot_data(games, prior_state if pr_pack else None)
        tot = shot_data['shots']
        placed = shot_data['coord'] + shot_data['text']
        if not tot:
            src('shot_data', False, 'ESPN play-by-play had no shot plays, so there are no shot maps yet')
        elif placed < 0.6 * tot:
            src('shot_data', False, 'only %d of %d shots could be placed on the court, so shot maps are off' % (placed, tot))
        elif len(shot_data['lfg']) < len(ZONES):
            src('shot_data', False, 'not enough shots in every zone yet to build shot maps (%d shots placed)' % placed)
        else:
            src('shot_data', True, '%d shots from %d games placed on the court (%d%% by coordinates)' % (placed, shot_data['games'], round(100.0 * shot_data['coord'] / placed)))
    except Exception as e:  # noqa
        shot_data = None
        src('shot_data', False, 'shot maps failed: %s' % str(e)[:140])
    if shot_data is not None and STATUS['sources'].get('shot_data', {}).get('ok') is False:
        shot_data = None

    name_to_abbr = {}
    for ev in slate_events:
        for s in ('home', 'away'):
            name_to_abbr[ev[s]['name'].lower()] = ev[s]['abbr']
    for g in games.values():
        for s in ('home', 'away'):
            name_to_abbr.setdefault(g[s]['name'].lower(), g[s]['ab'])
    injuries = fetch_injuries(name_to_abbr)
    if NEWS.get('reason'):
        note('Rerun after injury news: %s' % NEWS['reason'])
    try:
        if injuries is not None:
            watch_save(injuries, today.isoformat())
    except Exception as e:  # noqa
        note('Injury snapshot not saved: %s' % str(e)[:100])
    tg = team_gids(games)
    inj_ok = injuries is not None
    pdef = profiles = prof_lg = None
    try:
        pdef = build_pos_defense(games, prior_state if pr_pack else None)
        src('position_defense', bool(pdef), ('defense by position from %d games%s' % (pdef['games'], ' plus last season' if pdef['prior'] else '')) if pdef else 'not enough games with positions yet')
    except Exception as e:  # noqa
        src('position_defense', False, 'failed: %s' % str(e)[:120])
    try:
        profiles, prof_lg = build_team_profiles(games)
        src('team_profiles', bool(profiles), '%d teams profiled from this season\'s box scores' % len(profiles))
    except Exception as e:  # noqa
        profiles, prof_lg = {}, {}
        src('team_profiles', False, 'failed: %s' % str(e)[:120])
    rstats = None
    try:
        rstats = build_ref_stats(games)
        src('referees', True, ('%d officials measured from %d games' % (len(rstats['refs']), rstats['games'])) if rstats else 'not enough games with officials stored yet (they fill in as the season goes)')
    except Exception as e:  # noqa
        src('referees', False, 'referee stats failed: %s' % str(e)[:120])
    assign, manual = {}, {}
    try:
        assign = fetch_assignments(slate_events)
        manual, mbad = parse_refs_file(os.path.join(HERE, 'refs.txt'), slate_events)
        if manual:
            assign.update(manual)
        pre_n = sum(1 for e in slate_events if e.get('state') == 'pre')
        if pre_n:
            note('Officials found for %d of %d upcoming games%s' % (len(assign), pre_n, ' (the NBA usually posts them the morning of the game; later runs pick up the rest)' if len(assign) < pre_n else ''))
        if mbad:
            note('refs.txt: %d line(s) could not be matched to a game: %s' % (len(mbad), ' | '.join(mbad[:3])))
    except Exception as e:  # noqa
        note('Officials could not be loaded: %s' % str(e)[:100])
    ref_src = {}
    for gid in assign:
        ref_src[gid] = 'refs.txt' if gid in manual else 'ESPN'
    notes_data = None
    try:
        notes_data = parse_notes(os.path.join(HERE, 'notes.txt'), name_to_abbr, name_index, logs)
        if notes_data and (notes_data['n'] or notes_data['bad']):
            src('notes', True, '%d notes loaded from notes.txt%s' % (notes_data['n'], ('; %d lines could not be matched to a team or player: %s' % (len(notes_data['bad']), ' | '.join(notes_data['bad'][:3]))) if notes_data['bad'] else ''))
    except Exception as e:  # noqa
        src('notes', False, 'notes.txt could not be read: %s' % str(e)[:120])

    # grade finished picks first
    hist_path = os.path.join(OUT, 'history.json')
    history = jload(hist_path, [])
    ch = grade_history(history, games, today)
    if ch:
        note('Graded %d finished picks' % ch)
    parlays_ = jload(os.path.join(OUT, 'parlays.json'), [])
    try:
        if grade_parlays(parlays_, games, today):
            jsave(os.path.join(OUT, 'parlays.json'), parlays_)
    except Exception as e:  # noqa
        note('Parlay grading failed: %s' % str(e)[:120])
    try:
        cal, cal_n = build_calibration(history)
        CAL.update(cal)
        src('calibration', True, 'game picks x%.2f, prop picks x%.2f of the raw edge (from %d + %d graded picks; starting assumptions until a few hundred are graded)' % (
            CAL['game'], CAL['prop'], cal_n['game'], cal_n['prop']))
    except Exception as e:  # noqa
        src('calibration', False, 'self-calibration failed: %s' % str(e)[:120])
    try:
        build_judgment(history)
        src('judgment', True, ('Claude\'s own probability moves the model\'s by %.0f%% (at most %.0f points)%s' % (
            100 * JUDG['w'], 100 * JUDG_CAP, ('; so far Claude\'s probabilities score %s against the model\'s %s over %d graded picks (lower is better)' % (
                JUDG['brier_j'], JUDG['brier_m'], JUDG['n'])) if JUDG['brier_j'] is not None else '; the weight adjusts itself after 40 graded picks')) if JUDG['w'] > 0 else 'switched off')
    except Exception as e:  # noqa
        src('judgment', False, 'failed: %s' % str(e)[:120])

    # auditor: learn from the graded record before today's picks are made
    audit = None
    try:
        audit_path = os.path.join(OUT, 'audit.json')
        audit = run_audit(history, today, jload(audit_path, {}), parlays_)
        jsave(audit_path, audit)
        src('auditor', True, '%d graded picks reviewed; %d lessons%s' % (audit['graded'], len(audit['lessons']), ' (written by Claude)' if audit.get('by') == 'claude' else ''))
    except Exception as e:  # noqa
        src('auditor', False, 'the auditor failed: %s' % str(e)[:120])

    # props
    props_txt = ''
    try:
        with open(os.path.join(HERE, 'props.txt'), encoding='utf8') as f:
            props_txt = f.read()
    except Exception:
        pass
    lines = [l.strip() for l in props_txt.splitlines() if l.strip() and not l.strip().startswith('#')]
    props_cache_path = os.path.join(DATA, 'props_cache.json')
    pcache = jload(props_cache_path, {})
    parsed, left = parse_props_local(lines)
    auto = fetch_odds_props(slate_events)
    if auto:
        have = {(norm_name(p['player']), p['stat'], p['line']) for p in parsed}
        parsed = [p for p in auto if (norm_name(p['player']), p['stat'], p['line']) not in have] + parsed
    if left:
        if API_KEY:
            parsed += parse_props_llm(left, pcache)
            jsave(props_cache_path, pcache)
            src('props_parse', True, '%d lines read by Claude' % len(left))
        else:
            note('%d prop lines are not in the "Player | Stat | line | over | under" format and need an API key to be read' % len(left))
    slate_abbrs = {ev[s]['abbr'] for ev in slate_events for s in ('home', 'away')}
    routed, unmatched = route_props(parsed, logs, name_index, slate_abbrs)
    not_playing = [p for p in routed if p['team'] not in slate_abbrs]
    routed = [p for p in routed if p['team'] in slate_abbrs]

    # previous output, to keep analysis for games that already started
    prev = jload(os.path.join(OUT, 'latest.json'), {})
    prev_games = {g['id']: g for g in prev.get('games', [])} if prev.get('for_date') == today.isoformat() else {}

    out_games, all_picks = [], []
    lines_path = os.path.join(DATA, 'lines_seen.json')
    lines_seen = {k: v for k, v in jload(lines_path, {}).items() if k >= (today - dt.timedelta(days=2)).isoformat()}
    seen_today = lines_seen.setdefault(today.isoformat(), {})
    inj_names_by_team = {t: {norm_name(i['name']): i['status'].lower() for i in lst} for t, lst in (injuries or {}).items()}
    lg_total = 2 * lg_ortg * lg_pace / 100.0
    acache_path = os.path.join(DATA, 'analysis_cache.json')
    acache = {k: v for k, v in jload(acache_path, {}).items() if isinstance(v, dict) and v.get('date') == today.isoformat()}
    reused = 0
    for ev in sorted(slate_events, key=lambda e: e.get('utc') or ''):
        if ev['state'] != 'pre' and ev['id'] in prev_games:
            out_games.append(prev_games[ev['id']])
            all_picks.extend(prev_games[ev['id']].get('pick_objs', []))
            continue
        home = team_block(ev['home'], ratings, logs, injuries, inj_ok, today, name_to_abbr)
        away = team_block(ev['away'], ratings, logs, injuries, inj_ok, today, name_to_abbr)
        mo = project(home['abbr'], away['abbr'], ratings, lg_ortg, lg_pace, today)
        side_info = None
        if mo and side_tr:
            try:
                def mk_av(team_inj):
                    mp = {norm_name(i['name']): i['status'] for i in team_inj}

                    def nm(pid):
                        return norm_name((logs.get(pid) or {}).get('name', ''))
                    return (lambda pid: 1.0 - inj_weight(mp.get(nm(pid), ''))), (lambda pid: nm(pid) in mp)
                av_h, ls_h = mk_av(home['injuries'] if inj_ok else [])
                av_a, ls_a = mk_av(away['injuries'] if inj_ok else [])
                x_, xt_, det_ = side_features(side_tr, home['abbr'], away['abbr'], today.isoformat(), av_h, av_a, ls_h, ls_a)
                am_, at_ = side_adjust(x_, xt_)
                mo['margin'] += am_
                mo['total'] += at_
                names_ = {it[0]: (logs.get(it[0]) or {}).get('name', it[0]) for s_ in ('home', 'away') for it in det_[s_]['items']}
                side_info = {'margin': r1(am_), 'total': r1(at_), 'fitted': SIDE['fitted'], 'x': {k: r1(v) for k, v in x_.items()},
                             'missing': {'home': r1(det_['home']['missing']), 'away': r1(det_['away']['missing'])},
                             'lines': side_explain(x_, det_, home['abbr'], away['abbr'], names_)}
            except Exception as e:  # noqa
                note('Who-is-playing adjustment skipped for %s @ %s: %s' % (away['abbr'], home['abbr'], str(e)[:100]))
        crew = None
        try:
            crew = crew_info(assign.get(ev['id']), rstats, ref_src.get(ev['id'], 'ESPN'))
            if mo and crew and crew['adj_total']:
                mo['total'] += crew['adj_total']
        except Exception as e:  # noqa
            crew = None
            note('Officials skipped for %s: %s' % (ev['id'], str(e)[:80]))
        def n_eff_(t):
            r_ = t['ratings'] or {}
            return r_.get('n_eff', r_.get('n', 0))
        min_real = min((home['ratings'] or {}).get('n', 0), (away['ratings'] or {}).get('n', 0))
        min_n = min(n_eff_(home), n_eff_(away)) if sane else 0
        mk = ev['market']
        move = None
        if mk and ev['state'] == 'pre' and (mk.get('spread_home') is not None or mk.get('total') is not None):
            first = seen_today.setdefault(ev['id'], {'sp': mk.get('spread_home'), 'tot': mk.get('total'), 't': started.strftime('%Y-%m-%dT%H:%M:%SZ')})
            move = {'sp': r1(mk['spread_home'] - first['sp']) if mk.get('spread_home') is not None and first.get('sp') is not None else None,
                    'tot': r1(mk['total'] - first['tot']) if mk.get('total') is not None and first.get('tot') is not None else None,
                    'since': first['t'], 'sp0': first.get('sp'), 'tot0': first.get('tot')}
        blend = None
        if mo:
            w = BLEND_MAX * min_n / (min_n + 12.0) if mk else 1.0
            mk_margin = -mk['spread_home'] if (mk and mk.get('spread_home') is not None) else None
            mk_total = mk['total'] if (mk and mk.get('total') is not None) else None
            bm = mo['margin'] if mk_margin is None else mk_margin + w * (mo['margin'] - mk_margin)
            bt = mo['total'] if mk_total is None else mk_total + w * (mo['total'] - mk_total)
            blend = {'weight': r3(w), 'margin': r1(bm), 'total': r1(bt)}
        elif mk and mk.get('spread_home') is not None:
            blend = {'weight': 0.0, 'margin': r1(-mk['spread_home']), 'total': r1(mk['total']) if mk.get('total') is not None else None}
        h2h = []
        for gm in games.values():
            if {gm['home']['ab'], gm['away']['ab']} == {home['abbr'], away['abbr']} and gm['home'].get('pts') is not None:
                w_, l_ = (gm['home'], gm['away']) if gm['home']['pts'] > gm['away']['pts'] else (gm['away'], gm['home'])
                h2h.append('%s beat %s %d-%d at %s on %s' % (w_['ab'], l_['ab'], w_['pts'], l_['pts'], gm['home']['ab'], gm['date']))
        g = {'id': ev['id'], 'tip': ev['tip'], 'utc': ev['utc'], 'state': ev['state'], 'away': away, 'home': home, 'market': mk,
             'model': {k: r1(v) for k, v in mo.items()} if mo else None, 'blend': blend, 'h2h': h2h, 'crew': crew, 'side': side_info, 'move': move}
        # props for this game
        rows = []
        if mo:
            env = clamp((((blend['total'] if blend else mo['total'])) / lg_total) ** 0.5, 0.94, 1.06) if lg_total else 1.0
            for p in routed:
                if p['team'] not in (home['abbr'], away['abbr']):
                    continue
                tm_ = home if p['team'] == home['abbr'] else away
                bm = blowout_mult(logs[p['pid']], blend['margin'] if blend else mo['margin'])
                pf, pf_note = 1.0, ''
                try:
                    opp_ab = away['abbr'] if p['team'] == home['abbr'] else home['abbr']
                    pf, pf_note = pos_factor(pdef, opp_ab, logs[p['pid']].get('pos'), p['stat'])
                    rm, rm_note = role_mult(injury_role(p['pid'], tm_['injuries'], logs, tg), logs[p['pid']], p['stat'], bm, pf, pf_note)
                except Exception as e:  # noqa
                    rm, rm_note = 1.0, ''
                    note('Role adjustment skipped for %s: %s' % (p['name'], str(e)[:80]))
                pm = prop_model(logs[p['pid']], p['stat'], p['line'], env=env, rm=rm)
                if not pm:
                    unmatched.append(dict(p, why='Fewer than 3 games of data for this player'))
                    continue
                opp = away['abbr'] if p['team'] == home['abbr'] else home['abbr']
                st = inj_names_by_team.get(p['team'], {}).get(norm_name(p['name']), '')
                rows.append(dict(p, model=pm, vs_opp=vs_opp(logs[p['pid']], p['stat'], opp), injury=st, env=r3(env), rm=r3(rm), rm_note=rm_note, pf=r3(pf), pf_note=pf_note))
            def best_edge(r):
                m_ = r['model']
                e = []
                for pp, pr in ((m_['p_over'], r['over']), (m_['p_under'], r['under'])):
                    e.append(pp - impl(pr if pr is not None else -110))
                return max(e)
            rows.sort(key=lambda r: -best_edge(r))
            rows = rows[:MAX_PROP_ROWS]
            for i, r in enumerate(rows):
                r['id'] = 'p%d' % (i + 1)
        g['prop_rows'] = rows
        try:
            g['tendencies'] = {'away': (profiles or {}).get(away['abbr']), 'home': (profiles or {}).get(home['abbr']), 'league': prof_lg,
                               'lines': matchup_lines(home['abbr'], away['abbr'], profiles or {}, prof_lg or {}, ratings, pdef)} if profiles else None
        except Exception as e:  # noqa
            g['tendencies'] = None
            note('Tendencies skipped for %s @ %s: %s' % (away['abbr'], home['abbr'], str(e)[:100]))
        g['notes'] = notes_for_game(notes_data, home['abbr'], away['abbr'], logs)
        try:
            g['player_board'] = build_player_board([home, away], logs, tg, env, (blend['margin'] if blend else mo['margin']), inj_names_by_team, pdef) if mo else []
        except Exception as e:  # noqa
            g['player_board'] = []
            note('Player board skipped for %s: %s' % (ev['id'], str(e)[:100]))
        try:
            g['shot_profile'] = shot_profile(home, away, shot_data, logs, [r['pid'] for r in rows]) if shot_data else None
        except Exception as e:  # noqa
            g['shot_profile'] = None
            note('Shot profile failed for %s @ %s: %s' % (away['abbr'], home['abbr'], str(e)[:100]))

        for r in rows:
            try:
                r['sig'] = prop_signals(r)       # computed before Claude so the packet can show them; recomputed below after any adjustment
            except Exception:  # noqa
                r['sig'] = None
        # Claude
        analysis, adjust_m, adjust_t, pnotes, judg = None, 0.0, 0.0, {}, None
        if API_KEY and mo and ev['state'] == 'pre':
            prompt = packet(g) + ('\n\n' + AUDIT['text'] if AUDIT.get('text') else '') + '\n\nInjuries loaded: %s. Return the JSON object now.' % ('yes' if inj_ok else 'NO')
            ph = hashlib.sha1((SYSTEM + prompt).encode('utf8')).hexdigest()
            ent = acache.get(ev['id'])
            if ent and ent.get('h') == ph and isinstance(ent.get('js'), dict):
                js = ent['js']
                reused += 1
            else:
                txt = call_claude(prompt, SYSTEM, max_tokens=3500)
                js = extract_json(txt) if txt else None
                if isinstance(js, dict):
                    acache[ev['id']] = {'h': ph, 'date': today.isoformat(), 'js': js}
            if isinstance(js, dict):
                cap_m, cap_t = (2.0, 5.0) if mk else (4.0, 8.0)
                adjust_m, adjust_t = clamp(num_or(js.get('margin_adj')), -cap_m, cap_m), clamp(num_or(js.get('total_adj')), -cap_t, cap_t)
                a = js.get('analysis') if isinstance(js.get('analysis'), dict) else {}
                def lst(x):
                    return [str(i)[:300] for i in x][:6] if isinstance(x, list) else []
                jd_ = js.get('judgment') if isinstance(js.get('judgment'), dict) else None
                if jd_:
                    judg = {'home_win': prob_or_none(jd_.get('home_win')), 'cover_home': prob_or_none(jd_.get('cover_home')), 'over': prob_or_none(jd_.get('over')),
                            'conf': str(jd_.get('conf') or 'med')[:6].lower(), 'concern': bool(jd_.get('concern')), 'why': str(jd_.get('why') or '')[:240]}
                    judg['scale'] = 0.5 if judg['conf'] == 'low' else 1.0
                analysis = {'lean': str(a.get('lean') or '')[:300], 'why': lst(a.get('why')), 'risks': lst(a.get('risks')), 'matchups': lst(a.get('matchups')), 'verify': lst(a.get('verify'))}
                for pn in js.get('props') or []:
                    if isinstance(pn, dict) and pn.get('id'):
                        pnotes[str(pn['id'])] = {'mult': clamp(num_or(pn.get('mult'), 1.0), 0.9, 1.1), 'skip': bool(pn.get('skip')), 'note': str(pn.get('note') or '')[:240], 'p_over': prob_or_none(pn.get('p_over'))}
            else:
                note('No usable analysis for %s @ %s' % (away['abbr'], home['abbr']))
        g['analysis'] = analysis
        g['judgment'] = dict(judg, w=r3(JUDG['w'])) if judg else None
        g['adjust'] = {'margin': r1(adjust_m), 'total': r1(adjust_t)}
        try:
            pl = {}
            for pid in [b['pid'] for b in g.get('player_board') or []] + [r['pid'] for r in rows] + [x['id'] for t_ in (home, away) for x in t_['rotation']]:
                if pid in logs and pid not in pl:
                    pl[pid] = plog(logs[pid])
            g['plogs'] = pl
        except Exception as e:  # noqa
            g['plogs'] = {}
            note('Player logs skipped for %s: %s' % (ev['id'], str(e)[:80]))

        picks = []
        if blend and ev['state'] == 'pre':
            fm = blend['margin'] + adjust_m
            ft = blend['total'] + adjust_t if blend['total'] is not None else None
            g['final'] = {'margin': r1(fm), 'total': r1(ft), 'home_win': r3(phi(fm / SD_MARGIN))}
            if not mo:
                g['final']['note'] = 'No team ratings yet, so this is the market number only.'
            uncertain = any(0 < inj_weight(i['status']) < 1.0 and (i.get('avg') or {}).get('min', 0) >= 24
                            for t in (home, away) for i in t['injuries'])
            picks = game_picks(g, fm, ft, min_n, inj_ok, uncertain, abs(adjust_m) + abs(adjust_t) / 3.0, judg=judg)
            for p_ in picks:
                try:
                    p_['sig'] = game_signals(g, p_, lg_pace)
                except Exception as e:  # noqa
                    note('Signals skipped for a pick in %s: %s' % (ev['id'], str(e)[:80]))
            if not mk:
                g['final']['note'] = 'No market line loaded, so no bet can be priced for this game.'
        for r in rows:
            r['sig'] = None
            pn = pnotes.get(r['id'], {})
            mult = pn.get('mult', 1.0)
            if mult != 1.0:
                r['model'] = prop_model(logs[r['pid']], r['stat'], r['line'], env=r['env'], mult=mult, rm=r.get('rm', 1.0))
            r['mult'], r['note'] = mult, pn.get('note', '')
            r['skipped'] = bool(pn.get('skip'))
            try:
                r['sig'] = prop_signals(r)
            except Exception as e:  # noqa
                note('Signals skipped for %s: %s' % (r.get('player'), str(e)[:80]))
            try:   # the chances the site shows for every prop: the same calibrated numbers the picks use, not the raw model
                m_, fo_ = r['model'], r.get('fair_over')
                ne_ = m_.get('n_eff', m_['n'])
                r['cal'] = {'over': r3(cal_p(m_['p_over'], r['over'] if r['over'] is not None else -110, 'prop', fo_, ne_)),
                            'under': r3(cal_p(m_['p_under'], r['under'] if r['under'] is not None else -110, 'prop', None if fo_ is None else 1 - fo_, ne_))}
            except Exception:  # noqa
                r['cal'] = None
            if ev['state'] != 'pre' or r['skipped'] or is_blocked(r['injury']):
                r['skipped'] = r['skipped'] or is_blocked(r['injury'])
                continue
            pk = prop_pick(r, mult, bool(r['injury']), pn.get('p_over'), bool(judg and judg.get('concern')), (judg or {}).get('why', ''))
            if pk:
                pk['reason'] = r['note']
                try:
                    pk['sig'] = prop_signals(r, pk['side'])
                except Exception:  # noqa
                    pk['sig'] = None
                if not sane:
                    pk['conf'] = 'Low'
                picks.append(pk)
        for i, p in enumerate(picks):
            p['id'] = '%s-%s-%d' % (today.isoformat(), ev['id'], i)
            p.update({'game_id': ev['id'], 'away': away['abbr'], 'home': home['abbr'], 'date': today.isoformat(), 'utc': ev['utc']})
        g['pick_objs'] = picks
        g['picks'] = [p['id'] for p in picks]
        g['dq'] = {'min_games': min_real, 'min_games_eff': min_n, 'prior': bool((home['ratings'] or {}).get('prior') and (away['ratings'] or {}).get('prior')), 'shots': bool(g.get('shot_profile')), 'injuries_loaded': inj_ok, 'market_loaded': bool(mk), 'analysis': bool(analysis)}
        for r in rows:
            r.pop('raw', None)
        out_games.append(g)
        all_picks.extend(picks)

    for p in all_picks:
        try:
            audit_adjust(p)
        except Exception:  # noqa
            pass
    all_picks = finalize_picks(all_picks, out_games)

    # history: replace today's picks for games still 'pre', keep the rest
    today_s = today.isoformat()
    pre_ids = {g['id'] for g in out_games if g.get('state') == 'pre'}
    old_by_id = {h['id']: h for h in history}
    history = [h for h in history if not (h['date'] == today_s and h['game_id'] in pre_ids and not h.get('result'))]
    have = {h['id'] for h in history}
    for p in all_picks:
        if p['id'] not in have and p['game_id'] in pre_ids:
            row_ = ({k: p[k] for k in ('id', 'date', 'game_id', 'away', 'home', 'type', 'side', 'label', 'line', 'odds', 'est', 'est_raw', 'edge', 'kind', 'conf', 'units') if k in p}
                    | {k: p[k] for k in ('fair', 'book', 'n_books', 'jp', 'est_nj', 'sd', 'utc', 'alt') if p.get(k) is not None}
                           | ({'player': p['player'], 'stat': p['stat'], 'pid': p['pid']} if p['type'] == 'prop' else {})
                           | ({'sig': [i['k'] for i in p['sig']['items']], 'sig_tier': p['sig']['tier']} if p.get('sig') else {})
                    | {'posted': started.strftime('%Y-%m-%dT%H:%M:%SZ'), 'v': ENGINE_VERSION})
            oh_ = old_by_id.get(p['id']) or {}
            row_.update({k: oh_[k] for k in ('close', 'clv', 'clv_pts') if k in oh_})
            history.append(row_)
    jsave(lines_path, lines_seen)
    jsave(hist_path, history)
    try:
        update_parlays(all_picks, out_games, games, today, started.strftime('%Y-%m-%dT%H:%M:%SZ'))
    except Exception as e:  # noqa
        src('parlays', False, 'parlay tracker failed: %s' % str(e)[:120])
    jsave(acache_path, acache)
    if reused:
        note('Reused the written analysis for %d games whose data had not changed' % reused)

    league = {'ortg': r1(lg_ortg), 'pace': r1(lg_pace), 'games': len(games), 'avg_total': r1(lg_total),
              'zones': {z: r3(v) for z, v in (shot_data['lfg'] if shot_data else {}).items()}, 'zone_labels': ZONE_LABEL}
    src('anthropic', bool(API_KEY) and STATUS['sources'].get('anthropic', {}).get('ok', True), 'model %s' % MODEL if API_KEY else 'no API key set, so there is no written analysis')
    result = {
        'engine': ENGINE_VERSION, 'prop_source': 'odds_api' if auto else ('props.txt' if lines else 'none'),
        'generated_at': started.strftime('%Y-%m-%dT%H:%M:%SZ'), 'for_date': today_s, 'model': MODEL if API_KEY else None, 'league': league,
        'params': {'home_edge': HCA, 'sd_margin': SD_MARGIN, 'sd_total': SD_TOTAL, 'b2b_points': B2B_PTS, 'calibration': {k: r3(v) for k, v in CAL.items()},
                   'blend_max': r3(BLEND_MAX), 'judgment': {'w': r3(JUDG['w']), 'cap': JUDG_CAP, 'n': JUDG['n'], 'brier_j': JUDG['brier_j'], 'brier_m': JUDG['brier_m']}},
        'odds': {'left': ODDS['left'], 'used': ODDS['used'], 'calls': ODDS['calls']},
        'status': {k: v for k, v in STATUS.items() if not k.startswith('_')}, 'games': out_games,
        'picks': [dict(p, label=p['label']) for p in all_picks], 'props_unmatched': unmatched[:60], 'props_not_playing': [dict(p) for p in not_playing][:60],
        'props_lines': len(lines) + (len(auto) if auto else 0), 'audit': audit, 'replay': dict(REPLAY_SUM) or None}
    # drop heavy duplicates
    for g in result['games']:
        for t in ('away', 'home'):
            g[t].pop('injuries_ok', None)
    if not STATUS['sources'].get('espn_games', {}).get('ok', False) and prev.get('games'):
        # Game data failed today: keep the last good slate on the site instead of wiping it, and show the failure.
        note('Game data failed, so the site is still showing the previous run.')
        prev['status'] = result['status']
        jsave(os.path.join(OUT, 'latest.json'), prev)
        print('Game data failed; kept previous output', flush=True)
        return 0
    jsave(os.path.join(OUT, 'latest.json'), result)
    try:
        send_alerts(prev, result)
        if STATUS['sources'].get('alerts'):
            result['status'] = {k: v for k, v in STATUS.items() if not k.startswith('_')}
            jsave(os.path.join(OUT, 'latest.json'), result)
    except Exception as e:  # noqa
        print('Alerts failed: %s' % str(e)[:160], flush=True)
    print('Wrote %d games, %d picks' % (len(out_games), len(all_picks)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
