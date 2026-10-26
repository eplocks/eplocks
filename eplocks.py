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
  7. Grades yesterday's picks from real box scores and writes everything to docs/data/*.json for the website.

Only the Python standard library is used. Every data source is wrapped so one failure never stops the run; the
Status tab on the site shows what loaded and what did not.
"""
import os, sys, json, re, math, time, difflib, unicodedata, hashlib
import datetime as dt
import urllib.request, urllib.error
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
MAX_PROP_ROWS = 30
BLOCK_STATUS = ('out', 'doubtful', 'questionable', 'day-to-day', 'game time decision', 'gtd')  # no prop picks on these players                                            # props per game sent to Claude

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
                players.append({'id': str(ath.get('id')), 'name': ath.get('displayName') or '', 'tid': tid, 'min': mn,
                                'pts': toi(g('PTS')) or 0, 'reb': toi(g('REB')) or 0, 'ast': toi(g('AST')) or 0,
                                'tpm': (pair(g('3PT'))[0] or 0), 'stl': toi(g('STL')) or 0, 'blk': toi(g('BLK')) or 0, 'tov': toi(g('TO')) or 0})
    out = {'id': ev['id'], 'date': ev['date'], 'utc': ev['utc'], 'players': []}
    for side in ('home', 'away'):
        e = ev[side]
        out[side] = dict(teams.get(e['id'], {}), id=e['id'], ab=e['abbr'], name=e['name'], pts=e['score'])
    ab_by_id = {ev['home']['id']: ev['home']['abbr'], ev['away']['id']: ev['away']['abbr']}
    for p in players:
        p['t'] = ab_by_id.get(p.pop('tid'), '')
        out['players'].append(p)
    return out


def update_games(today):
    """Fetch any missing finished games (and their box scores) from SEASON_START to today. Returns the cache."""
    path = os.path.join(DATA, 'games.json')
    cache = jload(path, {'games': {}, 'days_done': []})
    games, done = cache.setdefault('games', {}), set(cache.setdefault('days_done', []))
    d = dt.date.fromisoformat(SEASON_START)
    fetched = failed = 0
    slate_events = []
    try:
        while d <= today:
            ds = d.isoformat()
            if ds in done:
                d += dt.timedelta(days=1)
                continue
            js = http_json('%s/scoreboard?dates=%s&limit=100' % (ESPN, d.strftime('%Y%m%d')))
            events = [parse_event(e) for e in js.get('events') or []]
            all_final = True
            for ev in events:
                if 'home' not in ev or 'away' not in ev:
                    continue
                if d == today:
                    slate_events.append(ev)
                if not ev['completed']:
                    all_final = False
                    continue
                if ev['season_type'] not in (None, 2, 3) or ev['id'] in games:
                    continue
                try:
                    games[ev['id']] = parse_summary(http_json('%s/summary?event=%s' % (ESPN, ev['id'])), ev)
                    fetched += 1
                except Exception as e:  # noqa
                    failed += 1
                    all_final = False
                    note('box score failed for game %s: %s' % (ev['id'], str(e)[:120]))
            if d < today and all_final:
                done.add(ds)
            d += dt.timedelta(days=1)
        src('espn_games', True, '%d games stored, %d new, %d failed' % (len(games), fetched, failed))
    except Exception as e:  # noqa
        src('espn_games', False, str(e))
        note('Game history could not be fully updated: %s' % str(e)[:160])
    cache['days_done'] = sorted(done)
    jsave(path, cache)
    return cache, slate_events


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
def build_ratings(games):
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
    off = {t: lg_ortg for t in recs}
    dfn = {t: lg_ortg for t in recs}
    pace = {t: lg_pace for t in recs}
    for _ in range(14):
        n_off, n_dfn, n_pace = {}, {}, {}
        for t, rs in recs.items():
            n = len(rs)
            so = sum(r['ortg'] - (dfn.get(r['opp'], lg_ortg) - lg_ortg) for r in rs)
            sd = sum(r['drtg'] - (off.get(r['opp'], lg_ortg) - lg_ortg) for r in rs)
            sp = sum(r['poss'] - (pace.get(r['opp'], lg_pace) - lg_pace) for r in rs)
            n_off[t] = (so + K * lg_ortg) / (n + K)
            n_dfn[t] = (sd + K * lg_ortg) / (n + K)
            n_pace[t] = (sp + KP * lg_pace) / (n + KP)
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
            'n': n, 'w': w, 'l': n - w,
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
STAT_ALIAS = {
    'pts': 'points', 'point': 'points', 'points': 'points', 'reb': 'rebounds', 'rebs': 'rebounds', 'rebounds': 'rebounds', 'ast': 'assists', 'asts': 'assists',
    'assists': 'assists', '3pm': 'threes', '3s': 'threes', 'threes': 'threes', '3-pointers': 'threes', '3 pointers made': 'threes', 'three pointers made': 'threes',
    'pra': 'pra', 'pts+reb+ast': 'pra', 'pts+rebs+asts': 'pra', 'points+rebounds+assists': 'pra', 'pts+reb': 'pts_reb', 'pts+rebs': 'pts_reb',
    'points+rebounds': 'pts_reb', 'pts+ast': 'pts_ast', 'pts+asts': 'pts_ast', 'points+assists': 'pts_ast', 'reb+ast': 'reb_ast', 'rebs+asts': 'reb_ast',
    'rebounds+assists': 'reb_ast', 'stl': 'steals', 'steals': 'steals', 'blk': 'blocks', 'blocks': 'blocks', 'to': 'turnovers', 'tov': 'turnovers', 'turnovers': 'turnovers'}


def build_player_logs(games):
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
    return logs, index


def prop_model(log, stat, line, env=1.0, mult=1.0):
    fn = STAT_FN.get(stat)
    if not fn or not log:
        return None
    vals = [fn(g) for g in log['games'] if g['min'] > 0]
    n = len(vals)
    if n < 3:
        return None
    season, l10, l5 = mean(vals), mean(vals[-10:]), mean(vals[-5:])
    base = 0.5 * season + 0.3 * l10 + 0.2 * l5
    prior = max(1.0, SD_K[stat] * base)
    sd = math.sqrt((n * pstdev(vals) ** 2 + 5 * prior ** 2) / (n + 5))
    mu = base * env * mult
    half = abs(line - math.floor(line)) > 1e-9
    if half:
        p_over = 1 - phi((line - mu) / sd)
        p_under = 1 - p_over
    else:
        p_over = 1 - phi((line + 0.5 - mu) / sd)
        p_under = phi((line - 0.5 - mu) / sd)
    last10 = vals[-10:]
    return {'n': n, 'season': r1(season), 'l10': r1(l10), 'l5': r1(l5), 'mean': r1(mu), 'sd': r1(sd), 'p_over': r3(p_over), 'p_under': r3(p_under),
            'hit_l10': sum(1 for v in last10 if v > line), 'n_l10': len(last10), 'hit_all': sum(1 for v in vals if v > line),
            'min_avg': r1(mean(g['min'] for g in log['games'])), 'min_l5': r1(mean(g['min'] for g in log['games'][-5:]))}


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


def fetch_odds_props(slate_events):
    """Optional: pull player props from The Odds API for today's games. Returns a list of prop dicts, or None on failure."""
    if not ODDS_KEY:
        return None
    markets = [m for m in (os.environ.get('EPLOCKS_PROP_MARKETS') or 'player_points,player_rebounds,player_assists,player_threes').split(',') if m.strip() in ODDS_MARKETS]
    base = 'https://api.the-odds-api.com/v4/sports/basketball_nba'
    try:
        events = http_json('%s/events?apiKey=%s' % (base, ODDS_KEY))
    except Exception as e:  # noqa
        src('odds_api', False, str(e).replace(ODDS_KEY, '***')[:200])
        return None
    mine = {}
    for ev in slate_events:
        if ev['state'] == 'pre':
            mine[(ev['home']['name'].lower(), ev['away']['name'].lower())] = ev
    out, used, fails = [], 0, 0
    for e in events if isinstance(events, list) else []:
        key = (str(e.get('home_team', '')).lower(), str(e.get('away_team', '')).lower())
        if key not in mine:
            continue
        try:
            js = http_json('%s/events/%s/odds?apiKey=%s&regions=us&markets=%s&oddsFormat=american&bookmakers=%s' % (base, e['id'], ODDS_KEY, ','.join(markets), ODDS_BOOK), tries=2)
        except Exception as ex:  # noqa
            fails += 1
            note('Props fetch failed for %s: %s' % (e.get('id'), str(ex).replace(ODDS_KEY, '***')[:120]))
            continue
        used += len(markets)
        grouped = {}
        for bk in js.get('bookmakers') or []:
            for mk in bk.get('markets') or []:
                stat = ODDS_MARKETS.get(mk.get('key'))
                if not stat:
                    continue
                for oc in mk.get('outcomes') or []:
                    try:
                        line = float(oc.get('point'))
                    except (TypeError, ValueError):
                        continue
                    who = str(oc.get('description') or '').strip()
                    if not who:
                        continue
                    d = grouped.setdefault((who, stat, line), {'player': who, 'stat': stat, 'line': line, 'over': None, 'under': None, 'raw': 'Odds API ' + bk.get('key', '')})
                    side = str(oc.get('name') or '').lower()
                    if side in ('over', 'under'):
                        d[side] = odds_or_none(oc.get('price'))
            break
        out.extend(grouped.values())
    ok = bool(out)
    src('odds_api', ok, '%d props for %d games (about %d credits used)%s' % (len(out), len(mine), used, ('; %d games failed' % fails) if fails else '') if ok else
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
- Roster changes, coaching tendencies and scheme come from your own memory and may be out of date after the 2026 offseason. Tag every such claim (from memory) and never let one drive a pick on its own.
- The market line already reflects public injury news. When a market line is given, keep margin_adj and total_adj at 0 unless the packet shows a specific, quantifiable reason the line misses (a rating or rest mismatch with numbers). Never exceed +/-2.0 margin or +/-5.0 total when a line exists.
- Early season samples are small. Say so when it matters and do not overstate.
- You do not decide probabilities, confidence or stake sizes. The code computes them from your adjustments.
- For props, mult scales the player's expected output (0.90 to 1.10). Change it only for a stated reason that is in the DATA (a starter listed out raising usage, a minutes change visible in the last games). Use skip=true for any player listed Out, Doubtful or Questionable, or whose role looks unclear.
- Plain language. No hype. Be specific and cite the numbers you use.

Reply with one JSON object only:
{"margin_adj": number, "total_adj": number,
 "analysis": {"lean": "one sentence", "why": ["3 to 5 short data-led points"], "risks": ["1 to 3 short points"], "matchups": ["0 to 3 short matchup notes, tagged (from memory) when they are"], "verify": ["what to check before betting"]},
 "props": [{"id": "p3", "mult": 1.0, "skip": false, "note": "one short sentence"}]}"""


def call_claude(prompt, system, max_tokens=3500, tries=3):
    if not API_KEY:
        return None
    body = json.dumps({'model': MODEL, 'max_tokens': max_tokens, 'system': system, 'messages': [{'role': 'user', 'content': prompt}]}).encode()
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
    mo = g['model']
    if mo:
        L.append('Our projection from ratings: home margin %+.1f, total %.1f, pace %.1f. Blended with the market at weight %.2f -> margin %+.1f, total %.1f.' % (
            mo['margin'], mo['total'], mo['pace'], g['blend']['weight'], g['blend']['margin'], g['blend']['total']))
    if g['h2h']:
        L.append('Earlier meetings this season: ' + '; '.join(g['h2h']))
    L.append('Not loaded: coaching data, referee data, travel distance, player tracking, shot quality.')
    rows = g.get('prop_rows') or []
    if rows:
        L.append('\nPROP TABLE (id | player team | stat line | over/under price | n games | season/L10/L5 avg | model mean | model P(over) | hit L10 over | minutes avg/L5 | vs this opponent):')
        for r in rows:
            m_ = r['model']
            L.append('%s | %s %s | %s %s | %s/%s | n=%d | %s/%s/%s | mean %s | P(over) %s | L10 over %d/%d | min %s/%s | vs opp %s' % (
                r['id'], r['player'], r['team'], STAT_NAMES.get(r['stat'], r['stat']), r['line'], r['over'], r['under'], m_['n'], m_['season'], m_['l10'], m_['l5'], m_['mean'],
                m_['p_over'], m_['hit_l10'], m_['n_l10'], m_['min_avg'], m_['min_l5'], ('%s avg in %d g' % (r['vs_opp']['avg'], r['vs_opp']['n'])) if r.get('vs_opp') else 'none'))
    return '\n'.join(L)


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


def game_picks(g, final_margin, final_total, min_n, inj_ok, uncertain_key, adj_abs):
    m = g['market']
    picks = []
    away, home = g['away']['abbr'], g['home']['abbr']
    if not m:
        return picks
    cands = []
    if m.get('spread_home') is not None:
        L = m['spread_home']
        p_home = phi((final_margin + L) / SD_MARGIN)
        cands.append(('spread', 'home', '%s %+g' % (home, L), L, m['spread_odds_home'], p_home))
        cands.append(('spread', 'away', '%s %+g' % (away, -L), -L, m['spread_odds_away'], 1 - p_home))
    if m.get('total') is not None and final_total is not None:
        T = m['total']
        p_over = 1 - phi((T - final_total) / SD_TOTAL)
        cands.append(('total', 'over', 'Over %g' % T, T, m['over_odds'], p_over))
        cands.append(('total', 'under', 'Under %g' % T, T, m['under_odds'], 1 - p_over))
    if m.get('ml_home') is not None and m.get('ml_away') is not None:
        pw = phi(final_margin / SD_MARGIN)
        cands.append(('moneyline', 'home', '%s ML' % home, None, m['ml_home'], pw))
        cands.append(('moneyline', 'away', '%s ML' % away, None, m['ml_away'], 1 - pw))
    cands = [c for c in cands if c[4] is not None]
    scored = sorted(((c[5] - impl(c[4]), c) for c in cands), key=lambda x: -x[0])
    seen_type = set()
    for edge, c in scored:
        if edge < 0.02 or len(picks) >= 2 or c[0] in seen_type:
            continue
        seen_type.add(c[0])
        conf = game_conf(edge, min_n, inj_ok, uncertain_key, adj_abs, True)
        flag = ''
        if edge > 0.08:
            conf, flag = 'Low', 'Unusually large edge for an NBA market. Check injuries, the line and the data before trusting it.'
        picks.append({'type': c[0], 'side': c[1], 'label': c[2], 'line': c[3], 'odds': c[4], 'est': r3(c[5]), 'implied': r3(impl(c[4])), 'edge': r3(edge),
                      'kind': 'priced', 'conf': conf, 'units': kelly_units(c[5], c[4]), 'flag': flag})
    return picks


def prop_pick(row, mult, injured_flag):
    """Choose the better side of a prop row from the model numbers; returns a pick dict or None."""
    pm = row['model']
    options = []
    for side, p, price in (('over', pm['p_over'], row['over']), ('under', pm['p_under'], row['under'])):
        kind = 'priced' if price is not None else 'assumed'
        pr = price if price is not None else -110
        options.append((p - impl(pr), side, p, pr, kind))
    edge, side, p, pr, kind = max(options)
    if edge < 0.02:
        return None
    n, minavg = pm['n'], pm['min_avg'] or 0
    same_hit = (pm['hit_l10'] / pm['n_l10'] >= 0.6) if side == 'over' else (pm['hit_l10'] / pm['n_l10'] <= 0.4) if pm['n_l10'] else False
    if n >= 12 and minavg >= 18 and edge >= 0.04 and same_hit and not injured_flag and kind == 'priced':
        conf = 'High'
    elif n >= 8 and edge >= 0.025 and not injured_flag:
        conf = 'Medium'
    else:
        conf = 'Low'
    if abs(mult - 1.0) > 0.05 and conf == 'High':
        conf = 'Medium'
    if n < 8:
        conf = 'Low'
    flag = ''
    if edge > 0.12:
        conf, flag = 'Low', 'Unusually large edge. Check the line, the player\'s role and injury news before trusting it.'
    return {'type': 'prop', 'side': side, 'label': '%s %s %g %s' % (row['player'], side, row['line'], STAT_NAMES.get(row['stat'], row['stat']).lower()), 'line': row['line'], 'odds': pr,
            'est': r3(p), 'implied': r3(impl(pr)), 'edge': r3(edge), 'kind': kind, 'conf': conf, 'units': kelly_units(p, pr), 'player': row['player'], 'stat': row['stat'],
            'pid': row['pid'], 'mean': pm['mean'], 'hit_l10': '%d/%d' % (pm['hit_l10'], pm['n_l10']), 'flag': flag}


def rank_key(p):
    tier = {'priced': 2, 'assumed': 1}.get(p['kind'], 0)
    return (tier, (p['edge'] or 0) * conf_weight(p['conf']))


# ----------------------------------------------------------------------------- grading
def grade_history(history, games):
    changed = 0
    for h in history:
        if h.get('result'):
            continue
        g = games.get(h['game_id'])
        if not g or g['home'].get('pts') is None:
            continue
        res = None
        if h['type'] == 'prop':
            row = next((p for p in g.get('players') or [] if p['id'] == h.get('pid')), None)
            if not row:
                res = 'V'
            else:
                v = STAT_FN[h['stat']](row)
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


# ----------------------------------------------------------------------------- main
def team_block(ev_team, ratings, logs, injuries, inj_ok, today, name_to_abbr):
    a = ev_team['abbr']
    rt = ratings.get(a)
    rest_txt, b2b, rest_days = rest_info(rt, today)
    roster = [L for L in logs.values() if L['team'] == a]
    roster.sort(key=lambda L: -mean(g['min'] for g in L['games']))
    rot = []
    for L in roster[:8]:
        gs = L['games']
        rot.append({'name': L['name'], 'min': r1(mean(g['min'] for g in gs)), 'pts': r1(mean(g['pts'] for g in gs)), 'reb': r1(mean(g['reb'] for g in gs)),
                    'ast': r1(mean(g['ast'] for g in gs)), 'tpm': r1(mean(g['tpm'] for g in gs)), 'pts_l5': r1(mean(g['pts'] for g in gs[-5:])), 'gp': len(gs)})
    inj = []
    idx = {norm_name(L['name']): L for L in roster}
    for i in (injuries or {}).get(a, []):
        e = dict(i)
        L = idx.get(norm_name(i['name']))
        if L:
            gs = L['games']
            e['avg'] = {'min': r1(mean(g['min'] for g in gs)), 'pts': r1(mean(g['pts'] for g in gs)), 'reb': r1(mean(g['reb'] for g in gs)), 'ast': r1(mean(g['ast'] for g in gs))}
        inj.append(e)
    return {'abbr': a, 'name': ev_team['name'], 'record': ev_team['record'] or ('%d-%d' % (rt['w'], rt['l']) if rt else ''), 'rest': rest_txt, 'b2b': b2b,
            'ratings': rt, 'rotation': rot, 'injuries': inj, 'injuries_ok': inj_ok}


def main(argv):
    only_data = '--no-llm' in argv
    global API_KEY
    if only_data:
        API_KEY = ''
    today = today_ct()
    started = dt.datetime.now(dt.timezone.utc)
    print('EpLocks run for', today, flush=True)

    cache, slate_events = update_games(today)
    games = {k: v for k, v in cache['games'].items() if v.get('date') and v['date'] <= today.isoformat()}
    ratings, lg_ortg, lg_pace = build_ratings(games) if games else ({}, 112.0, 99.0)
    logs, name_index = build_player_logs(games)
    src('ratings', bool(ratings), '%d teams rated from %d games' % (len(ratings), len(games)))

    name_to_abbr = {}
    for ev in slate_events:
        for s in ('home', 'away'):
            name_to_abbr[ev[s]['name'].lower()] = ev[s]['abbr']
    for g in games.values():
        for s in ('home', 'away'):
            name_to_abbr.setdefault(g[s]['name'].lower(), g[s]['ab'])
    injuries = fetch_injuries(name_to_abbr)
    inj_ok = injuries is not None

    # grade finished picks first
    hist_path = os.path.join(OUT, 'history.json')
    history = jload(hist_path, [])
    ch = grade_history(history, games)
    if ch:
        note('Graded %d finished picks' % ch)

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
    inj_names_by_team = {t: {norm_name(i['name']): i['status'].lower() for i in lst} for t, lst in (injuries or {}).items()}
    lg_total = 2 * lg_ortg * lg_pace / 100.0
    for ev in sorted(slate_events, key=lambda e: e.get('utc') or ''):
        if ev['state'] != 'pre' and ev['id'] in prev_games:
            out_games.append(prev_games[ev['id']])
            all_picks.extend(prev_games[ev['id']].get('pick_objs', []))
            continue
        home = team_block(ev['home'], ratings, logs, injuries, inj_ok, today, name_to_abbr)
        away = team_block(ev['away'], ratings, logs, injuries, inj_ok, today, name_to_abbr)
        mo = project(home['abbr'], away['abbr'], ratings, lg_ortg, lg_pace, today)
        min_n = min((home['ratings'] or {}).get('n', 0), (away['ratings'] or {}).get('n', 0))
        mk = ev['market']
        blend = None
        if mo:
            w = 0.5 * min_n / (min_n + 12.0) if mk else 1.0
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
             'model': {k: r1(v) for k, v in mo.items()} if mo else None, 'blend': blend, 'h2h': h2h}
        g['model_raw'] = mo
        # props for this game
        rows = []
        if mo:
            env = clamp((((blend['total'] if blend else mo['total'])) / lg_total) ** 0.5, 0.94, 1.06) if lg_total else 1.0
            for p in routed:
                if p['team'] not in (home['abbr'], away['abbr']):
                    continue
                pm = prop_model(logs[p['pid']], p['stat'], p['line'], env=env)
                if not pm:
                    unmatched.append(dict(p, why='Fewer than 3 games of data for this player'))
                    continue
                opp = away['abbr'] if p['team'] == home['abbr'] else home['abbr']
                st = inj_names_by_team.get(p['team'], {}).get(norm_name(p['name']), '')
                rows.append(dict(p, model=pm, vs_opp=vs_opp(logs[p['pid']], p['stat'], opp), injury=st, env=r3(env)))
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

        # Claude
        analysis, adjust_m, adjust_t, pnotes = None, 0.0, 0.0, {}
        if API_KEY and mo and ev['state'] == 'pre':
            txt = call_claude(packet(g) + '\n\nInjuries loaded: %s. Return the JSON object now.' % ('yes' if inj_ok else 'NO'), SYSTEM, max_tokens=3500)
            js = extract_json(txt) if txt else None
            if isinstance(js, dict):
                cap_m, cap_t = (2.0, 5.0) if mk else (4.0, 8.0)
                adjust_m, adjust_t = clamp(num_or(js.get('margin_adj')), -cap_m, cap_m), clamp(num_or(js.get('total_adj')), -cap_t, cap_t)
                a = js.get('analysis') if isinstance(js.get('analysis'), dict) else {}
                def lst(x):
                    return [str(i)[:300] for i in x][:6] if isinstance(x, list) else []
                analysis = {'lean': str(a.get('lean') or '')[:300], 'why': lst(a.get('why')), 'risks': lst(a.get('risks')), 'matchups': lst(a.get('matchups')), 'verify': lst(a.get('verify'))}
                for pn in js.get('props') or []:
                    if isinstance(pn, dict) and pn.get('id'):
                        pnotes[str(pn['id'])] = {'mult': clamp(num_or(pn.get('mult'), 1.0), 0.9, 1.1), 'skip': bool(pn.get('skip')), 'note': str(pn.get('note') or '')[:240]}
            else:
                note('No usable analysis for %s @ %s' % (away['abbr'], home['abbr']))
        g['analysis'] = analysis
        g['adjust'] = {'margin': r1(adjust_m), 'total': r1(adjust_t)}

        picks = []
        if blend and ev['state'] == 'pre':
            fm = blend['margin'] + adjust_m
            ft = blend['total'] + adjust_t if blend['total'] is not None else None
            g['final'] = {'margin': r1(fm), 'total': r1(ft), 'home_win': r3(phi(fm / SD_MARGIN))}
            if not mo:
                g['final']['note'] = 'No team ratings yet, so this is the market number only.'
            uncertain = any(i['status'].lower() in ('questionable', 'doubtful', 'day-to-day', 'game time decision', 'gtd') and (i.get('avg') or {}).get('min', 0) >= 24
                            for t in (home, away) for i in t['injuries'])
            picks = game_picks(g, fm, ft, min_n, inj_ok, uncertain, abs(adjust_m) + abs(adjust_t) / 3.0)
            if not mk:
                g['final']['note'] = 'No market line loaded, so no bet can be priced for this game.'
        for r in rows:
            pn = pnotes.get(r['id'], {})
            mult = pn.get('mult', 1.0)
            if mult != 1.0:
                r['model'] = prop_model(logs[r['pid']], r['stat'], r['line'], env=r['env'], mult=mult)
            r['mult'], r['note'] = mult, pn.get('note', '')
            r['skipped'] = bool(pn.get('skip'))
            if ev['state'] != 'pre' or r['skipped'] or r['injury'] in BLOCK_STATUS:
                r['skipped'] = r['skipped'] or r['injury'] in BLOCK_STATUS
                continue
            pk = prop_pick(r, mult, bool(r['injury']))
            if pk:
                pk['reason'] = r['note']
                picks.append(pk)
        for i, p in enumerate(picks):
            p['id'] = '%s-%s-%d' % (today.isoformat(), ev['id'], i)
            p.update({'game_id': ev['id'], 'away': away['abbr'], 'home': home['abbr'], 'date': today.isoformat()})
        g['pick_objs'] = picks
        g['picks'] = [p['id'] for p in picks]
        g['dq'] = {'min_games': min_n, 'injuries_loaded': inj_ok, 'market_loaded': bool(mk), 'analysis': bool(analysis)}
        for r in rows:
            r.pop('raw', None)
        out_games.append(g)
        all_picks.extend(picks)

    all_picks.sort(key=lambda p: rank_key(p), reverse=True)

    # history: replace today's picks for games still 'pre', keep the rest
    today_s = today.isoformat()
    pre_ids = {g['id'] for g in out_games if g.get('state') == 'pre'}
    history = [h for h in history if not (h['date'] == today_s and h['game_id'] in pre_ids and not h.get('result'))]
    have = {h['id'] for h in history}
    for p in all_picks:
        if p['id'] not in have and p['game_id'] in pre_ids:
            history.append({k: p[k] for k in ('id', 'date', 'game_id', 'away', 'home', 'type', 'side', 'label', 'line', 'odds', 'est', 'edge', 'kind', 'conf', 'units') if k in p}
                           | ({'player': p['player'], 'stat': p['stat'], 'pid': p['pid']} if p['type'] == 'prop' else {}))
    jsave(hist_path, history)

    league = {'ortg': r1(lg_ortg), 'pace': r1(lg_pace), 'games': len(games), 'avg_total': r1(lg_total)}
    src('anthropic', bool(API_KEY) and STATUS['sources'].get('anthropic', {}).get('ok', True), 'model %s' % MODEL if API_KEY else 'no API key set, so there is no written analysis')
    result = {
        'prop_source': 'odds_api' if auto else ('props.txt' if lines else 'none'),
        'generated_at': started.strftime('%Y-%m-%dT%H:%M:%SZ'), 'for_date': today_s, 'model': MODEL if API_KEY else None, 'league': league,
        'params': {'home_edge': HCA, 'sd_margin': SD_MARGIN, 'sd_total': SD_TOTAL, 'b2b_points': B2B_PTS},
        'status': STATUS, 'games': [{k: v for k, v in g.items() if k not in ('model_raw',)} for g in out_games],
        'picks': [dict(p, label=p['label']) for p in all_picks], 'props_unmatched': unmatched[:60], 'props_not_playing': [dict(p) for p in not_playing][:60],
        'props_lines': len(lines) + (len(auto) if auto else 0)}
    # drop heavy duplicates
    for g in result['games']:
        for t in ('away', 'home'):
            g[t].pop('injuries_ok', None)
    if not STATUS['sources'].get('espn_games', {}).get('ok', False) and prev.get('games'):
        # Game data failed today: keep the last good slate on the site instead of wiping it, and show the failure.
        note('Game data failed, so the site is still showing the previous run.')
        prev['status'] = STATUS
        jsave(os.path.join(OUT, 'latest.json'), prev)
        print('Game data failed; kept previous output', flush=True)
        return 0
    jsave(os.path.join(OUT, 'latest.json'), result)
    print('Wrote %d games, %d picks' % (len(out_games), len(all_picks)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
