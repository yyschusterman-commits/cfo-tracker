"""CFO insider-buy forward tracker. Run once each evening (after 10pm ET).
Usage: python3 tracker.py <data.json> <config.json> [--today YYYY-MM-DD]
Rules (same as the backtest):
  - OpenInsider open-market purchases, each row >= $25k, grouped by ticker + filing day
  - at least one buyer's title is CFO / Chief Financial Officer; trade date <= 30 days before filing
  - CFO-day average price >= $2
  - score = average of percentile(buy value / market cap) and percentile(60-day volatility),
    percentiles vs 2013-2020 CFO buys; need >= 0.33
  - routine: check once each evening -> buy at the NEXT trading day's open with a limit at 3% above the
    insider price. If the open is above the limit it counts as skipped (conservative: no later fill).
  - sell at the close 63 trading days after entry. Compare with SPY over the exact same open->close.
"""
import sys, json, io, time, datetime as dt, subprocess
import numpy as np, pandas as pd, yfinance as yf
from zoneinfo import ZoneInfo

DATA, CFG = sys.argv[1], sys.argv[2]
cfg = json.load(open(CFG))
state = json.load(open(DATA)) if __import__('os').path.exists(DATA) else {"signals": [], "runs": []}
ET = ZoneInfo('America/New_York')
today = pd.Timestamp(sys.argv[sys.argv.index('--today')+1]) if '--today' in sys.argv else pd.Timestamp(dt.datetime.now(ET).date())
errors = []

BASE = ("http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=-1&fdr={a}+-+{b}&td=0&tdr=&fdlyl=&fdlyh=&daysago="
        "&xp=1&vl=25&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
        "&sortcol=0&cnt=5000&page=1")

def scrape(d1, d2):
    a, b = d1.strftime('%m%%2F%d%%2F%Y'), d2.strftime('%m%%2F%d%%2F%Y')
    for k in range(4):
        r = subprocess.run(["curl", "-s", "-m", "90", "-A", "Mozilla/5.0", BASE.format(a=a, b=b)], capture_output=True, text=True)
        try:
            return pd.read_html(io.StringIO(r.stdout), attrs={'class': 'tinytable'})[0]
        except Exception as e:
            if 'No tables' in str(e): return pd.DataFrame()
            time.sleep(10)
    raise RuntimeError('OpenInsider scrape failed')

def num(s): return pd.to_numeric(pd.Series(s).astype(str).str.replace(r'[\$,+]', '', regex=True), errors='coerce')

def is_cfo(t):
    t = str(t).lower(); return ('cfo' in t) or ('chief financial' in t)

def pct(ref, x): ref = np.asarray(ref); return float(np.searchsorted(ref, x) / len(ref))

PX = {}
def hist(t, adjusted=True, start=None):
    key = (t, adjusted)
    if key not in PX:
        d = yf.download(t.replace('.', '-'), start=start or (today - pd.Timedelta(days=200 + LB)).strftime('%Y-%m-%d'),
                        end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), auto_adjust=adjusted, progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = [c[0] for c in d.columns]
        PX[key] = d.dropna(subset=['Close'])
    return PX[key]

# ---------- 1. new CFO buy events ----------
known = {(s['ticker'], s['filing_day']) for s in state['signals']}
LB = int(sys.argv[sys.argv.index('--lookback')+1]) if '--lookback' in sys.argv else 60
look_from = max(pd.Timestamp(cfg['start']), today - pd.Timedelta(days=LB))
try:
    raw = scrape(look_from, today)
except Exception as e:
    raw = pd.DataFrame(); errors.append(f'scrape: {e}')
if len(raw):
    raw.columns = [str(c).replace('\xa0', ' ').strip() for c in raw.columns]
    col = lambda name: [c for c in raw.columns if c.lower().startswith(name)][0]
    df = pd.DataFrame({'fdt': pd.to_datetime(raw[col('filing')]), 'tdate': pd.to_datetime(raw[col('trade d')]),
                       'ticker': raw[col('ticker')].astype(str).str.strip(), 'company': raw[col('company')],
                       'insider': raw[col('insider')], 'title': raw[col('title')], 'ttype': raw[col('trade type')],
                       'price': num(raw[col('price')]).values, 'value': num(raw[col('value')]).values})
    df = df[df.ttype.astype(str).str.startswith('P') & (df.value >= cfg['min_value'])]
    df['lag'] = (df.fdt.dt.normalize() - df.tdate).dt.days
    df = df[(df.lag >= 0) & (df.lag <= 30)]
    df['fday'] = df.fdt.dt.normalize(); df['cfo'] = df.title.map(is_cfo)
    ev = df.groupby(['ticker', 'fday']).agg(company=('company', 'first'), value=('value', 'sum'), price=('price', 'mean'),
                                            cfo=('cfo', 'max'), fdt=('fdt', 'min'),
                                            cfo_names=('insider', lambda s: ', '.join(sorted(set(s))))).reset_index()
    ev = ev[ev.cfo & (ev.fday >= pd.Timestamp(cfg['start']))]
    now_et = dt.datetime.now(ET)
    if '--today' not in sys.argv and (now_et.hour, now_et.minute) < (22, 5):   # SEC takes filings until 10pm ET:
        ev = ev[ev.fday < today]                                              # leave today's for tonight's run
    for _, e in ev.iterrows():
        k = (e.ticker, e.fday.strftime('%Y-%m-%d'))
        if k in known: continue
        s = dict(ticker=e.ticker, company=e.company, filing_day=k[1], filing_time=e.fdt.strftime('%Y-%m-%d %H:%M'),
                 insiders=e.cfo_names, value=round(float(e.value)), insider_price=round(float(e.price), 4), status='new', note='')
        state['signals'].append(s); known.add(k)

# ---------- 2. score new signals ----------
for s in state['signals']:
    if s['status'] != 'new': continue
    try:
        if s['insider_price'] < cfg['min_price']:
            s['status'] = 'skipped'; s['note'] = 'price under $2'; continue
        h = hist(s['ticker']); h = h[h.index <= pd.Timestamp(s['filing_day'])]
        if len(h) < 61: s['status'] = 'skipped'; s['note'] = 'not enough price history'; continue
        lr = np.log(h.Close).diff().dropna().values[-60:]
        vol60 = float(np.std(lr) * np.sqrt(252)); adv20 = float((h.Close * h.Volume).iloc[-20:].mean())
        shares = None
        try: shares = yf.Ticker(s['ticker'].replace('.', '-')).fast_info.get('shares')
        except Exception: pass
        if not shares:
            try: shares = yf.Ticker(s['ticker'].replace('.', '-')).info.get('sharesOutstanding')
            except Exception: shares = None
        v2m = s['value'] / (shares * s['insider_price']) if shares else None
        p1 = pct(cfg['ref_v2m'], v2m) if v2m is not None else 0.5
        p2 = pct(cfg['ref_vol60'], vol60)
        s.update(vol60=round(vol60, 3), adv20=round(adv20), mcap=round(shares * s['insider_price']) if shares else None,
                 buy_pct_of_company=round(v2m * 100, 4) if v2m is not None else None, score=round((p1 + p2) / 2 * 100))
        if s['score'] < cfg['score_min'] * 100:
            s['status'] = 'skipped'; s['note'] = f"score {s['score']} under 33"
        else:
            s['status'] = 'waiting'; s['limit'] = round(s['insider_price'] * (1 + cfg['limit']), 4)
    except Exception as ex:
        errors.append(f"score {s['ticker']}: {ex}")

# ---------- 3. entries (next trading day's open after the filing day) ----------
spy_adj = hist('SPY', True, '2026-01-01')
for s in state['signals']:
    if s['status'] != 'waiting': continue
    try:
        raw_h = hist(s['ticker'], False); adj_h = hist(s['ticker'], True)
        after = raw_h[raw_h.index > pd.Timestamp(s['filing_day'])]
        if not len(after): continue                       # next open hasn't happened yet
        d0 = after.index[0]; op = float(after.Open.iloc[0])
        s['entry_date'] = d0.strftime('%Y-%m-%d'); s['open_price'] = round(op, 4)
        if op > s['limit']:
            s['status'] = 'skipped'; s['note'] = f"opened {op / s['insider_price'] * 100 - 100:+.1f}% vs CFO price (limit not filled)"
        else:
            s['status'] = 'open'
    except Exception as ex:
        errors.append(f"entry {s['ticker']}: {ex}")

# ---------- 4. exits (close 63 trading days after entry) and live marks ----------
def cost_est(adv):  # same round-trip cost model as the backtest (+0.5% extra slippage)
    c = 0.002 if adv >= 2e7 else (0.006 if adv >= 2e6 else 0.015)
    return c + 0.005
for s in state['signals']:
    if s['status'] != 'open': continue
    try:
        adj_h = hist(s['ticker'], True, s['entry_date'])
        a = adj_h[adj_h.index >= pd.Timestamp(s['entry_date'])]
        sp = spy_adj[spy_adj.index >= pd.Timestamp(s['entry_date'])]
        if not len(a) or not len(sp): continue
        n = len(a) - 1; done = n >= cfg['hold']
        stale = (today - a.index[-1]).days > 10                  # delisted / no data: close at last price
        i = min(cfg['hold'], n)
        r = float(a.Close.iloc[i] / a.Open.iloc[0] - 1)
        j = min(len(sp) - 1, sp.index.searchsorted(a.index[i], side='right') - 1)
        rs = float(sp.Close.iloc[j] / sp.Open.iloc[0] - 1)
        s.update(days_held=int(i), ret=round(r * 100, 2), spy_ret=round(rs * 100, 2), excess=round((r - rs) * 100, 2),
                 net_excess=round((r - rs - cost_est(s.get('adv20', 1e6))) * 100, 2))
        if done or stale:
            s['status'] = 'closed'; s['exit_date'] = a.index[i].strftime('%Y-%m-%d')
            if stale and not done: s['note'] = 'data stopped (delisted?) - closed at last price'
    except Exception as ex:
        errors.append(f"exit {s['ticker']}: {ex}")


# ---------- 5. paper account ($100k, same rules as the backtest) ----------
# Rebuilt from scratch every night from the signals above + Yahoo prices, so a missed night changes nothing.
#  - spare cash sits in SPY (whole shares, 0.01% cost each way)
#  - each filled signal: buy at its entry-day open, 10% of the account (value at the previous close),
#    paid for by selling SPY; max 10 stocks at once; if all 10 are full the signal is skipped
#  - several signals on one day with too few free slots: highest score first
#  - sell at the close 63 trading days later (same day the tracker closes it); cash goes back into SPY
#  - dividends paid as cash on the ex-date, then swept into SPY; trading costs = backtest cost model
def paper_account():
    START_CASH, SLOTS, FRAC, SPYC = 100000.0, 10, 0.10, 0.0001
    def dl(t, start):
        d = yf.download(t.replace('.', '-'), start=start, end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                        auto_adjust=False, actions=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex): d.columns = [c[0] for c in d.columns]
        d = d.dropna(subset=['Close'])
        for c in ('Dividends', 'Stock Splits'):
            if c not in d: d[c] = 0.0
        return d
    spy = dl('SPY', (pd.Timestamp(cfg['start']) - pd.Timedelta(days=7)).strftime('%Y-%m-%d'))
    days = spy.index[spy.index > pd.Timestamp(cfg['start'])]
    if not len(days): return {'start': None, 'note': 'starts at the next market open'}
    spy_adj = hist('SPY', True, '2026-01-01'); spy_adj = spy_adj[spy_adj.index >= days[0]]
    sigs = [s for s in state['signals'] if s['status'] in ('open', 'closed') and s.get('entry_date')]
    sigs.sort(key=lambda s: (s['entry_date'], -(s.get('score') or 0), s.get('filing_time', '')))
    px = {}
    for s in sigs:
        if s['ticker'] not in px: px[s['ticker']] = dl(s['ticker'], s['entry_date'])
    cash, spy_sh, pos, log, hist_rows, peak, mdd = START_CASH, 0, [], [], [], START_CASH, 0.0
    taken = set(); trades = []
    def last_close(t, d):
        h = px[t]; h = h[h.index <= d]
        return float(h.Close.iloc[-1]) if len(h) else None
    def add(d, act, tic='', sh=0, price=None, amt=0.0, note=''):
        log.append(dict(date=d.strftime('%Y-%m-%d'), action=act, ticker=tic, shares=int(sh),
                        price=None if price is None else round(price, 4), amount=round(amt, 2), cash=round(cash, 2), note=note))
    prev_val = START_CASH
    for k, d in enumerate(days):
        so, sc = float(spy.Open.loc[d]), float(spy.Close.loc[d])
        # ---- open: first day buys SPY; then new entries funded by selling SPY
        if k == 0:
            n = int(cash // (so * (1 + SPYC))); cash -= n * so * (1 + SPYC); spy_sh += n
            add(d, 'SPY BUY', 'SPY', n, so, -n * so * (1 + SPYC), 'start: $100,000 into the S&P')
        for s in [s for s in sigs if s['entry_date'] == d.strftime('%Y-%m-%d')]:
            key = (s['ticker'], s['filing_day'])
            if len(pos) >= SLOTS:
                add(d, 'SKIP', s['ticker'], note='all 10 slots full'); continue
            h = px[s['ticker']]
            if d not in h.index: add(d, 'SKIP', s['ticker'], note='no price that day'); continue
            half = cost_est(s.get('adv20') or 1e6) / 2
            p = float(h.Open.loc[d]) * (1 + half)
            budget = min(FRAC * prev_val, cash + spy_sh * so * (1 - SPYC))
            n = int(budget // p)
            if n < 1: add(d, 'SKIP', s['ticker'], note='not enough cash'); continue
            need = n * p - cash
            if need > 0:
                m = min(spy_sh, int(np.ceil(need / (so * (1 - SPYC)))))
                spy_sh -= m; cash += m * so * (1 - SPYC)
                add(d, 'SPY SELL', 'SPY', m, so, m * so * (1 - SPYC), f"to pay for {s['ticker']}")
            cash -= n * p
            ent = h.index.get_loc(d); due = h.index[ent + cfg['hold']] if ent + cfg['hold'] < len(h) else None
            exit_d = pd.Timestamp(s['exit_date']) if s['status'] == 'closed' and s.get('exit_date') else due
            pos.append(dict(ticker=s['ticker'], shares=n, entry=d, cost=n * p, buy_px=p, half=half, exit=exit_d, score=s.get('score')))
            add(d, 'BUY', s['ticker'], n, p, -n * p, f"score {s.get('score')}, {n * p / prev_val * 100:.1f}% of account")
            taken.add(key)
        # ---- close: dividends, 63-day sells, sweep cash into SPY
        dv = float(spy.Dividends.loc[d])
        if dv > 0 and spy_sh: cash += spy_sh * dv; add(d, 'DIVIDEND', 'SPY', spy_sh, dv, spy_sh * dv)
        for q in pos:
            h = px[q['ticker']]
            if d in h.index and float(h.Dividends.loc[d]) > 0:
                a = q['shares'] * float(h.Dividends.loc[d]); cash += a; q['divs'] = q.get('divs', 0) + a
                add(d, 'DIVIDEND', q['ticker'], q['shares'], float(h.Dividends.loc[d]), a)
        for q in list(pos):
            if q['exit'] is not None and d >= q['exit']:
                c = last_close(q['ticker'], d); p = c * (1 - q['half']); a = q['shares'] * p
                cash += a; pos.remove(q)
                r = (a + q.get('divs', 0)) / q['cost'] - 1
                trades.append(dict(ticker=q['ticker'], bought=q['entry'].strftime('%Y-%m-%d'), sold=d.strftime('%Y-%m-%d'),
                                   shares=q['shares'], buy_px=round(q['buy_px'], 4), sell_px=round(p, 4), ret=round(r * 100, 2),
                                   profit=round(a + q.get('divs', 0) - q['cost'], 2)))
                add(d, 'SELL', q['ticker'], q['shares'], p, a, f"after {cfg['hold']} trading days: {r * 100:+.1f}%")
        n = int(cash // (sc * (1 + SPYC)))
        if n > 0: cash -= n * sc * (1 + SPYC); spy_sh += n; add(d, 'SPY BUY', 'SPY', n, sc, -n * sc * (1 + SPYC), 'park spare cash')
        stocks = sum(q['shares'] * (last_close(q['ticker'], d) or 0) for q in pos)
        val = cash + spy_sh * sc + stocks
        prev_val = val; peak = max(peak, val); mdd = min(mdd, val / peak - 1)
        bench = START_CASH * float(spy_adj.Close.loc[d]) / float(spy_adj.Open.iloc[0]) if d in spy_adj.index else None
        hist_rows.append(dict(date=d.strftime('%Y-%m-%d'), value=round(val, 2), sp=None if bench is None else round(bench, 2),
                              in_stocks=round(stocks / val * 100, 1)))
    d = days[-1]
    holdings = []
    for q in pos:
        c = last_close(q['ticker'], d); v = q['shares'] * c
        held = int((px[q['ticker']].index[(px[q['ticker']].index >= q['entry']) & (px[q['ticker']].index <= d)]).size - 1)
        sell = q['exit'].strftime('%Y-%m-%d') if q['exit'] is not None else \
            'about ' + (q['entry'] + pd.offsets.BDay(cfg['hold'] + 2)).strftime('%Y-%m-%d')
        holdings.append(dict(ticker=q['ticker'], shares=q['shares'], bought=q['entry'].strftime('%Y-%m-%d'),
                             buy_px=round(q['buy_px'], 4), last_px=round(c, 4), value=round(v, 2),
                             ret=round((v + q.get('divs', 0)) / q['cost'] * 100 - 100, 2), days_held=held, sell_on=sell))
    last = hist_rows[-1]
    return dict(start=days[0].strftime('%Y-%m-%d'), start_value=START_CASH, as_of=last['date'], value=last['value'],
                ret=round(last['value'] / START_CASH * 100 - 100, 2), sp_value=last['sp'],
                sp_ret=None if last['sp'] is None else round(last['sp'] / START_CASH * 100 - 100, 2),
                cash=round(cash, 2), spy_shares=int(spy_sh), spy_value=round(spy_sh * float(spy.Close.loc[d]), 2),
                in_stocks=last['in_stocks'], max_drop=round(mdd * 100, 2), slots_used=len(pos), slots=SLOTS,
                holdings=holdings, trades=trades, log=log, history=hist_rows)
try:
    state['paper'] = paper_account()
except Exception as ex:
    errors.append(f'paper account: {ex}')

state['runs'] = (state.get('runs', []) + [dict(at=dt.datetime.now(ET).strftime('%Y-%m-%d %H:%M ET'), for_day=today.strftime('%Y-%m-%d'),
                 new=sum(1 for s in state['signals'] if s.get('filing_day', '') >= (today - pd.Timedelta(days=10)).strftime('%Y-%m-%d')),
                 errors=errors)])[-60:]
state['updated'] = dt.datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')
state['signals'].sort(key=lambda s: (s['filing_day'], s['ticker']), reverse=True)
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o): return None
    return o
json.dump(clean(state), open(DATA, 'w'), indent=1, allow_nan=False)
c = pd.Series([s['status'] for s in state['signals']]).value_counts().to_dict() if state['signals'] else {}
print('updated', state['updated'], c, 'errors:', errors)
