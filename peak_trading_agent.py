# peak_trading_agent.py
# ------------------------------------------------------------
# One-file, copy-paste scaffold of a peak-performance crypto
# trading agent with modern research & execution features.
#
# Quickstart:
#   pip install pandas numpy scikit-learn matplotlib pyyaml
#   python peak_trading_agent.py
#
# Outputs: ./reports/ per-market CSVs + simple HTML reports,
#          portfolio CSV + report.
#
# NOTE: This is a research scaffold. Before live use:
#  - connect real data (ccxt/websockets),
#  - calibrate slippage/fees per venue/hour,
#  - harden risk & alerts, add promotion gate, CI tests.
# ------------------------------------------------------------

from __future__ import annotations
import os, json, math, time, random, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Optional ML imports (auto-fallback if missing)
_SK = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.covariance import LedoitWolf
except Exception:
    _SK = False
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Basic indicators
# -------------------------
def ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(3, n//3)).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100/(1+rs)

def _tr(h, l, c):
    pc = c.shift(1)
    a = (h - l).abs()
    b = (h - pc).abs()
    c2 = (l - pc).abs()
    return pd.concat([a, b, c2], axis=1).max(axis=1)

def atr(h, l, c, n: int = 14) -> pd.Series:
    return _tr(h, l, c).rolling(n, min_periods=n).mean()

def adx(h, l, c, n: int = 14) -> pd.Series:
    up = h.diff()
    dn = -l.diff()
    plus_dm  = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0.0)
    trv = _tr(h,l,c)
    atrn = trv.rolling(n, min_periods=n).sum()/n
    plus  = 100*(pd.Series(plus_dm, index=h.index).rolling(n, min_periods=n).sum()/n)/(atrn + 1e-12)
    minus = 100*(pd.Series(minus_dm, index=h.index).rolling(n, min_periods=n).sum()/n)/(atrn + 1e-12)
    dx = 100*((plus-minus).abs()/((plus+minus)+1e-12))
    return dx.rolling(n, min_periods=n).mean()

def donchian(h, l, n: int = 20) -> Tuple[pd.Series, pd.Series]:
    return h.rolling(n, min_periods=n).max(), l.rolling(n, min_periods=n).min()

def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    m = ma(s, n); sd = s.rolling(n, min_periods=n).std()
    return m, m + k*sd, m - k*sd

# -------------------------
# Data helpers
# -------------------------
def make_synth(n=2600, start="2023-01-01", mu=0.03, sigma=0.9, price0=20000) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="H")
    r = np.random.normal(mu/100.0, sigma/100.0, size=n)
    p = [price0]
    for x in r[1:]:
        p.append(p[-1]*(1+x))
    p = np.array(p)
    high = p*(1+np.abs(np.random.normal(0.001,0.0006,n)))
    low  = p*(1-np.abs(np.random.normal(0.001,0.0006,n)))
    op   = p*(1+np.random.normal(0,0.0002,n))
    vol  = np.random.randint(500, 4000, n)
    return pd.DataFrame({"timestamp":idx,"open":op,"high":high,"low":low,"close":p,"volume":vol})

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    need = {'open','high','low','close','volume'}
    assert need.issubset(df.columns), f"CSV missing: {need - set(df.columns)}"
    return df.set_index('timestamp')

def check_sla(df: pd.DataFrame, freq='H') -> bool:
    rng = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return len(rng.difference(df.index)) == 0

# -------------------------
# Feature Store
# -------------------------
class FeatureStore:
    def __init__(self, zwin: int = 500):
        self.zwin = zwin

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['atr14'] = atr(out['high'], out['low'], out['close'], 14)
        out['rsi2']  = rsi(out['close'], 2)
        out['ma_fast'] = ma(out['close'], 20)
        out['ma_slow'] = ma(out['close'], 200)
        out['adx14'] = adx(out['high'], out['low'], out['close'], 14)
        u, l = donchian(out['high'], out['low'], 20)
        out['donchian_upper'] = u
        out['donchian_lower'] = l
        m, ub, lb = bbands(out['close'], 20, 2.0)
        out['bb_mid'] = m; out['bb_up'] = ub; out['bb_dn'] = lb

        # simple microstructure placeholders; swap with real L2/trades live
        out['spread_bps'] = 2.0

        out['regime'] = np.where((out['close']>out['ma_slow']) & (out['adx14']>=20), 'Trending', 'Ranging')

        for col in ['close','atr14','rsi2','adx14']:
            roll = out[col].rolling(self.zwin, min_periods=50)
            out[col+'_z'] = (out[col] - roll.mean()) / (roll.std() + 1e-9)

        return out.dropna()

# -------------------------
# Strategies
# -------------------------
@dataclass
class Signal:
    side: str = "flat"
    reason: str = ""
    strength: float = 0.0

class DonchianBreakout20:
    name = "donchian_breakout_20"
    def __init__(self, params): self.n=int(params.get('donchian',20))
    def generate(self, row):
        u=row.get('donchian_upper'); c=row.get('close')
        if u is None or c is None: return Signal()
        if c>u: return Signal("buy","breakout",0.7)
        return Signal()

class MACrossADX:
    name = "ma_cross_adx"
    def __init__(self, params): self.f=int(params.get('fast',20)); self.s=int(params.get('slow',100)); self.ax=float(params.get('adx_min',18))
    def generate(self, row):
        f=row.get('ma_fast'); s=row.get('ma_slow'); a=row.get('adx14')
        if None in (f,s,a): return Signal()
        if f>s and a>=self.ax: return Signal("buy","ma+adx",0.6)
        return Signal()

class RSI2MeanRevert:
    name = "rsi2_mean_revert"
    def __init__(self, params): self.buy=float(params.get('rsi_buy',10)); self.sell=float(params.get('rsi_sell',90))
    def generate(self, row):
        r=row.get('rsi2');
        if r is None: return Signal()
        if r < self.buy: return Signal("buy","rsi2_pullback",0.5)
        return Signal()

class BollingerRevert:
    name = "bollinger_revert"
    def __init__(self, params): self.n=int(params.get('n',20)); self.k=float(params.get('k',2.0))
    def generate(self, row):
        c=row.get('close'); lb=row.get('bb_dn')
        if None in (c,lb): return Signal()
        if c < lb: return Signal("buy","bb_revert",0.5)
        return Signal()

STRATS = {
    "donchian_breakout_20": DonchianBreakout20,
    "ma_cross_adx": MACrossADX,
    "rsi2_mean_revert": RSI2MeanRevert,
    "bollinger_revert": BollingerRevert,
}

def load_strategies(overrides: Optional[Dict[str,dict]]=None) -> List:
    spec = [
        {"name":"donchian_breakout_20","params":{"donchian":20}},
        {"name":"ma_cross_adx","params":{"fast":20,"slow":100,"adx_min":18}},
        {"name":"rsi2_mean_revert","params":{"rsi_buy":10,"rsi_sell":90}},
        {"name":"bollinger_revert","params":{"n":20,"k":2.0}},
    ]
    out=[]
    for it in spec:
        name=it["name"]; params=dict(it["params"])
        if overrides and name in overrides: params.update(overrides[name])
        out.append(STRATS[name](params))
    return out

# -------------------------
# Triple-Barrier & Purged WFCV (scaffold)
# -------------------------
def triple_barrier(df: pd.DataFrame, up_k: float = 2.0, dn_k: float = 2.0, max_hold: int = 48, atr_col: str = 'atr14'):
    c,h,l,a = df['close'].values, df['high'].values, df['low'].values, df[atr_col].values
    n=len(df); y=np.zeros(n,dtype=int); t=np.full(n,-1,dtype=int)
    for i in range(n-1):
        if np.isnan(a[i]): continue
        up=c[i]+up_k*a[i]; dn=c[i]-dn_k*a[i]
        end=min(n-1,i+max_hold); win=0; when=end
        for k in range(i+1,end+1):
            if h[k]>=up: win=1; when=k; break
            if l[k]<=dn: win=-1; when=k; break
        y[i]=win; t[i]=when
    out=df.copy(); out['tb_label']=y; out['tb_horizon']=t; return out

@dataclass
class Fold:
    train_start: pd.Timestamp; train_end: pd.Timestamp
    valid_start: pd.Timestamp; valid_end: pd.Timestamp
    embargo: int

def purged_walk_forward(index: pd.DatetimeIndex, n_folds: int, min_train: int, horizon_bars: int, embargo_mult: float=1.0):
    n=len(index); fold_size=max(1,(n-min_train)//max(1,n_folds)); folds=[]; emb=int(max(1,horizon_bars*embargo_mult))
    for k in range(n_folds):
        te=min_train+k*fold_size; vs=te+emb; ve=min(n-1, vs+fold_size)
        if ve<=vs or vs<=te: break
        folds.append(Fold(index[0], index[te], index[vs], index[ve], emb))
    return folds

# -------------------------
# Meta-label model (optional)
# -------------------------
class MetaModel:
    def __init__(self):
        self.model=None; self.brier=None; self.auc=None
        if not _SK: print("[MetaModel] scikit-learn not available; meta-label gating disabled.")

    def fit(self, X: pd.DataFrame, y: pd.Series, method: str = "isotonic"):
        if not _SK: return
        base = LogisticRegression(max_iter=250)
        self.model = CalibratedClassifierCV(base, method=method, cv=3)
        self.model.fit(X, y)
        p = self.model.predict_proba(X)[:,1]
        self.brier = brier_score_loss(y, p); self.auc = roc_auc_score(y, p)

    def predict_proba(self, X: pd.DataFrame):
        if not _SK or self.model is None: return np.zeros(len(X))
        return self.model.predict_proba(X)[:,1]

# -------------------------
# Sizing: ATR + Half-Kelly (2% cap) + shrinkage
# -------------------------
def atr_sizing(equity: float, atr_value: float, k_atr: float, per_trade: float) -> Tuple[float,float,float]:
    stop_dist = max(1e-9, k_atr * atr_value); dollar_risk = equity * per_trade
    size = dollar_risk / stop_dist
    return size, stop_dist, dollar_risk

def fractional_kelly(p_win: float, r_win: float, r_loss: float, cap: float = 0.02) -> float:
    if r_win<=0 or r_loss<=0: return 0.0
    edge = p_win*r_win - (1-p_win)*r_loss
    denom = r_win*r_loss
    if denom<=0: return 0.0
    k = max(0.0, edge/denom)*0.5
    return min(cap, k)

def shrink_p(p_hat: float, n: int, alpha: float = 12.0):
    a = alpha; b = alpha
    return (p_hat*n + a) / (n + a + b)

# -------------------------
# Execution & Router
# -------------------------
def regime_path_priors(regime: str):
    return [('OHLC',0.6), ('OLHC',0.4)] if regime=='Trending' else [('OLHC',0.6), ('OHLC',0.4)]

def simulate_intrabar(open_, high, low, close, stop, tp, side: str, regime: str='Ranging'):
    def seq_vals(seq): return [open_, high, low, close] if seq=='OHLC' else [open_, low, high, close]
    prob_tp=0.0; prob_sl=0.0
    for seq,p in regime_path_priors(regime):
        s = seq_vals(seq); hit_tp=False; hit_sl=False
        for v in s:
            if side=='buy':
                if v>=tp: hit_tp=True; break
                if v<=stop: hit_sl=True; break
            else:
                if v<=tp: hit_tp=True; break
                if v>=stop: hit_sl=True; break
        if hit_tp and not hit_sl: prob_tp+=p
        elif hit_sl and not hit_tp: prob_sl+=p
        elif hit_tp and hit_sl:
            for v in s:
                if (side=='buy' and v>=tp) or (side!='buy' and v<=tp): prob_tp+=p; break
                if (side=='buy' and v<=stop) or (side!='buy' and v>=stop): prob_sl+=p; break
    return prob_tp, prob_sl

@dataclass
class OrderFill:
    filled: bool; entry: float; size: float; slip_bps: float; order_type: str

def place_order(side: str, price: float, size: float, spread_bps: float, urgency: float, order_type: Optional[str] = None) -> OrderFill:
    if order_type is None: order_type = 'taker' if urgency>0.6 else 'maker'
    if order_type=='taker':
        slip = np.random.uniform(0, max(1.0, spread_bps*1.2))
        px = price * (1 + (slip/10000.0) * (1 if side=='buy' else -1))
        return OrderFill(True, px, size, slip, 'taker')
    else:
        if np.random.rand() < 0.8:  # maker queue miss model (80% fill)
            return OrderFill(True, price, size, 0.0, 'maker')
        return OrderFill(False, price, 0.0, 0.0, 'maker')

def choose_route(cost_curves: dict, hour: int, urgency: float) -> Tuple[str,str]:
    best=None; best_cost=1e9
    for venue, spec in (cost_curves or {'LOCAL':{}}).items():
        maker = spec.get('maker_bps',{}).get(hour,2.0) + 0.5*spec.get('slip_bps',{}).get(hour,5.0)
        taker = spec.get('taker_bps',{}).get(hour,10.0) + spec.get('slip_bps',{}).get(hour,5.0)
        cand_type = 'taker' if urgency>0.6 else 'maker'
        cand = taker if cand_type=='taker' else maker
        if cand < best_cost: best_cost=cand; best=(venue,cand_type)
    return best or ('LOCAL','taker')

def fee_R(entry_price: float, stop_dist: float, bps: float) -> float:
    return (bps/10000.0) * (entry_price / max(1e-9, stop_dist))

def slip_R(slip_bps: float, entry_price: float, stop_dist: float) -> float:
    return (slip_bps/10000.0) * (entry_price / max(1e-9, stop_dist))

# -------------------------
# Portfolio allocator (shrinkage + risk parity/min-var)
# -------------------------
def shrink_cov(returns: pd.DataFrame) -> pd.DataFrame:
    if _SK and len(returns)>3 and returns.shape[1]>1:
        lw = LedoitWolf().fit(returns.values)
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return returns.cov()

def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    C=cov.values; n=C.shape[0]
    try:
        inv=np.linalg.pinv(C); ones=np.ones((n,1))
        w = inv@ones; w=w/(ones.T@inv@ones); w=w.flatten()
    except Exception:
        w = np.ones(n)/n
    return pd.Series(w, index=cov.index)

def risk_parity_weights(cov: pd.DataFrame, iters:int=100) -> pd.Series:
    n=cov.shape[0]; w=np.ones(n)/n
    for _ in range(iters):
        mrc = cov.values@w
        if (mrc<=0).any(): break
        w = 1.0/mrc; w = w/np.sum(w)
    return pd.Series(w, index=cov.index)

# -------------------------
# Reporting
# -------------------------
def kpis(df: pd.DataFrame) -> Dict[str,float]:
    wins = df[df['net_R']>0]['net_R'].sum()
    losses = -df[df['net_R']<0]['net_R'].sum()
    pf = (wins/losses) if losses>0 else 0.0
    hit = (df['net_R']>0).mean() if len(df)>0 else 0.0
    avgR = df['net_R'].mean() if len(df)>0 else 0.0
    peak = df['equity'].cummax() if 'equity' in df else (df['net_R'].cumsum().cummax())
    eq = df['equity'] if 'equity' in df else df['net_R'].cumsum()
    dd = (peak - eq).max() / max(1e-9, peak.max())
    cost_share = float((df.get('fee_R',0)+df.get('slip_R',0)).sum() / max(1e-9, df['pnl_R'].abs().sum())) if 'pnl_R' in df else 0.0
    return dict(pf=float(pf), hit=float(hit), avgR=float(avgR), maxDD=float(dd), cost_share=float(cost_share))

def write_html_report(out_dir: str, title: str, k: Dict, extras: Optional[str]=None):
    os.makedirs(out_dir, exist_ok=True)
    html = f"""<html><head><meta charset="utf-8"><title>{title}</title></head>
<body><h1>{title}</h1><h3>KPIs</h3><pre>{json.dumps(k, indent=2)}</pre>"""
    if extras: html += f"<h3>Notes</h3><pre>{extras}</pre>"
    html += "</body></html>"
    with open(os.path.join(out_dir, f"{title.replace(' ','_').lower()}.html"),"w") as f:
        f.write(html)

# -------------------------
# Ensemble & backtest loop
# -------------------------
def ensemble_decision(strats, row: dict, require: int = 2) -> Tuple[str,str]:
    votes=[]; reasons=[]
    for s in strats:
        sig = s.generate(row)
        if sig.side in ('buy','sell'):
            votes.append(sig.side); reasons.append(getattr(s,'name','?'))
    side='flat'
    if votes.count('buy') >= require: side='buy'
    elif votes.count('sell') >= require: side='sell'
    return side, ",".join(reasons)

def backtest_one(df_raw: pd.DataFrame, config: dict, strategy_overrides: dict | None = None, cost_curves: dict | None = None, symbol: str = "BTCUSDT") -> pd.DataFrame:
    feats = FeatureStore().compute(df_raw)
    strats = load_strategies(strategy_overrides)
    equity=10000.0; per_trade=config['risk']['per_trade']
    res=[]; open_trade=None

    # optional meta model (scaffold—train offline; here we use a simple default p̂)
    meta_threshold = 0.6
    meta_active = False  # set True if you wire a trained model

    for ts, row in feats.iterrows():
        # manage open
        if open_trade is not None:
            high=row['high']; low=row['low']; close=row['close']
            tp=open_trade['tp']; st=open_trade['stop']; entry=open_trade['entry']; side=open_trade['side']
            prob_tp, prob_sl = simulate_intrabar(entry, high, low, close, st, tp, side, row.get('regime','Ranging'))
            hit = 'tp' if prob_tp>=prob_sl else 'sl'
            pnl_R = (tp-entry)/open_trade['stop_dist'] if (hit=='tp' and side=='buy') else \
                    (entry-st)/open_trade['stop_dist'] if (hit=='sl' and side=='buy') else \
                    (entry-tp)/open_trade['stop_dist'] if (hit=='tp' and side=='sell') else \
                    (st-entry)/open_trade['stop_dist']
            fees = open_trade['fee_R']; slips = open_trade['slip_R']
            net_R = pnl_R - (fees+slips)
            equity += net_R * open_trade['dollar_risk']
            res.append(dict(timestamp=ts, decision=hit, pnl_R=pnl_R, fee_R=fees, slip_R=slips,
                            net_R=net_R, equity=equity, symbol=symbol, votes=open_trade['votes'],
                            order_type=open_trade['order_type']))
            open_trade=None

        # ensemble vote
        side, votes = ensemble_decision(strats, row.to_dict(), require=config.get('ensemble',{}).get('require_agreement',2))
        if side=='flat': continue

        # meta gating (disabled by default without trained model)
        if meta_active:
            pass  # add model inference here if you've trained MetaModel

        # position sizing: ATR baseline + Kelly modulation
        k_atr = 2.0 if row.get('regime')=='Trending' else 1.5
        size, stop_dist, dollar_risk = atr_sizing(equity, float(row['atr14']), k_atr, per_trade)
        p_hat = shrink_p(0.55, 500, alpha=12.0)   # conservative default; replace with calibrated p̂
        k_frac = fractional_kelly(p_hat, 1.5, 1.0, cap=0.02)
        dollar_risk *= max(0.5, min(1.5, (k_frac/0.02)))

        entry=float(row['close'])
        if side=='buy':
            stop = entry - stop_dist; tp = entry + stop_dist*(2.0 if row.get('regime')=='Trending' else 1.2)
        else:
            stop = entry + stop_dist; tp = entry - stop_dist*(2.0 if row.get('regime')=='Trending' else 1.2)

        hour = ts.hour
        venue, order_type = choose_route(cost_curves or {'LOCAL':{}}, hour, urgency=0.8 if 'breakout' in votes else 0.4)
        fill = place_order(side, entry, size, spread_bps=row.get('spread_bps',2.0), urgency=0.8 if 'breakout' in votes else 0.4, order_type=order_type)
        if not fill.filled: continue

        fees = fee_R(fill.entry, stop_dist, bps=10.0 if fill.order_type=='taker' else 2.0)
        slips = slip_R(fill.slip_bps, fill.entry, stop_dist)
        open_trade=dict(entry=fill.entry, side=side, tp=tp, stop=stop, size=size, stop_dist=stop_dist, dollar_risk=dollar_risk,
                        votes=votes, slip_R=slips, fee_R=fees, order_type=fill.order_type)

    return pd.DataFrame(res)

# -------------------------
# Portfolio combiner (bar-synchronous)
# -------------------------
def combine_equity(csv_map: Dict[str,str], method: str = "risk_parity", window: int = 100, out_dir: str = "reports") -> str:
    dfs = {k: pd.read_csv(v, parse_dates=['timestamp']) for k,v in csv_map.items()}
    idx = sorted(set().union(*[set(d['timestamp']) for d in dfs.values()]))
    R = []
    for sym, df in dfs.items():
        s = pd.Series(df.set_index('timestamp')['net_R'], index=idx).fillna(0.0).rename(sym)
        R.append(s)
    R = pd.concat(R, axis=1).fillna(0.0)
    W_rows=[]
    for i in range(len(R)):
        lo=max(0,i-window)
        cov = shrink_cov(R.iloc[lo:i+1])
        w = risk_parity_weights(cov) if method=='risk_parity' else min_variance_weights(cov)
        W_rows.append(w)
    W = pd.DataFrame(W_rows, index=R.index).fillna(method='ffill').fillna(1.0/len(R.columns))
    port_step = (R * W).sum(axis=1)
    eq = port_step.cumsum()
    out = pd.DataFrame({'timestamp': R.index, 'net_R': port_step.values, 'equity': eq.values})
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"portfolio_{method}.csv")
    out.to_csv(out_csv, index=False)
    return out_csv

# -------------------------
# Config & Runner
# -------------------------
DEFAULT_CONFIG = {
    "risk": {"per_trade": 0.008, "daily_loss_cap_R": 2.0, "week_loss_cap_R": 5.0, "kill_switch_drawdown_pct": 0.25},
    "fees": {"maker_bps": 2.0, "taker_bps": 10.0},
    "reports": {"dir": "reports"},
    "ensemble": {"require_agreement": 2},
}

def run_pipeline(markets: List[str], data_dir: str = "data", reports_dir: str = "reports", config: dict = None):
    cfg = config or DEFAULT_CONFIG
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Make synthetic if not present
    seeds = {"BTCUSDT": (0.03,20000),"ETHUSDT": (0.05,1400),"SOLUSDT": (0.08,25)}
    for m,(mu,p0) in seeds.items():
        path = os.path.join(data_dir, f"{m}_1h.csv")
        if not os.path.exists(path):
            make_synth(2600, mu=mu, price0=p0).to_csv(path, index=False)

    cost_curves = {'LOCAL': {'maker_bps':{}, 'taker_bps':{}, 'slip_bps':{}}}

    out_map={}
    for m in markets:
        csv = os.path.join(data_dir, f"{m}_1h.csv")
        df = load_csv(csv)
        assert check_sla(df), f"Data SLA failed for {m}"
        res = backtest_one(df, cfg, symbol=m, cost_curves=cost_curves)
        out = os.path.join(reports_dir, f"backtest_{m}.csv")
        res.to_csv(out, index=False)
        k = kpis(res.assign(equity=res['net_R'].cumsum()))
        write_html_report(reports_dir, f"{m}_report", k)
        out_map[m]=out

    # Portfolio report
    p_csv = combine_equity(out_map, method='risk_parity', window=100, out_dir=reports_dir)
    pdf = pd.read_csv(p_csv, parse_dates=['timestamp'])
    write_html_report(reports_dir, "portfolio_report", kpis(pdf))

    print(f"Done. See {reports_dir}/ for CSVs and HTML reports.")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # You can edit the markets list or pass via CLI by extending this main.
    run_pipeline(markets=["BTCUSDT","ETHUSDT","SOLUSDT"])

