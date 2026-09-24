import time
import pandas as pd
import requests
import streamlit as st

BASE = "https://fapi.binance.com"
TIMEOUT = 12
MIN_24H_QUOTE_VOLUME = 5_000_000
MAX_SYMBOLS = 120
KLINES_LIMIT = 120


def get(path, params=None):
    r = requests.get(BASE + path, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def pct(a, b):
    if b == 0 or b is None:
        return 0.0
    return (a / b - 1.0) * 100.0


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def fetch_universe():
    tickers = get("/fapi/v1/ticker/24hr")
    rows = []
    for x in tickers:
        s = x["symbol"]
        if not s.endswith("USDT") or s.endswith(("USDC", "BUSD")):
            continue
        qv = float(x.get("quoteVolume", 0))
        if qv < MIN_24H_QUOTE_VOLUME:
            continue
        rows.append({
            "symbol": s,
            "last": float(x["lastPrice"]),
            "change24": float(x["priceChangePercent"]),
            "quote_volume_24h": qv,
        })
    rows.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
    return rows[:MAX_SYMBOLS]


def fetch_klines(symbol):
    raw = get("/fapi/v1/klines", {
        "symbol": symbol,
        "interval": "1h",
        "limit": KLINES_LIMIT,
    })
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_funding(symbol):
    return float(get("/fapi/v1/premiumIndex", {"symbol": symbol})["lastFundingRate"])


def fetch_oi_history(symbol):
    data = get("/futures/data/openInterestHist", {
        "symbol": symbol,
        "period": "1h",
        "limit": 25,
    })
    if not data:
        return None, None
    oi = pd.Series([float(x["sumOpenInterestValue"]) for x in data])
    if len(oi) < 2 or oi.iloc[0] == 0:
        return None, None
    return float(oi.iloc[-1]), float((oi.iloc[-1] / oi.iloc[0] - 1) * 100)


def fetch_long_short(symbol):
    data = get("/futures/data/globalLongShortAccountRatio", {
        "symbol": symbol,
        "period": "1h",
        "limit": 1,
    })
    return float(data[-1]["longShortRatio"]) if data else None


def score_coin(row, df, oi_change, funding, ls_ratio):
    close, volume = df["close"], df["quote_volume"]
    r1 = pct(close.iloc[-1], close.iloc[-2])
    r4 = pct(close.iloc[-1], close.iloc[-5])
    r24 = pct(close.iloc[-1], close.iloc[-25])
    e20, e50 = ema(close, 20).iloc[-1], ema(close, 50).iloc[-1]

    vol_base = volume.iloc[-21:-1].mean()
    rel_vol = float(volume.iloc[-1] / vol_base) if vol_base else 0

    momentum = min(
        sum([
            4 if r1 > 0 else 0,
            6 if r4 > 0 else 0,
            6 if 0 < r24 <= 20 else (3 if r24 > 20 else 0),
            2 if r4 > 3 else 0,
            2 if r24 > 5 else 0,
        ]),
        20,
    )
    volume_score = min(
        sum([
            4 if rel_vol >= 1.5 else 0,
            4 if rel_vol >= 2.0 else 0,
            4 if rel_vol >= 3.0 else 0,
            3 if rel_vol >= 5.0 else 0,
            2 if r1 > 0 and rel_vol >= 2 else 0,
        ]),
        15,
    )

    oi_score = 0
    if oi_change is not None:
        oi_score = min(
            sum([
                5 if r24 > 0 and oi_change > 0 else 0,
                3 if oi_change >= 5 else 0,
                4 if oi_change >= 10 else 0,
                3 if oi_change >= 20 else 0,
            ]),
            15,
        )

    funding_score = 0
    if funding is not None:
        if -0.0005 <= funding <= 0.0005:
            funding_score += 4
        if funding < 0 and r24 > 0:
            funding_score += 5
        elif 0 < funding <= 0.001:
            funding_score += 3
        if funding > 0.003:
            funding_score -= 3
        if funding < -0.003:
            funding_score -= 2
    funding_score = int(clamp(funding_score, 0, 10))

    squeeze = min(
        sum([
            5 if ls_ratio and r24 > 0 and ls_ratio < 0.9 else 0,
            3 if ls_ratio and r24 > 0 and ls_ratio < 0.75 else 0,
            2 if r1 > 0 and rel_vol >= 2 else 0,
        ]),
        10,
    )
    trend = (5 if close.iloc[-1] > e20 else 0) + (5 if close.iloc[-1] > e50 else 0)
    liquidity = 10 if row["quote_volume_24h"] >= 50_000_000 else 7
    chase = int(
        clamp(
            10
            - (3 if r24 > 15 else 0)
            - (3 if r24 > 30 else 0)
            - (4 if r24 > 50 else 0)
            - (3 if r1 > 8 else 0),
            0,
            10,
        )
    )

    total = int(
        clamp(
            round(
                momentum
                + volume_score
                + oi_score
                + funding_score
                + squeeze
                + trend
                + liquidity
                + chase
            ),
            0,
            100,
        )
    )

    reasons = []
    if rel_vol >= 2:
        reasons.append(f"vol {rel_vol:.1f}x")
    if oi_change and oi_change >= 5:
        reasons.append(f"OI +{oi_change:.1f}%")
    if r24 > 0:
        reasons.append(f"24h +{r24:.1f}%")
    if close.iloc[-1] > e20 and close.iloc[-1] > e50:
        reasons.append("above 20/50 EMA")
    if funding and funding < 0 and r24 > 0:
        reasons.append("neg funding")

    return {
        "score": total,
        "r1h_pct": r1,
        "r4h_pct": r4,
        "r24h_pct": r24,
        "rel_volume": rel_vol,
        "oi_change_24h_pct": oi_change,
        "funding": funding,
        "long_short": ls_ratio,
        "reasons": "; ".join(reasons[:5]),
    }


st.set_page_config(page_title="Crypto Pump Detector", layout="wide")
st.title("⚡ Crypto Pump Detector V1")

st.sidebar.header("Scan Parameters")
min_score = st.sidebar.slider("Minimum Score", 0, 100, 30)
display_limit = st.sidebar.number_input("Max Display Count", 5, 50, 15)

if st.button("Run Live Scanner"):
    universe = fetch_universe()
    progress_bar = st.progress(0)
    results = []

    for i, row in enumerate(universe):
        try:
            df = fetch_klines(row["symbol"])
            if len(df) >= 60:
                _, oi_change = fetch_oi_history(row["symbol"])
                funding = fetch_funding(row["symbol"])
                ls = fetch_long_short(row["symbol"])
                scored = score_coin(row, df, oi_change, funding, ls)
                results.append({**row, **scored})
        except Exception:
            pass
        progress_bar.progress((i + 1) / len(universe))
        time.sleep(0.02)

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)
    filtered = df_res[df_res["score"] >= min_score].head(display_limit)

    st.subheader(f"Top Signals (Score >= {min_score})")
    st.dataframe(
        filtered[[
            "symbol",
            "score",
            "r1h_pct",
            "r4h_pct",
            "r24h_pct",
            "rel_volume",
            "reasons",
        ]],
        use_container_width=True,
    )