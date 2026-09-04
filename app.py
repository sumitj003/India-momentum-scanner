import hashlib
import os
import time
from datetime import date, timedelta
from io import StringIO
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="India Momentum Scanner", page_icon="📈", layout="wide")

BASE = "https://api.kite.trade"
IST = ZoneInfo("Asia/Kolkata")


def get_secret(name, default=""):
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)


API_KEY = get_secret("KITE_API_KEY", "")
API_SECRET = get_secret("KITE_API_SECRET", "")
REDIRECT_URL = get_secret("KITE_REDIRECT_URL", "")


def kite_headers(token):
    return {
        "X-Kite-Version": "3",
        "Authorization": f"token {API_KEY}:{token}",
    }


def login_url():
    return "https://kite.zerodha.com/connect/login?" + urlencode({
        "v": "3",
        "api_key": API_KEY,
    })


def exchange_request_token(request_token):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Kite API key/secret are not configured.")
    checksum = hashlib.sha256(
        (API_KEY + request_token + API_SECRET).encode()
    ).hexdigest()
    r = requests.post(
        BASE + "/session/token",
        headers={"X-Kite-Version": "3"},
        data={
            "api_key": API_KEY,
            "request_token": request_token,
            "checksum": checksum,
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["data"]["access_token"]


def api_get(path, token, params=None, timeout=30):
    r = requests.get(
        BASE + path,
        headers=kite_headers(token),
        params=params,
        timeout=timeout,
    )
    if r.status_code == 429:
        time.sleep(1.2)
        r = requests.get(
            BASE + path,
            headers=kite_headers(token),
            params=params,
            timeout=timeout,
        )
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(j.get("message", "Kite API error"))
    return j["data"]


@st.cache_data(ttl=86400, show_spinner=False)
def download_instruments():
    r = requests.get(BASE + "/instruments", timeout=60)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def historical(token, instrument_token, start, end):
    data = api_get(
        f"/instruments/historical/{int(instrument_token)}/day",
        token,
        params={"from": start.isoformat(), "to": end.isoformat(), "oi": 0},
        timeout=30,
    )
    if not data.get("candles"):
        return pd.DataFrame()
    return pd.DataFrame(
        data["candles"],
        columns=["date", "open", "high", "low", "close", "volume"],
    )


def quote_batch(token, keys):
    if not keys:
        return {}
    # /quote accepts up to 500 instruments per request.
    return api_get("/quote", token, params={"i": keys}, timeout=30)


def previous_weekday(d):
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def metrics(df):
    if df.empty:
        return None
    df = df.copy().sort_values("date")
    if len(df) < 60:
        return None

    c = df.close.astype(float)
    h = df.high.astype(float)
    l = df.low.astype(float)
    v = df.volume.astype(float)

    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = 100 * df["atr14"] / c
    df["ret30"] = 100 * (c / c.shift(30) - 1)
    df["ret50"] = 100 * (c / c.shift(50) - 1)
    df["ret10"] = 100 * (c / c.shift(10) - 1)
    df["traded_value_cr"] = c * v / 1e7

    x = df.iloc[-1]
    return x, df


def score(row):
    m30 = np.clip((row.ret30 - 5) / 15, 0, 1) * 20
    m50 = np.clip((row.ret50 - 5) / 25, 0, 1) * 15
    rs = np.clip(row.rs / 15, 0, 1) * 20
    trend = 20.0 if row.close > row.sma20 > row.sma50 > row.sma50_5d else 0.0
    liq = np.clip((row.avg20_value_cr - 10) / 40, 0, 1) * 10

    atr = row.atr_pct
    if 2 <= atr <= 4:
        vol = 10
    elif 1.5 <= atr < 2 or 4 < atr <= 5:
        vol = 7
    elif 1.0 <= atr < 1.5 or 5 < atr <= 6:
        vol = 4
    else:
        vol = 0

    accel = np.clip(row.ret10 / 10, 0, 1) * 5
    return round(m30 + m50 + rs + trend + liq + vol + accel, 1)


st.title("India Momentum Scanner")
st.caption("Weekly NSE + BSE momentum scan • closing-data based • recommendations only")

if not API_KEY:
    st.error("Kite API key is not configured. Add KITE_API_KEY in Streamlit Secrets.")
    st.stop()

# -----------------------------
# Zerodha login
# -----------------------------
with st.sidebar:
    st.header("Zerodha")
    if REDIRECT_URL:
        st.caption("Cloud login is configured.")
    else:
        st.warning("KITE_REDIRECT_URL is missing in Secrets.")

    if st.session_state.get("access_token"):
        st.success("Zerodha connected")
        if st.button("Disconnect"):
            st.session_state.pop("access_token", None)
            st.rerun()
    else:
        st.link_button("LOGIN WITH ZERODHA", login_url(), use_container_width=True)
        st.caption("Zerodha login is manual. The session normally needs a fresh login each day.")

params = st.query_params
request_token = params.get("request_token")
if request_token and not st.session_state.get("access_token"):
    try:
        st.session_state.access_token = exchange_request_token(request_token)
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

if not st.session_state.get("access_token"):
    st.info("Tap LOGIN WITH ZERODHA first.")
    st.stop()

token = st.session_state.access_token

try:
    profile = api_get("/user/profile", token)
    st.sidebar.caption(f"Logged in: {profile.get('user_name', 'Zerodha user')}")
except Exception:
    st.session_state.pop("access_token", None)
    st.error("Zerodha session expired. Log in again.")
    st.stop()

# -----------------------------
# Scan
# -----------------------------
today_ist = date.today()
default_asof = previous_weekday(today_ist)
asof = st.date_input("Closing date", value=default_asof)
run = st.button("RUN SCAN", type="primary", use_container_width=True)

if run:
    with st.spinner("Loading NSE/BSE instruments..."):
        inst = download_instruments()

    # Keep NSE/BSE cash equities. Deduplicate by ISIN where possible; prefer NSE.
    eq = inst[
        inst.exchange.isin(["NSE", "BSE"])
        & (inst.instrument_type == "EQ")
        & inst.tradingsymbol.notna()
    ].copy()
    eq["priority"] = np.where(eq.exchange.eq("NSE"), 0, 1)
    eq["dedupe"] = eq["isin"].fillna("") if "isin" in eq.columns else ""
    eq["dedupe"] = np.where(
        eq["dedupe"].eq(""),
        eq["exchange"].astype(str) + ":" + eq["tradingsymbol"].astype(str),
        eq["dedupe"],
    )
    eq = eq.sort_values(["dedupe", "priority"]).drop_duplicates("dedupe", keep="first")

    start = asof - timedelta(days=180)
    end = asof

    # Nifty 500 benchmark.
    bench = inst[
        (inst.exchange == "NSE")
        & (inst.tradingsymbol.astype(str).str.upper() == "NIFTY 500")
    ]
    bench_ret30 = 0.0
    bench_ret50 = 0.0
    if not bench.empty:
        try:
            bdf = historical(token, int(bench.iloc[0].instrument_token), start, end)
            bm = metrics(bdf)
            if bm:
                bx, _ = bm
                bench_ret30 = float(bx.ret30)
                bench_ret50 = float(bx.ret50)
        except Exception as e:
            st.warning(f"Nifty 500 benchmark unavailable: {e}")

    # Fast pre-filter using current quote: price >= ₹50 and latest traded value >= ₹10 Cr.
    # This avoids thousands of historical calls for obvious rejects.
    eq["quote_key"] = eq.exchange.astype(str) + ":" + eq.tradingsymbol.astype(str)
    keys = eq.quote_key.tolist()
    quotes = {}
    progress = st.progress(0)
    total_batches = max(1, (len(keys) + 499) // 500)
    for bi in range(0, len(keys), 500):
        batch = keys[bi:bi + 500]
        try:
            quotes.update(quote_batch(token, batch))
        except Exception:
            pass
        progress.progress(min((bi // 500 + 1) / total_batches, 1.0))
        time.sleep(1.05)

    candidates = []
    for r in eq.itertuples(index=False):
        q = quotes.get(r.quote_key, {})
        try:
            price = float(q.get("last_price", 0))
            volume = float(q.get("volume", 0))
            latest_tv = price * volume / 1e7
        except Exception:
            continue
        if price < 50 or latest_tv < 10:
            continue
        candidates.append(r)

    st.info(f"Universe: {len(eq):,} listed equities → {len(candidates):,} passed the quick price/liquidity check.")

    rows = []
    progress = st.progress(0)
    status = st.empty()
    total = len(candidates)

    for i, r in enumerate(candidates, start=1):
        try:
            df = historical(token, int(r.instrument_token), start, end)
            m = metrics(df)
            if not m:
                continue
            x, hist = m
            if pd.Timestamp(x.date).date() != asof:
                continue

            price = float(x.close)
            ret30 = float(x.ret30)
            ret50 = float(x.ret50)
            sma20 = float(x.sma20)
            sma50 = float(x.sma50)
            sma50_5d = float(hist["sma50"].iloc[-6])
            atr = float(x.atr_pct)
            avg20 = float(hist["traded_value_cr"].tail(20).mean())

            # Exact system hard filters.
            if price < 50:
                continue
            if ret30 < 5 and ret50 < 5:
                continue
            if not (price > sma20 > sma50):
                continue
            if not (sma50 > sma50_5d):
                continue
            if not (1 <= atr <= 6):
                continue
            if avg20 < 10:
                continue

            rs = max(ret30 - bench_ret30, ret50 - bench_ret50)
            if rs <= 0:
                continue

            rows.append({
                "Stock": r.tradingsymbol,
                "Exchange": r.exchange,
                "Price": price,
                "30D %": ret30,
                "50D %": ret50,
                "RS %": rs,
                "ATR %": atr,
                "Avg20 ₹Cr": avg20,
                "Latest ₹Cr": float(x.traded_value_cr),
                "sma20": sma20,
                "sma50": sma50,
                "sma50_5d": sma50_5d,
                "ret10": float(x.ret10),
            })
        except Exception:
            pass

        progress.progress(i / max(total, 1))
        if i % 25 == 0:
            status.write(f"Checked {i:,}/{total:,}")
        time.sleep(0.34)

    out = pd.DataFrame(rows)
    if out.empty:
        st.error("0 — no stock qualified under the complete system.")
    else:
        out["Score"] = out.apply(score, axis=1)
        out = out.sort_values(["Score", "RS %", "30D %"], ascending=False).reset_index(drop=True).head(10)
        out.insert(0, "Rank", range(1, len(out) + 1))
        out["Allocation %"] = round(100 / len(out), 1)

        st.success(f"{len(out)} stocks qualified")
        st.caption(f"Nifty 500: 30D {bench_ret30:.1f}% | 50D {bench_ret50:.1f}%")
        st.dataframe(
            out[[
                "Rank", "Stock", "Exchange", "Price", "30D %", "50D %",
                "RS %", "ATR %", "Avg20 ₹Cr", "Latest ₹Cr", "Score", "Allocation %"
            ]].round(2),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "DOWNLOAD CSV",
            out.to_csv(index=False).encode(),
            f"momentum_{asof}.csv",
            "text/csv",
            use_container_width=True,
        )
