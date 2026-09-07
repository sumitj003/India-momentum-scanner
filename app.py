import hashlib
import os
import time
from datetime import date, timedelta
from io import BytesIO, StringIO
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="India Momentum Scanner", page_icon="📈", layout="wide")

BASE = "https://api.kite.trade"
IST = ZoneInfo("Asia/Kolkata")
NSE_BHAV = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
BSE_BHAV = "https://www.bseindia.com/download/BhavCopy/Equity/EQ{ddmmyy}_CSV.ZIP"


def get_secret(name, default=""):
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)


API_KEY = get_secret("KITE_API_KEY", "")
API_SECRET = get_secret("KITE_API_SECRET", "")
REDIRECT_URL = get_secret("KITE_REDIRECT_URL", "")


def kite_headers(token):
    return {"X-Kite-Version": "3", "Authorization": f"token {API_KEY}:{token}"}


def login_url():
    return "https://kite.zerodha.com/connect/login?" + urlencode({"v": "3", "api_key": API_KEY})


def exchange_request_token(request_token):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Kite API key/secret are not configured.")
    checksum = hashlib.sha256((API_KEY + request_token + API_SECRET).encode()).hexdigest()
    r = requests.post(
        BASE + "/session/token",
        headers={"X-Kite-Version": "3"},
        data={"api_key": API_KEY, "request_token": request_token, "checksum": checksum},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["data"]["access_token"]


def api_get(path, token, params=None, timeout=30):
    r = requests.get(BASE + path, headers=kite_headers(token), params=params, timeout=timeout)
    if r.status_code == 429:
        time.sleep(1.2)
        r = requests.get(BASE + path, headers=kite_headers(token), params=params, timeout=timeout)
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
    return pd.DataFrame(data["candles"], columns=["date", "open", "high", "low", "close", "volume"])


def previous_weekday(d):
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def request_file(url, timeout=30):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
        "Accept": "*/*",
        "Referer": "https://www.nseindia.com/",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    if r.status_code != 200 or len(r.content) < 1000:
        raise RuntimeError(f"Download failed ({r.status_code})")
    return r.content


def read_zip_csv(blob):
    with ZipFile(BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not names:
            raise RuntimeError("No CSV/TXT file inside bhavcopy zip")
        with z.open(names[0]) as f:
            return pd.read_csv(f, low_memory=False)


def clean_num(s):
    return pd.to_numeric(s, errors="coerce")


def nse_day(d):
    blob = request_file(NSE_BHAV.format(yyyymmdd=d.strftime("%Y%m%d")))
    df = read_zip_csv(blob)
    # UDiFF common bhavcopy field names. Keep flexible fallbacks for minor format changes.
    cols = {str(c).lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None
    sym = col("TckrSymb", "SYMBOL")
    isin = col("ISIN", "Isin")
    series = col("SctySrs", "SERIES")
    close = col("ClsPric", "CLOSE")
    high = col("HghPric", "HIGH")
    low = col("LwPric", "LOW")
    vol = col("TtlTradgVol", "VOLUME")
    value = col("TtlTrfVal", "TOTTRDVAL")
    if not all([sym, close, high, low, vol]):
        raise RuntimeError(f"Unexpected NSE bhavcopy columns: {list(df.columns)[:12]}")
    if series:
        df = df[df[series].astype(str).str.upper().eq("EQ")].copy()
    out = pd.DataFrame({
        "exchange": "NSE",
        "symbol": df[sym].astype(str),
        "isin": df[isin].astype(str) if isin else "",
        "date": d,
        "close": clean_num(df[close]),
        "high": clean_num(df[high]),
        "low": clean_num(df[low]),
        "volume": clean_num(df[vol]),
        "value": clean_num(df[value]) if value else np.nan,
    })
    out = out[(out.close > 0) & (out.volume >= 0)].dropna(subset=["close", "high", "low"])
    return out


def bse_day(d):
    blob = request_file(BSE_BHAV.format(ddmmyy=d.strftime("%d%m%y")))
    df = read_zip_csv(blob)
    cols = {str(c).strip().upper(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n.upper() in cols:
                return cols[n.upper()]
        return None
    sym = col("SYMBOL", "SC_CODE")
    isin = col("ISIN")
    series = col("SC_GROUP", "SERIES")
    close = col("CLOSE", "ClsPric")
    high = col("HIGH", "HghPric")
    low = col("LOW", "LwPric")
    vol = col("VOLUME", "TtlTradgVol")
    value = col("NET_TURNOV", "TtlTrfVal")
    if not all([sym, close, high, low, vol]):
        raise RuntimeError(f"Unexpected BSE bhavcopy columns: {list(df.columns)[:12]}")
    if series:
        # BSE cash equity group; exclude SME groups and non-equity rows.
        s = df[series].astype(str).str.upper()
        df = df[s.isin(["A", "B", "T", "XT", "X", "Z", "ZP", "M", "P", "Q", "EQ"])].copy()
    out = pd.DataFrame({
        "exchange": "BSE",
        "symbol": df[sym].astype(str),
        "isin": df[isin].astype(str) if isin else "",
        "date": d,
        "close": clean_num(df[close]),
        "high": clean_num(df[high]),
        "low": clean_num(df[low]),
        "volume": clean_num(df[vol]),
        "value": clean_num(df[value]) if value else np.nan,
    })
    out = out[(out.close > 0) & (out.volume >= 0)].dropna(subset=["close", "high", "low"])
    return out


def trading_dates(end_date, wanted=65, max_calendar=110):
    dates = []
    d = end_date
    for _ in range(max_calendar):
        if d.weekday() < 5:
            dates.append(d)
            if len(dates) >= wanted:
                break
        d -= timedelta(days=1)
    return dates


@st.cache_data(ttl=21600, show_spinner=False)
def load_market_history(asof_iso, min_days=60):
    asof = date.fromisoformat(asof_iso)
    days = trading_dates(asof, wanted=min_days + 5)
    nse_parts, bse_parts, failures = [], [], []
    for d in days:
        try:
            nse_parts.append(nse_day(d))
        except Exception as e:
            failures.append(f"NSE {d}: {e}")
        try:
            bse_parts.append(bse_day(d))
        except Exception as e:
            failures.append(f"BSE {d}: {e}")
        if len(nse_parts) + len(bse_parts) >= 2 * min_days and len(failures) > 20:
            break

    if not nse_parts and not bse_parts:
        raise RuntimeError("Could not download NSE/BSE daily bhavcopy data.")

    all_rows = pd.concat(nse_parts + bse_parts, ignore_index=True)
    all_rows["isin"] = all_rows["isin"].replace({"nan": "", "None": ""}).fillna("")
    all_rows["symbol"] = all_rows["symbol"].astype(str)

    # If the same ISIN traded on both exchanges, prefer NSE. BSE-only stocks remain.
    all_rows["priority"] = np.where(all_rows.exchange.eq("NSE"), 0, 1)
    all_rows["dedupe"] = np.where(
        all_rows["isin"].ne(""),
        all_rows["isin"],
        all_rows["exchange"] + ":" + all_rows["symbol"],
    )
    all_rows = all_rows.sort_values(["date", "dedupe", "priority"]).drop_duplicates(["date", "dedupe"], keep="first")
    return all_rows, failures


def compute_metrics(g):
    g = g.sort_values("date").copy()
    if len(g) < 55:
        return None
    c = g.close.astype(float)
    h = g.high.astype(float)
    l = g.low.astype(float)
    v = g.volume.astype(float)
    g["sma20"] = c.rolling(20).mean()
    g["sma50"] = c.rolling(50).mean()
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14).mean()
    g["atr_pct"] = 100 * g["atr14"] / c
    g["ret30"] = 100 * (c / c.shift(30) - 1)
    g["ret50"] = 100 * (c / c.shift(50) - 1)
    g["ret10"] = 100 * (c / c.shift(10) - 1)
    g["traded_value_cr"] = c * v / 1e7
    x = g.iloc[-1]
    if pd.isna(x.sma20) or pd.isna(x.sma50) or pd.isna(x.ret50):
        return None
    return x, g


def score_row(r):
    m30 = np.clip((r["30D %"] - 5) / 15, 0, 1) * 20
    m50 = np.clip((r["50D %"] - 5) / 25, 0, 1) * 15
    rs = np.clip(r["RS %"] / 15, 0, 1) * 20
    trend = 20.0 if r["Price"] > r["sma20"] > r["sma50"] > r["sma50_5d"] else 0.0
    liq = np.clip((r["Avg20 ₹Cr"] - 10) / 40, 0, 1) * 10
    atr = r["ATR %"]
    if 2 <= atr <= 4:
        vol = 10
    elif 1.5 <= atr < 2 or 4 < atr <= 5:
        vol = 7
    elif 1 <= atr < 1.5 or 5 < atr <= 6:
        vol = 4
    else:
        vol = 0
    accel = np.clip(r["10D %"] / 10, 0, 1) * 5
    return round(m30 + m50 + rs + trend + liq + vol + accel, 1)


st.title("India Momentum Scanner")
st.caption("Weekly NSE + BSE momentum scan • closing-data based • recommendations only")

if not API_KEY:
    st.error("Kite API key is not configured. Add KITE_API_KEY in Streamlit Secrets.")
    st.stop()

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


today_ist = date.today()
default_asof = previous_weekday(today_ist)
asof = st.date_input("Closing date", value=default_asof)
run = st.button("RUN SCAN", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Loading NSE/BSE daily market history..."):
            market, failures = load_market_history(asof.isoformat(), min_days=60)
    except Exception as e:
        st.error(f"Market-data load failed: {e}")
        st.stop()

    # Exact selected closing date only.
    market = market[market.date <= asof].copy()
    available_dates = sorted(market.date.unique())
    if not available_dates or available_dates[-1] != asof:
        st.error(f"No complete NSE/BSE closing data found for {asof}. Choose the latest completed trading date.")
        st.stop()

    # Benchmark from Kite; one API call only.
    bench_ret30 = bench_ret50 = 0.0
    try:
        inst = download_instruments()
        bench = inst[(inst.exchange == "NSE") & (inst.tradingsymbol.astype(str).str.upper() == "NIFTY 500")]
        if not bench.empty:
            bdf = historical(token, int(bench.iloc[0].instrument_token), asof - timedelta(days=180), asof)
            if len(bdf) >= 55:
                bdf = bdf.sort_values("date")
                bc = bdf.close.astype(float)
                bench_ret30 = float(100 * (bc.iloc[-1] / bc.iloc[-31] - 1))
                bench_ret50 = float(100 * (bc.iloc[-1] / bc.iloc[-51] - 1))
    except Exception as e:
        st.warning(f"Nifty 500 benchmark unavailable; RS will use 0 benchmark: {e}")

    # Build latest-date universe and apply price/liquidity only after complete history exists.
    latest = market[market.date.eq(asof)].copy()
    hist_groups = market.groupby("dedupe", sort=False)
    rows = []
    counts = {"history": 0, "price": 0, "momentum": 0, "trend": 0, "atr": 0, "liquidity": 0, "rs": 0}
    total = len(latest)
    progress = st.progress(0)
    status = st.empty()

    for i, (_, r) in enumerate(latest.iterrows(), start=1):
        g = hist_groups.get_group(r.dedupe)
        m = compute_metrics(g)
        if not m:
            progress.progress(i / max(total, 1))
            continue
        counts["history"] += 1
        x, hist = m
        price = float(x.close)
        if price < 50:
            progress.progress(i / max(total, 1))
            continue
        counts["price"] += 1

        ret30, ret50 = float(x.ret30), float(x.ret50)
        if ret30 < 5 and ret50 < 5:
            progress.progress(i / max(total, 1))
            continue
        counts["momentum"] += 1

        sma20, sma50 = float(x.sma20), float(x.sma50)
        sma50_5d = float(hist["sma50"].iloc[-6])
        if not (price > sma20 > sma50 and sma50 > sma50_5d):
            progress.progress(i / max(total, 1))
            continue
        counts["trend"] += 1

        atr = float(x.atr_pct)
        if not (1 <= atr <= 6):
            progress.progress(i / max(total, 1))
            continue
        counts["atr"] += 1

        avg20 = float(hist["traded_value_cr"].tail(20).mean())
        latest_tv = float(x.traded_value_cr)
        if avg20 < 10 or latest_tv < 10:
            progress.progress(i / max(total, 1))
            continue
        counts["liquidity"] += 1

        rs = max(ret30 - bench_ret30, ret50 - bench_ret50)
        if rs <= 0:
            progress.progress(i / max(total, 1))
            continue
        counts["rs"] += 1

        rows.append({
            "Stock": str(r.symbol),
            "Exchange": str(r.exchange),
            "Price": price,
            "30D %": ret30,
            "50D %": ret50,
            "10D %": float(x.ret10),
            "RS %": rs,
            "ATR %": atr,
            "Avg20 ₹Cr": avg20,
            "Latest ₹Cr": latest_tv,
            "sma20": sma20,
            "sma50": sma50,
            "sma50_5d": sma50_5d,
        })
        progress.progress(i / max(total, 1))
        if i % 100 == 0:
            status.write(f"Checked {i:,}/{total:,}")

    out = pd.DataFrame(rows)
    if out.empty:
        st.error("0 — no stock qualified under the complete system.")
    else:
        out["Score"] = out.apply(score_row, axis=1)
        out = out.sort_values(["Score", "RS %", "30D %"], ascending=False).reset_index(drop=True).head(10)
        out.insert(0, "Rank", range(1, len(out) + 1))
        out["Allocation %"] = round(100 / len(out), 1)

        st.success(f"{len(out)} stocks qualified")
        st.caption(f"Nifty 500: 30D {bench_ret30:.1f}% | 50D {bench_ret50:.1f}%")
        st.dataframe(
            out[[
                "Rank", "Stock", "Exchange", "Price", "30D %", "50D %", "RS %", "ATR %",
                "Avg20 ₹Cr", "Latest ₹Cr", "Score", "Allocation %"
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

    with st.expander("Scan details"):
        st.write({
            "Latest-date securities": total,
            "Enough history": counts["history"],
            "Price ≥ ₹50": counts["price"],
            "Momentum": counts["momentum"],
            "Trend": counts["trend"],
            "ATR 1–6%": counts["atr"],
            "Liquidity": counts["liquidity"],
            "Relative strength": counts["rs"],
        })
        if failures:
            st.caption(f"Data files unavailable on {len(failures)} exchange/date attempts; qualifying results use the downloaded completed history.")
