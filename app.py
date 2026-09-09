import hashlib
import os
from datetime import date, timedelta
from io import BytesIO, StringIO
from urllib.parse import urlencode
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="India Momentum Engine", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# STRATEGY CONFIG
# -----------------------------------------------------------------------------
NSE_BHAV = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
BSE_BHAV = "https://www.bseindia.com/download/BhavCopy/Equity/EQ{ddmmyy}_CSV.ZIP"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}

MIN_AVG_VALUE_CR = 10.0
MIN_LATEST_VALUE_CR = 10.0
ATR_MIN = 1.0
ATR_MAX = 6.0
MOM30_MIN = 5.0
MOM50_MIN = 5.0
DISPLAY_CANDIDATES = 12
MAX_RISK_HOLDINGS = 12
MAX_SECTOR_DISPLAY = 2
MAX_LOSS_PCT = 5.0
WARNING_LOSS_PCT = 3.0

BASE = "https://api.kite.trade"
API_KEY = ""
API_SECRET = ""
REDIRECT_URL = ""


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
    checksum = hashlib.sha256((API_KEY + request_token + API_SECRET).encode()).hexdigest()
    r = requests.post(
        BASE + "/session/token",
        headers={"X-Kite-Version": "3"},
        data={"api_key": API_KEY, "request_token": request_token, "checksum": checksum},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["data"]["access_token"]


def kite_get(path, token, params=None, timeout=30):
    r = requests.get(BASE + path, headers=kite_headers(token), params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(j.get("message", "Kite API error"))
    return j["data"]


def request_file(url, timeout=30):
    r = requests.get(url, headers=NSE_HEADERS, timeout=timeout)
    if r.status_code != 200 or len(r.content) < 1000:
        raise RuntimeError(f"Download failed: HTTP {r.status_code}")
    return r.content


def read_zip_csv(blob):
    with ZipFile(BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not names:
            raise RuntimeError("No CSV/TXT inside ZIP")
        with z.open(names[0]) as f:
            return pd.read_csv(f, low_memory=False)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def pick_col(df, *names):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_day(d_iso):
    d = date.fromisoformat(d_iso)
    df = read_zip_csv(request_file(NSE_BHAV.format(yyyymmdd=d.strftime("%Y%m%d"))))
    sym = pick_col(df, "TckrSymb", "SYMBOL")
    isin = pick_col(df, "ISIN")
    series = pick_col(df, "SctySrs", "SERIES")
    open_ = pick_col(df, "OpnPric", "OPEN")
    close = pick_col(df, "ClsPric", "CLOSE")
    high = pick_col(df, "HghPric", "HIGH")
    low = pick_col(df, "LwPric", "LOW")
    vol = pick_col(df, "TtlTradgVol", "VOLUME")
    value = pick_col(df, "TtlTrfVal", "TOTTRDVAL")
    if not all([sym, close, high, low, vol]):
        raise RuntimeError(f"Unexpected NSE columns: {list(df.columns)[:15]}")
    if series:
        df = df[df[series].astype(str).str.upper().eq("EQ")].copy()
    out = pd.DataFrame(
        {
            "exchange": "NSE",
            "symbol": df[sym].astype(str).str.strip(),
            "isin": df[isin].astype(str).str.strip() if isin else "",
            "date": d,
            "open": num(df[open_]) if open_ else num(df[close]),
            "close": num(df[close]),
            "high": num(df[high]),
            "low": num(df[low]),
            "volume": num(df[vol]),
            "value": num(df[value]) if value else np.nan,
        }
    )
    return out[(out.close > 0) & out.high.notna() & out.low.notna()].copy()


@st.cache_data(ttl=86400, show_spinner=False)
def load_bse_day(d_iso):
    d = date.fromisoformat(d_iso)
    df = read_zip_csv(request_file(BSE_BHAV.format(ddmmyy=d.strftime("%d%m%y"))))
    sym = pick_col(df, "SYMBOL", "SC_CODE")
    isin = pick_col(df, "ISIN")
    group = pick_col(df, "SC_GROUP", "SERIES")
    open_ = pick_col(df, "OPEN", "OpnPric")
    close = pick_col(df, "CLOSE", "ClsPric")
    high = pick_col(df, "HIGH", "HghPric")
    low = pick_col(df, "LOW", "LwPric")
    vol = pick_col(df, "VOLUME", "TtlTradgVol")
    value = pick_col(df, "NET_TURNOV", "TtlTrfVal")
    if not all([sym, close, high, low, vol]):
        raise RuntimeError(f"Unexpected BSE columns: {list(df.columns)[:15]}")
    if group:
        allowed = {"A", "B", "T", "XT", "X", "Z", "ZP", "M", "P", "Q", "EQ"}
        df = df[df[group].astype(str).str.upper().isin(allowed)].copy()
    out = pd.DataFrame(
        {
            "exchange": "BSE",
            "symbol": df[sym].astype(str).str.strip(),
            "isin": df[isin].astype(str).str.strip() if isin else "",
            "date": d,
            "open": num(df[open_]) if open_ else num(df[close]),
            "close": num(df[close]),
            "high": num(df[high]),
            "low": num(df[low]),
            "volume": num(df[vol]),
            "value": num(df[value]) if value else np.nan,
        }
    )
    return out[(out.close > 0) & out.high.notna() & out.low.notna()].copy()


def weekdays(start, end):
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@st.cache_data(ttl=21600, show_spinner=False)
def load_history(start_iso, end_iso):
    start, end = date.fromisoformat(start_iso), date.fromisoformat(end_iso)
    dates = weekdays(start, end)
    nse, bse, failures = [], [], []
    for d in dates:
        try:
            nse.append(load_nse_day(d.isoformat()))
        except Exception as e:
            failures.append(f"NSE {d}: {e}")
        try:
            bse.append(load_bse_day(d.isoformat()))
        except Exception as e:
            failures.append(f"BSE {d}: {e}")
    if not nse:
        raise RuntimeError("NSE history could not be downloaded. NSE bhavcopy is required.")
    market = pd.concat(nse + bse, ignore_index=True)
    market["isin"] = market["isin"].replace({"nan": "", "None": ""}).fillna("")
    market["dedupe"] = np.where(
        market["isin"].ne(""), market["isin"], market["exchange"] + ":" + market["symbol"]
    )
    market["priority"] = np.where(market["exchange"].eq("NSE"), 0, 1)
    market = market.sort_values(["date", "dedupe", "priority"]).drop_duplicates(
        ["date", "dedupe"], keep="first"
    )
    market = market.sort_values(["dedupe", "date"]).reset_index(drop=True)
    return market, failures


def add_indicators(g):
    g = g.sort_values("date").copy()
    c = g.close.astype(float)
    h = g.high.astype(float)
    l = g.low.astype(float)
    v = g.volume.astype(float)
    g["sma20"] = c.rolling(20).mean()
    g["sma50"] = c.rolling(50).mean()
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14).mean()
    g["atr_pct"] = 100 * g.atr14 / c
    g["ret30"] = 100 * (c / c.shift(30) - 1)
    g["ret50"] = 100 * (c / c.shift(50) - 1)
    g["ret63"] = 100 * (c / c.shift(63) - 1)
    g["ret10"] = 100 * (c / c.shift(10) - 1)
    g["value_cr"] = c * v / 1e7
    g["avg20_value_cr"] = g.value_cr.rolling(20).mean()
    return g


def build_daily_candidates(market, asof, nifty_ret63=0.0, min_history=55):
    day = market[market.date.eq(asof)].copy()
    if day.empty:
        return pd.DataFrame()
    groups = market.groupby("dedupe", sort=False)
    rows = []
    for _, r in day.iterrows():
        try:
            g = groups.get_group(r.dedupe)
        except KeyError:
            continue
        if len(g) < min_history:
            continue
        g = add_indicators(g)
        x = g.iloc[-1]
        if pd.isna(x.ret50) or pd.isna(x.sma50) or pd.isna(x.avg20_value_cr):
            continue
        price = float(x.close)
        ret30, ret50 = float(x.ret30), float(x.ret50)
        if not (ret30 >= MOM30_MIN or ret50 >= MOM50_MIN):
            continue
        sma20, sma50 = float(x.sma20), float(x.sma50)
        sma50_5d = float(g.sma50.iloc[-6]) if len(g) >= 55 and pd.notna(g.sma50.iloc[-6]) else np.nan
        if not (price > sma20 > sma50 and sma50 > sma50_5d):
            continue
        atr = float(x.atr_pct)
        if not (ATR_MIN <= atr <= ATR_MAX):
            continue
        avg20 = float(x.avg20_value_cr)
        latest = float(x.value_cr)
        if avg20 < MIN_AVG_VALUE_CR or latest < MIN_LATEST_VALUE_CR:
            continue
        rs63 = float(x.ret63) - float(nifty_ret63) if pd.notna(x.ret63) else np.nan
        if pd.isna(rs63):
            continue
        rows.append(
            {
                "Stock": str(r.symbol),
                "Exchange": str(r.exchange),
                "ISIN": str(r.isin),
                "dedupe": str(r.dedupe),
                "Price": price,
                "30D %": ret30,
                "50D %": ret50,
                "63D %": float(x.ret63),
                "RS63 %": rs63,
                "10D %": float(x.ret10),
                "ATR %": atr,
                "Avg20 ₹Cr": avg20,
                "Latest ₹Cr": latest,
                "sma20": sma20,
                "sma50": sma50,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["RS Percentile"] = out["RS63 %"].rank(pct=True, method="average") * 100
    out["Score"] = (
        np.clip((out["30D %"] - MOM30_MIN) / 20, 0, 1) * 20
        + np.clip((out["50D %"] - MOM50_MIN) / 30, 0, 1) * 15
        + out["RS Percentile"] / 100 * 25
        + 20
        + np.clip((out["Avg20 ₹Cr"] - 10) / 50, 0, 1) * 10
        + np.where(out["ATR %"].between(1.5, 5.0), 10, 5)
    )
    return out.sort_values(["Score", "RS Percentile", "50D %"], ascending=False).reset_index(drop=True)


def nifty500_history(token, start, end):
    if not token:
        return pd.DataFrame()
    try:
        r = requests.get(BASE + "/instruments", timeout=60)
        r.raise_for_status()
        inst = pd.read_csv(StringIO(r.text))
        x = inst[(inst.exchange.eq("NSE")) & (inst.tradingsymbol.astype(str).str.upper() == "NIFTY 500")]
        if x.empty:
            return pd.DataFrame()
        data = kite_get(
            f"/instruments/historical/{int(x.iloc[0].instrument_token)}/day",
            token,
            {"from": start.isoformat(), "to": end.isoformat(), "oi": 0},
            30,
        )
        out = pd.DataFrame(data.get("candles", []), columns=["date", "open", "high", "low", "close", "volume"])
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
            out["close"] = pd.to_numeric(out["close"], errors="coerce")
        return out
    except Exception:
        return pd.DataFrame()


def nse_index_history(index_name, start, end):
    url = "https://www.nseindia.com/api/historical/indicesHistory"
    params = {"indexType": index_name, "from": start.strftime("%d-%m-%Y"), "to": end.strftime("%d-%m-%Y")}
    r = requests.get(url, headers=NSE_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    rows = j.get("data", [])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    date_col = next((c for c in out.columns if str(c).lower() in {"ch_timestamp", "date", "timestamp"}), None)
    close_col = next((c for c in out.columns if str(c).lower() in {"index_close", "close", "closeindex"}), None)
    if not date_col or not close_col:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce").dt.date
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    return out[["date", "close"]].dropna().sort_values("date")


def nse_sector_for_symbols(symbols):
    """Best-effort current NSE industry classification for displayed candidates."""
    if not symbols:
        return {}
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get("https://www.nseindia.com/", timeout=20)
    except Exception:
        pass
    result = {}
    for symbol in list(dict.fromkeys([str(x) for x in symbols])):
        try:
            r = session.get(
                "https://www.nseindia.com/api/equity-meta-info",
                params={"symbol": symbol},
                timeout=15,
            )
            if r.ok:
                j = r.json()
                data = j.get("data", j) if isinstance(j, dict) else {}
                industry = data.get("industry") or data.get("Industry") or data.get("sector") or data.get("Sector")
                if industry:
                    result[symbol] = str(industry)
        except Exception:
            continue
    return result


def add_sector_column(candidates):
    if candidates.empty:
        return candidates.copy(), False
    x = candidates.copy()
    sector_map = nse_sector_for_symbols(x["Stock"].tolist())
    x["Sector"] = x["Stock"].map(sector_map).fillna("Unknown")
    return x, bool(sector_map)


def sector_warning(selected):
    if selected.empty or "Sector" not in selected.columns:
        return pd.Series(dtype=int)
    return selected["Sector"].value_counts()


def regime_for_date(index_df, asof):
    if index_df.empty:
        return {"available": False, "above_200dma": None, "dma_rising": None, "breadth": None, "ok": None}
    x = index_df[index_df.date <= asof].copy().sort_values("date")
    if len(x) < 200:
        return {"available": False, "above_200dma": None, "dma_rising": None, "breadth": None, "ok": None}
    x["dma200"] = x.close.rolling(200).mean()
    last = x.iloc[-1]
    prev = x.iloc[-6]
    above = bool(last.close > last.dma200)
    rising = bool(last.dma200 > prev.dma200)
    return {"available": True, "above_200dma": above, "dma_rising": rising, "breadth": None, "ok": bool(above and rising)}


def last_three_sessions(market):
    return sorted(pd.unique(market["date"]))[-3:]


def latest_session(market):
    return max(market["date"]) if not market.empty else None


def build_risk_monitor(market, holdings):
    if not holdings:
        return pd.DataFrame()
    latest_date = max(market.date)
    groups = market.groupby("dedupe", sort=False)
    rows = []
    for h in holdings:
        dedupe = str(h.get("dedupe", ""))
        entry = float(h.get("entry_price", np.nan))
        name = str(h.get("Stock", dedupe))
        if not dedupe or not np.isfinite(entry) or entry <= 0:
            continue
        try:
            g = groups.get_group(dedupe).sort_values("date")
        except KeyError:
            continue
        if g.empty:
            continue
        g = add_indicators(g)
        x = g.iloc[-1]
        price = float(x.close)
        pnl = 100 * (price / entry - 1)
        stop_price = entry * (1 - MAX_LOSS_PCT / 100)
        warning_price = entry * (1 - WARNING_LOSS_PCT / 100)
        if price <= stop_price:
            status = "🔴 STOP / EXIT REVIEW"
        elif price <= warning_price:
            status = "🟠 WARNING"
        elif pd.notna(x.sma20) and price < float(x.sma20):
            status = "🟡 TREND WEAKENING"
        else:
            status = "🟢 HOLD / NO RISK TRIGGER"
        rows.append(
            {
                "Stock": name,
                "Latest Date": latest_date,
                "Entry Price": entry,
                "Latest Price": price,
                "P&L %": pnl,
                "5% Stop Price": stop_price,
                "3% Warning Price": warning_price,
                "SMA20": float(x.sma20) if pd.notna(x.sma20) else np.nan,
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("India Momentum Engine")
st.caption("Rolling 3 completed trading sessions → persistent momentum → your choice of up to 12 candidates")

with st.sidebar:
    st.header("Strategy")
    st.write("• NSE + BSE equities")
    st.write("• No ₹50 price floor")
    st.write("• 30D/50D momentum")
    st.write("• Trend + liquidity + ATR")
    st.write("• 63D relative-strength percentile")
    st.write("• Last 3 completed trading sessions")
    st.write("• Show up to 12 persistent candidates")
    st.write("• Sector concentration = warning, not forced rejection")
    st.write("• Nifty 500 regime = displayed")
    st.write("• Daily -3% warning / -5% hard risk trigger")
    st.divider()
    st.header("Zerodha")
    if API_KEY and API_SECRET and REDIRECT_URL:
        if st.session_state.get("access_token"):
            st.success("Zerodha connected")
            if st.button("Disconnect"):
                st.session_state.pop("access_token", None)
                st.rerun()
        else:
            st.link_button("LOGIN WITH ZERODHA", login_url(), use_container_width=True)
            st.caption("Optional: used as Nifty 500 benchmark-data fallback.")
    else:
        st.caption("Zerodha login is optional for this version.")

params = st.query_params
request_token = params.get("request_token")
if request_token and API_KEY and API_SECRET and not st.session_state.get("access_token"):
    try:
        st.session_state.access_token = exchange_request_token(request_token)
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")


tab1, tab2, tab3 = st.tabs(["MOMENTUM SCAN", "BACKTEST", "DAILY RISK CHECK"])

with tab1:
    st.subheader("Rolling momentum scan")
    st.caption(
        "Every time you run this, the app uses the latest 3 completed market sessions available. "
        "It does not use the first 3 days of a month and does not wait for a fixed calendar date."
    )
    run = st.button("RUN MOMENTUM SCAN", type="primary", use_container_width=True)

    if run:
        # 120 calendar days normally gives >50 trading sessions plus enough history for indicators.
        end = date.today()
        start = end - timedelta(days=120)
        try:
            with st.spinner("Downloading recent NSE/BSE closing data..."):
                market, failures = load_history(start.isoformat(), end.isoformat())
                st.session_state.last_market = market
        except Exception as e:
            st.error(f"Market-data load failed: {e}")
            st.stop()

        sessions = last_three_sessions(market)
        if len(sessions) < 3:
            st.error("Fewer than 3 completed market sessions are available. Try again later.")
            st.stop()

        token = st.session_state.get("access_token")
        index_df = nifty500_history(token, start, end) if token else pd.DataFrame()
        if index_df.empty:
            try:
                index_df = nse_index_history("NIFTY 500", start, end)
            except Exception as e:
                st.warning(f"Nifty 500 index history unavailable: {e}")
                index_df = pd.DataFrame()

        st.success(f"Latest completed session: {sessions[-1]}")
        st.write(f"### Sessions analysed: **{sessions[0]} | {sessions[1]} | {sessions[2]}**")

        daily = []
        cols = st.columns(3)
        for i, d in enumerate(sessions):
            nr = index_df[index_df.date <= d].sort_values("date") if not index_df.empty else pd.DataFrame()
            nret = 0.0 if len(nr) < 64 else 100 * (nr.close.iloc[-1] / nr.close.iloc[-64] - 1)
            c = build_daily_candidates(market[market.date <= d], d, nret)
            daily.append(c)
            cols[i].metric(f"{d}", f"{len(c)} candidates")

        if any(x.empty for x in daily):
            st.warning("At least one of the last 3 scans returned 0 qualifying stocks. Persistent set = 0.")
            st.session_state.monthly_selected_for_risk = []
        else:
            common = set(daily[0]["dedupe"])
            for x in daily[1:]:
                common &= set(x["dedupe"])
            st.write(f"### Persistent across all 3 sessions: **{len(common)} stocks**")

            if not common:
                st.warning("0 stocks appeared in all 3 sessions. No candidates to select.")
                st.session_state.monthly_selected_for_risk = []
            else:
                d3 = daily[-1].set_index("dedupe")
                candidates = d3.loc[d3.index.intersection(common)].copy()
                rs_parts = [x.set_index("dedupe")["RS Percentile"] for x in daily]
                candidates["Persistence RS"] = pd.concat(rs_parts, axis=1).mean(axis=1).reindex(candidates.index)
                candidates = candidates.sort_values(["Score", "Persistence RS"], ascending=False)
                candidates = candidates.head(DISPLAY_CANDIDATES).copy()
                candidates, sector_available = add_sector_column(candidates)
                candidates.insert(0, "Rank", range(1, len(candidates) + 1))

                st.info(
                    f"Showing the top {len(candidates)} persistent candidates. "
                    "There is no forced 5-stock limit — you choose which ones you want."
                )
                display_cols = [
                    "Rank", "Stock", "Exchange", "Sector", "Price", "30D %", "50D %", "63D %",
                    "RS63 %", "RS Percentile", "ATR %", "Avg20 ₹Cr", "Latest ₹Cr", "Score",
                ]
                st.dataframe(candidates[display_cols].round(2), use_container_width=True, hide_index=True)

                regime = regime_for_date(index_df, sessions[-1])
                st.subheader("Market regime")
                if regime["available"]:
                    st.write(
                        f"Nifty 500 > 200DMA: **{regime['above_200dma']}** | "
                        f"200DMA rising: **{regime['dma_rising']}**"
                    )
                    if regime["ok"]:
                        st.success("REGIME: ON")
                    else:
                        st.warning("REGIME: OFF — candidates remain visible for your decision; no automatic block.")
                else:
                    st.warning("Regime unavailable from the available index history.")

                if sector_available:
                    st.subheader("Sector concentration")
                    counts = candidates["Sector"].value_counts().rename_axis("Sector").reset_index(name="Candidates")
                    st.dataframe(counts, use_container_width=True, hide_index=True)
                    crowded = counts[(counts["Sector"] != "Unknown") & (counts["Candidates"] > MAX_SECTOR_DISPLAY)]
                    if not crowded.empty:
                        st.warning(
                            "Some sectors have more than 2 candidates. This is a warning only; "
                            "the app will not force your selection."
                        )

                st.subheader("Choose your stocks")
                options = candidates["Stock"].tolist()
                selected_names = st.multiselect(
                    "Select the stocks you actually want to hold",
                    options=options,
                    default=[],
                    max_selections=MAX_RISK_HOLDINGS,
                    key=f"selected_stocks_{sessions[-1]}",
                )
                if selected_names:
                    selected = candidates[candidates["Stock"].isin(selected_names)].copy()
                    selected["Allocation %"] = round(100 / len(selected), 2)
                    st.success(f"Your selection: {len(selected)} stock(s)")
                    st.dataframe(
                        selected[["Rank", "Stock", "Sector", "Price", "RS Percentile", "Score", "Allocation %"]].round(2),
                        use_container_width=True,
                        hide_index=True,
                    )
                    risk_selection = selected[["Stock", "Price"]].copy()
                    risk_selection.insert(1, "dedupe", selected.index.astype(str))
                    st.session_state.monthly_selected_for_risk = risk_selection.reset_index(drop=True).to_dict("records")
                    st.download_button(
                        "DOWNLOAD SELECTED STOCKS",
                        selected.to_csv(index=False).encode(),
                        "momentum_selected_stocks.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.caption("No stocks selected yet. You can choose from the 12-candidate list above.")
                    st.session_state.monthly_selected_for_risk = []

        if failures:
            st.caption(f"Non-fatal download failures: {len(failures)}. NSE data is required; BSE is best-effort.")

with tab2:
    st.subheader("Rolling 3-session backtest")
    st.caption(
        "Research only: for each selected signal date, use the preceding 3 completed sessions, "
        "take the persistent candidates, select the top 5 by score, enter next session, and exit at a chosen horizon."
    )
    bt_year = st.number_input(
        "Year", min_value=2018, max_value=date.today().year, value=max(2025, date.today().year - 1), step=1
    )
    bt_months = st.multiselect("Months", list(range(1, 13)), default=list(range(1, 13)))
    holding_days = st.number_input("Holding sessions", min_value=5, max_value=25, value=20, step=1)
    run_bt = st.button("RUN BACKTEST", type="primary", use_container_width=True)

    if run_bt:
        if not bt_months:
            st.error("Select at least one month.")
            st.stop()
        start = date(int(bt_year), min(bt_months), 1) - timedelta(days=140)
        max_month = max(bt_months)
        if max_month == 12:
            end = date(int(bt_year) + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(int(bt_year), max_month + 1, 1) - timedelta(days=1)
        try:
            with st.spinner("Downloading historical NSE/BSE data..."):
                market, failures = load_history(start.isoformat(), end.isoformat())
        except Exception as e:
            st.error(f"Backtest data load failed: {e}")
            st.stop()
        token = st.session_state.get("access_token")
        index_df = nifty500_history(token, start, end) if token else pd.DataFrame()
        if index_df.empty:
            try:
                index_df = nse_index_history("NIFTY 500", start, end)
            except Exception:
                index_df = pd.DataFrame()

        rows = []
        dates = sorted(set(market.date))
        for signal_date in dates:
            if signal_date.year != int(bt_year) or signal_date.month not in bt_months:
                continue
            pos = dates.index(signal_date)
            if pos < 3 or pos + holding_days >= len(dates):
                continue
            prev3 = dates[pos - 2 : pos + 1]
            future_exit = dates[pos + holding_days]
            daily = []
            for d in prev3:
                nr = index_df[index_df.date <= d].sort_values("date") if not index_df.empty else pd.DataFrame()
                nret = 0.0 if len(nr) < 64 else 100 * (nr.close.iloc[-1] / nr.close.iloc[-64] - 1)
                daily.append(build_daily_candidates(market[market.date <= d], d, nret))
            if any(c.empty for c in daily):
                continue
            common = set(daily[0]["dedupe"])
            for c in daily[1:]:
                common &= set(c["dedupe"])
            if not common:
                continue
            d3 = daily[-1].set_index("dedupe")
            candidates = d3.loc[d3.index.intersection(common)].sort_values("Score", ascending=False).head(5)
            entry_date_idx = pos + 1
            if entry_date_idx >= len(dates):
                continue
            entry_date = dates[entry_date_idx]
            entry = market[(market.date == entry_date) & market.dedupe.isin(candidates.index)].set_index("dedupe")["open"]
            exitp = market[(market.date == future_exit) & market.dedupe.isin(candidates.index)].set_index("dedupe")["close"]
            rets = []
            for idx in candidates.index:
                if idx in entry.index and idx in exitp.index and float(entry[idx]) > 0:
                    rets.append(100 * (float(exitp[idx]) / float(entry[idx]) - 1))
            if rets:
                rows.append(
                    {
                        "Signal Date": signal_date,
                        "Entry Date": entry_date,
                        "Exit Date": future_exit,
                        "Stocks": ", ".join(candidates["Stock"].tolist()),
                        "Stock Count": len(rets),
                        "Portfolio Return %": float(np.mean(rets)),
                    }
                )
        bt = pd.DataFrame(rows)
        if bt.empty:
            st.warning("No valid rolling 3-session signals could be evaluated for the selected period.")
        else:
            st.dataframe(bt.round(2), use_container_width=True, hide_index=True)
            st.metric("Average portfolio return", f"{bt['Portfolio Return %'].mean():.2f}%")
            st.metric("Win rate", f"{(bt['Portfolio Return %'] > 0).mean() * 100:.1f}%")
            st.metric("Worst signal", f"{bt['Portfolio Return %'].min():.2f}%")
            st.download_button(
                "DOWNLOAD BACKTEST",
                bt.to_csv(index=False).encode(),
                f"rolling_momentum_backtest_{bt_year}.csv",
                "text/csv",
                use_container_width=True,
            )
            if failures:
                st.caption(f"Non-fatal download failures: {len(failures)}.")

with tab3:
    st.subheader("Daily risk check")
    st.caption("Use this once per trading day after the market close. Enter your actual buy prices.")
    st.info(
        "Risk rule: -3% = early warning. -5% = hard risk trigger / exit review. "
        "A close below SMA20 is trend weakening, not an automatic exit."
    )

    saved = st.session_state.get("monthly_selected_for_risk", [])
    if saved:
        st.write("### Current selection")
        st.caption("Enter the actual price you bought each stock at.")
        risk_inputs = []
        for i, item in enumerate(saved):
            c1, c2 = st.columns([2, 2])
            c1.write(f"**{item['Stock']}**")
            default_entry = float(item.get("Price", 0))
            ep = c2.number_input(
                "Actual entry price",
                min_value=0.01,
                value=default_entry if default_entry > 0 else 1.0,
                key=f"risk_entry_{i}",
            )
            risk_inputs.append({"Stock": item["Stock"], "dedupe": item["dedupe"], "entry_price": ep})
        st.session_state.risk_holdings = risk_inputs

    upload = st.file_uploader("Optional: upload your selected-stock CSV", type=["csv"], key="risk_upload")
    if upload is not None:
        try:
            u = pd.read_csv(upload)
            if "Stock" not in u.columns or "Price" not in u.columns:
                st.error("CSV must contain Stock and Price columns.")
            else:
                latest_market = st.session_state.get("last_market")
                if latest_market is not None:
                    mp = latest_market[["dedupe", "symbol"]].drop_duplicates("symbol")
                    merged = u.merge(mp, left_on="Stock", right_on="symbol", how="left")
                    st.session_state.risk_holdings = [
                        {"Stock": r.Stock, "dedupe": r.dedupe, "entry_price": float(r.Price)}
                        for r in merged.itertuples()
                        if pd.notna(r.dedupe) and pd.notna(r.Price) and float(r.Price) > 0
                    ]
                    st.success(f"Loaded {len(st.session_state.risk_holdings)} holdings.")
                else:
                    st.warning("Run the momentum scan first so the app has the current market mapping.")
        except Exception as e:
            st.error(f"Could not read holdings CSV: {e}")

    run_risk = st.button("RUN DAILY RISK CHECK", type="primary", use_container_width=True)
    if run_risk:
        holdings = st.session_state.get("risk_holdings", [])
        if not holdings:
            st.warning("No holdings entered. Run the momentum scan first and enter your actual entry prices.")
        else:
            try:
                latest = date.today()
                start = latest - timedelta(days=15)
                end = latest
                with st.spinner("Downloading recent closing data..."):
                    risk_market, failures = load_history(start.isoformat(), end.isoformat())
                result = build_risk_monitor(risk_market, holdings)
                if result.empty:
                    st.warning("No matching recent market data was found for the entered holdings.")
                else:
                    st.write(f"### Latest completed session: **{max(risk_market.date)}**")
                    st.dataframe(result.round(2), use_container_width=True, hide_index=True)
                    hard = result[result["Status"].str.contains("STOP", na=False)]
                    warn = result[result["Status"].str.contains("WARNING|WEAKENING", regex=True, na=False)]
                    if not hard.empty:
                        st.error(
                            f"{len(hard)} holding(s) crossed the -5% hard risk threshold. "
                            "Review/exit promptly; do not wait for month-end."
                        )
                    elif not warn.empty:
                        st.warning(f"{len(warn)} holding(s) need attention. No -5% hard trigger yet.")
                    else:
                        st.success("No -5% hard risk trigger. Continue monitoring through the monthly plan.")
                    st.download_button(
                        "DOWNLOAD DAILY RISK REPORT",
                        result.to_csv(index=False).encode(),
                        "daily_risk_report.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    if failures:
                        st.caption(f"Non-fatal download failures: {len(failures)}.")
            except Exception as e:
                st.error(f"Daily risk check failed: {e}")
