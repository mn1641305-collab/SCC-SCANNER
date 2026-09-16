"""
SCC-style crypto candidate scanner.

Scans the top-volume Binance USDT pairs and flags symbols that show
BOTH:
  A) a clean trend regime (same spirit as the SCC indicator's Gate 1/2:
     ADX strength + EMA structure + not choppy), and
  B) an unusual volatility or volume spike (something is actually
     happening right now, not just a quiet trend).

This is a SCREENER, not a trading system: it only narrows down which
coins are worth opening in TradingView and running the real SCC
indicator on for a full, confirmed signal. It does not place orders
and does not replace SCC's own gates.

Designed to run periodically (e.g. every 30 min via GitHub Actions).
On each run it sends ONE Telegram digest message listing any fresh
candidates (a per-symbol cooldown avoids re-alerting the same coin
every single run).
"""

import os
import time
import json
import numpy as np
import pandas as pd
import requests
import ta

# ======================================================
# CONFIG
# ======================================================

BINANCE_BASE = "https://data-api.binance.vision"  # public market-data-only
                                                    # endpoint - not subject to
                                                    # the geo-restriction that
                                                    # blocks api.binance.com
                                                    # from US-hosted CI runners
                                                    # (e.g. GitHub Actions).
INTERVAL = "1h"          # candle timeframe for the scan
KLINES_LIMIT = 200       # history depth (enough for EMA50/ADX14/Chop14)
TOP_N_BY_VOLUME = 150    # universe size -> liquid pairs only

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
ADX_LEN = 14
MIN_ADX = 18.0
CHOP_LEN = 14
CHOP_LIMIT = 61.8
EMA_SPREAD_ATR_MIN = 0.25

VOL_SPIKE_MULT = 2.0      # current volume vs avg of last 20 closed bars
RANGE_SPIKE_ATR = 1.5     # current candle range vs ATR

STATE_FILE = "scanner_state.json"
ALERT_COOLDOWN_HOURS = 6  # don't re-alert the same symbol within this window

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ======================================================
# BINANCE DATA
# ======================================================

def get_top_symbols(n):
    tickers = requests.get(f"{BINANCE_BASE}/api/v3/ticker/24hr", timeout=20).json()
    exch = requests.get(f"{BINANCE_BASE}/api/v3/exchangeInfo", timeout=20).json()

    tradable = {
        s["symbol"] for s in exch["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
        and s.get("isSpotTradingAllowed", True)
    }

    usdt_pairs = [t for t in tickers if t["symbol"] in tradable]
    usdt_pairs.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in usdt_pairs[:n]]


def get_klines(symbol, interval, limit):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # The last row from Binance is usually the still-forming candle.
    # Drop it so the scan only ever looks at CLOSED bars (no lookahead,
    # same principle SCC itself uses with barstate.isconfirmed).
    return df.iloc[:-1].reset_index(drop=True)


# ======================================================
# INDICATORS
# ======================================================

def choppiness(df, length):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)

    tr_sum = tr.rolling(length).sum()
    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()
    width = (hh - ll).where(lambda w: w > 0)

    ratio = tr_sum / width
    return 100 * np.log10(ratio) / np.log10(length)


def analyze(symbol, df):
    if len(df) < EMA_TREND + 5:
        return None

    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    ema_fast = ta.trend.ema_indicator(close, window=EMA_FAST)
    ema_slow = ta.trend.ema_indicator(close, window=EMA_SLOW)
    ema_trend = ta.trend.ema_indicator(close, window=EMA_TREND)

    adx_ind = ta.trend.ADXIndicator(high, low, close, window=ADX_LEN)
    adx_val = adx_ind.adx()
    plus_di = adx_ind.adx_pos()
    minus_di = adx_ind.adx_neg()

    atr_val = ta.volatility.average_true_range(high, low, close, window=14)
    chop = choppiness(df, CHOP_LEN)

    c = close.iloc[-1]
    ema_f, ema_s, ema_t = ema_fast.iloc[-1], ema_slow.iloc[-1], ema_trend.iloc[-1]
    adx_now = adx_val.iloc[-1]
    pdi, mdi = plus_di.iloc[-1], minus_di.iloc[-1]
    atr_now = atr_val.iloc[-1]
    chop_now = chop.iloc[-1]

    if pd.isna(atr_now) or atr_now <= 0 or pd.isna(adx_now):
        return None

    ema_spread_atr = abs(ema_f - ema_s) / atr_now
    structure_ok = ema_spread_atr >= EMA_SPREAD_ATR_MIN
    strength_ok = adx_now >= MIN_ADX
    is_range = (not pd.isna(chop_now)) and chop_now >= CHOP_LIMIT

    bull_trend = (
        c > ema_t and ema_f > ema_s and pdi > mdi
        and strength_ok and structure_ok and not is_range
    )
    bear_trend = (
        c < ema_t and ema_f < ema_s and mdi > pdi
        and strength_ok and structure_ok and not is_range
    )

    regime_ok = bull_trend or bear_trend
    direction = "BULL" if bull_trend else ("BEAR" if bear_trend else "NONE")

    # --- volatility / volume spike ---
    avg_vol = vol.iloc[-21:-1].mean()
    vol_now = vol.iloc[-1]
    vol_ratio = (vol_now / avg_vol) if avg_vol > 0 else None
    vol_spike = vol_ratio is not None and vol_ratio >= VOL_SPIKE_MULT

    candle_range = high.iloc[-1] - low.iloc[-1]
    range_atr = candle_range / atr_now
    range_spike = range_atr >= RANGE_SPIKE_ATR

    volatility_ok = vol_spike or range_spike

    candidate = regime_ok and volatility_ok

    return {
        "symbol": symbol,
        "candidate": candidate,
        "direction": direction,
        "adx": round(adx_now, 1),
        "chop": round(chop_now, 1) if not pd.isna(chop_now) else None,
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "range_atr": round(range_atr, 2),
        "close": c,
    }


# ======================================================
# STATE / COOLDOWN
# ======================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ======================================================
# TELEGRAM
# ======================================================

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
              "missing) - printing instead:\n", text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }, timeout=20)

    if not resp.ok:
        print("Telegram send failed:", resp.status_code, resp.text)


# ======================================================
# MAIN
# ======================================================

def main():
    state = load_state()
    now = time.time()

    try:
        symbols = get_top_symbols(TOP_N_BY_VOLUME)
    except Exception as e:
        print(f"FATAL: could not reach Binance market-data API: {e}")
        print("If this is an HTTP 451 error, the runner's IP is being "
              "geo-blocked. This script already uses the "
              "data-api.binance.vision endpoint to avoid that - if it's "
              "still happening, Binance may have changed its rules again.")
        raise

    print(f"Scanning {len(symbols)} symbols on {INTERVAL}...")

    results = []
    for sym in symbols:
        try:
            df = get_klines(sym, INTERVAL, KLINES_LIMIT)
            r = analyze(sym, df)
            if r and r["candidate"]:
                results.append(r)
        except Exception as e:
            print(f"skip {sym}: {e}")
        time.sleep(0.05)  # stay well under Binance's rate limit

    fresh = []
    for r in results:
        last_alert = state.get(r["symbol"], 0)
        if now - last_alert >= ALERT_COOLDOWN_HOURS * 3600:
            fresh.append(r)
            state[r["symbol"]] = now

    if fresh:
        lines = [f"<b>SCC Scanner — {len(fresh)} candidate(s)</b> ({INTERVAL})"]
        for r in fresh:
            arrow = "🟢" if r["direction"] == "BULL" else "🔴"
            lines.append(
                f"{arrow} <b>{r['symbol']}</b>  "
                f"ADX {r['adx']}  Chop {r['chop']}  "
                f"Vol x{r['vol_ratio']}  Range {r['range_atr']}xATR  "
                f"@ {r['close']}"
            )
        send_telegram("\n".join(lines))
        print(f"Sent {len(fresh)} fresh candidate(s).")
    else:
        print("No fresh candidates this run.")

    save_state(state)


if __name__ == "__main__
