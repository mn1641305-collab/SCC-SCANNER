"""
SCC-style Binance USDⓈ-M Futures candidate scanner.

Purpose:
    Scan Binance USDⓈ-M PERPETUAL futures contracts and identify coins
    that currently show:

        A) A clean trend regime
           - EMA 9 / 21 / 50 structure
           - ADX strength
           - CHOP below threshold
           - minimum EMA spread relative to ATR

        B) An unusual activity condition
           - volume spike OR
           - candle range spike

This is ONLY a COIN SCREENER.

It does NOT:
    - place orders
    - calculate entry
    - calculate SL / TP
    - generate trading setups
    - replace the SCC TradingView indicator

The scanner's job is simply:
    "Which Binance USDⓈ-M Futures coins are worth opening
     in TradingView and checking with SCC?"

Data source:
    Binance USDⓈ-M Futures REST API

Universe:
    USDT-margined PERPETUAL contracts that are currently TRADING.

Designed for:
    GitHub Actions
    Periodic execution (e.g. every 30 minutes)

Telegram:
    Sends ONE digest containing fresh candidates.

State:
    scanner_state.json
    Prevents the same symbol from being alerted repeatedly
    during the cooldown period.
"""

import os
import time
import json
import tempfile

import numpy as np
import pandas as pd
import requests
import ta


# ======================================================
# CONFIG
# ======================================================

# Binance USDⓈ-M Futures
BINANCE_BASE = "https://fapi.binance.com"

# Candle timeframe used ONLY for screening
INTERVAL = "1h"

# Historical candles requested for each symbol
KLINES_LIMIT = 200

# Number of highest-volume Futures contracts to scan
TOP_N_BY_VOLUME = 150


# ------------------------------------------------------
# SCC-style regime parameters
# ------------------------------------------------------

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50

ADX_LEN = 14
MIN_ADX = 18.0

CHOP_LEN = 14
CHOP_LIMIT = 61.8

EMA_SPREAD_ATR_MIN = 0.25


# ------------------------------------------------------
# Activity / volatility parameters
# ------------------------------------------------------

# Current CLOSED candle volume compared with
# the average volume of the previous 20 CLOSED candles.
VOL_SPIKE_MULT = 2.0

# Current CLOSED candle range compared with ATR.
RANGE_SPIKE_ATR = 1.5


# ------------------------------------------------------
# Alert state
# ------------------------------------------------------

STATE_FILE = "scanner_state.json"

# Same symbol cannot alert again during this period.
ALERT_COOLDOWN_HOURS = 6


# ------------------------------------------------------
# HTTP settings
# ------------------------------------------------------

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

# Small delay between individual kline requests.
REQUEST_DELAY = 0.05


# ------------------------------------------------------
# Telegram
# ------------------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ======================================================
# HTTP HELPERS
# ======================================================

def request_json(url, params=None, label="Binance API"):
    """
    GET JSON with retry handling.

    Handles:
        - connection errors
        - timeouts
        - HTTP 429
        - HTTP 5xx
        - malformed JSON
        - Binance JSON error payloads
    """

    headers = {
        "User-Agent": "SCC-Futures-Scanner/1.0",
        "Accept": "application/json",
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            # --------------------------------------------------
            # Rate limit
            # --------------------------------------------------
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                try:
                    wait_seconds = float(retry_after)
                except (TypeError, ValueError):
                    wait_seconds = 5.0 * attempt

                print(
                    f"{label}: HTTP 429 rate limit. "
                    f"Waiting {wait_seconds:.1f}s "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )

                time.sleep(wait_seconds)
                continue

            # --------------------------------------------------
            # Temporary server errors
            # --------------------------------------------------
            if response.status_code >= 500:
                last_error = RuntimeError(
                    f"{label}: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

                if attempt < MAX_RETRIES:
                    wait_seconds = 2.0 * attempt
                    print(
                        f"{label}: server error "
                        f"{response.status_code}. "
                        f"Retrying in {wait_seconds:.1f}s..."
                    )
                    time.sleep(wait_seconds)
                    continue

                raise last_error

            # --------------------------------------------------
            # Other HTTP errors
            # --------------------------------------------------
            response.raise_for_status()

            # --------------------------------------------------
            # JSON
            # --------------------------------------------------
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"{label}: invalid JSON response: "
                    f"{response.text[:500]}"
                ) from exc

            # --------------------------------------------------
            # Binance error payload
            # Example:
            # {"code": -1121, "msg": "Invalid symbol."}
            # --------------------------------------------------
            if isinstance(data, dict):
                if "code" in data and "msg" in data:
                    raise RuntimeError(
                        f"{label}: Binance API error "
                        f"{data.get('code')}: {data.get('msg')}"
                    )

            return data

        except requests.RequestException as exc:
            last_error = exc

            if attempt < MAX_RETRIES:
                wait_seconds = 2.0 * attempt
                print(
                    f"{label}: request failed: {exc}. "
                    f"Retrying in {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)
            else:
                raise RuntimeError(
                    f"{label}: request failed after "
                    f"{MAX_RETRIES} attempts: {exc}"
                ) from exc

    raise RuntimeError(
        f"{label}: request failed: {last_error}"
    )


# ======================================================
# BINANCE USDⓈ-M FUTURES DATA
# ======================================================

def get_futures_exchange_info():
    """
    Get Binance USDⓈ-M Futures exchange information.

    Returns the full exchangeInfo object.
    """

    url = f"{BINANCE_BASE}/fapi/v1/exchangeInfo"

    data = request_json(
        url,
        label="Futures exchangeInfo",
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "Futures exchangeInfo returned unexpected "
            f"type: {type(data).__name__}"
        )

    if "symbols" not in data:
        raise RuntimeError(
            "Futures exchangeInfo response does not contain "
            f"'symbols'. Keys received: {list(data.keys())}. "
            f"Response: {str(data)[:1000]}"
        )

    if not isinstance(data["symbols"], list):
        raise RuntimeError(
            "Futures exchangeInfo 'symbols' is not a list."
        )

    return data


def get_top_symbols(n):
    """
    Return the top-N USDT-margined PERPETUAL Futures contracts
    ranked by 24h quote volume.

    Only currently TRADING contracts are included.
    """

    # --------------------------------------------------
    # Exchange information
    # --------------------------------------------------

    exch = get_futures_exchange_info()

    tradable = set()

    for s in exch["symbols"]:

        if not isinstance(s, dict):
            continue

        symbol = s.get("symbol")

        if not symbol:
            continue

        # USDⓈ-M Futures:
        # We specifically want USDT-margined perpetual contracts.
        if s.get("quoteAsset") != "USDT":
            continue

        if s.get("marginAsset") != "USDT":
            continue

        if s.get("contractType") != "PERPETUAL":
            continue

        if s.get("status") != "TRADING":
            continue

        tradable.add(symbol)

    if not tradable:
        raise RuntimeError(
            "No active USDT-margined PERPETUAL Futures "
            "contracts were found."
        )

    print(
        f"Found {len(tradable)} active USDT-margined "
        f"PERPETUAL Futures contracts."
    )

    # --------------------------------------------------
    # 24h Futures ticker
    # --------------------------------------------------

    ticker_url = f"{BINANCE_BASE}/fapi/v1/ticker/24hr"

    tickers = request_json(
        ticker_url,
        label="Futures 24h ticker",
    )

    if not isinstance(tickers, list):
        raise RuntimeError(
            "Futures 24h ticker returned unexpected "
            f"type: {type(tickers).__name__}"
        )

    # --------------------------------------------------
    # Filter + rank by quote volume
    # --------------------------------------------------

    usdt_pairs = []

    for ticker in tickers:

        if not isinstance(ticker, dict):
            continue

        symbol = ticker.get("symbol")

        if symbol not in tradable:
            continue

        try:
            quote_volume = float(
                ticker.get("quoteVolume", 0)
            )
        except (TypeError, ValueError):
            continue

        if not np.isfinite(quote_volume):
            continue

        if quote_volume <= 0:
            continue

        usdt_pairs.append(
            (symbol, quote_volume)
        )

    if not usdt_pairs:
        raise RuntimeError(
            "No valid USDT-margined Futures tickers "
            "were available after filtering."
        )

    usdt_pairs.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    selected = [
        symbol
        for symbol, _ in usdt_pairs[:n]
    ]

    return selected


def get_klines(symbol, interval, limit):
    """
    Get Futures klines and return only CLOSED candles.

    Binance returns the currently forming candle as the
    last row when the market is open.

    That candle is removed to avoid using unfinished data.
    """

    url = f"{BINANCE_BASE}/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    data = request_json(
        url,
        params=params,
        label=f"Klines {symbol}",
    )

    if not isinstance(data, list):
        raise RuntimeError(
            f"Klines {symbol}: unexpected response type "
            f"{type(data).__name__}"
        )

    if len(data) < 3:
        raise RuntimeError(
            f"Klines {symbol}: insufficient candle data "
            f"({len(data)} rows)"
        )

    rows = []

    for row in data:

        if not isinstance(row, list):
            continue

        if len(row) < 12:
            continue

        rows.append(row[:12])

    if len(rows) < 3:
        raise RuntimeError(
            f"Klines {symbol}: malformed candle data."
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    if len(df) < 3:
        raise RuntimeError(
            f"Klines {symbol}: no valid numeric candles."
        )

    # --------------------------------------------------
    # IMPORTANT:
    # Remove currently forming candle.
    # --------------------------------------------------

    return df.iloc[:-1].reset_index(drop=True)


# ======================================================
# INDICATORS
# ======================================================

def choppiness(df, length):
    """
    Choppiness Index.

    Higher values = more range/chop.
    Lower values = more directional behavior.
    """

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - df["close"].shift()
            ).abs(),
            (
                df["low"]
                - df["close"].shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(length).sum()

    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()

    width = hh - ll

    width = width.where(
        width > 0
    )

    ratio = tr_sum / width

    return (
        100
        * np.log10(ratio)
        / np.log10(length)
    )


# ======================================================
# SYMBOL ANALYSIS
# ======================================================

def analyze(symbol, df):

    minimum_history = max(
        EMA_TREND + 10,
        ADX_LEN * 3,
        CHOP_LEN + 10,
        30,
    )

    if len(df) < minimum_history:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # --------------------------------------------------
    # EMA
    # --------------------------------------------------

    ema_fast = ta.trend.ema_indicator(
        close,
        window=EMA_FAST,
    )

    ema_slow = ta.trend.ema_indicator(
        close,
        window=EMA_SLOW,
    )

    ema_trend = ta.trend.ema_indicator(
        close,
        window=EMA_TREND,
    )

    # --------------------------------------------------
    # ADX
    # --------------------------------------------------

    adx_ind = ta.trend.ADXIndicator(
        high,
        low,
        close,
        window=ADX_LEN,
    )

    adx_val = adx_ind.adx()
    plus_di = adx_ind.adx_pos()
    minus_di = adx_ind.adx_neg()

    # --------------------------------------------------
    # ATR
    # --------------------------------------------------

    atr_val = ta.volatility.average_true_range(
        high,
        low,
        close,
        window=14,
    )

    # --------------------------------------------------
    # CHOP
    # --------------------------------------------------

    chop = choppiness(
        df,
        CHOP_LEN,
    )

    # --------------------------------------------------
    # Latest CLOSED candle
    # --------------------------------------------------

    c = close.iloc[-1]

    ema_f = ema_fast.iloc[-1]
    ema_s = ema_slow.iloc[-1]
    ema_t = ema_trend.iloc[-1]

    adx_now = adx_val.iloc[-1]

    pdi = plus_di.iloc[-1]
    mdi = minus_di.iloc[-1]

    atr_now = atr_val.iloc[-1]
    chop_now = chop.iloc[-1]

    # --------------------------------------------------
    # Basic validity
    # --------------------------------------------------

    values = [
        c,
        ema_f,
        ema_s,
        ema_t,
        adx_now,
        pdi,
        mdi,
        atr_now,
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return None

    if atr_now <= 0:
        return None

    # --------------------------------------------------
    # EMA structure
    # --------------------------------------------------

    ema_spread_atr = (
        abs(ema_f - ema_s)
        / atr_now
    )

    structure_ok = (
        ema_spread_atr
        >= EMA_SPREAD_ATR_MIN
    )

    # --------------------------------------------------
    # ADX
    # --------------------------------------------------

    strength_ok = (
        adx_now >= MIN_ADX
    )

    # --------------------------------------------------
    # CHOP
    # --------------------------------------------------

    is_range = (
        not pd.isna(chop_now)
        and chop_now >= CHOP_LIMIT
    )

    # --------------------------------------------------
    # Bull trend
    # --------------------------------------------------

    bull_trend = (
        c > ema_t
        and ema_f > ema_s
        and pdi > mdi
        and strength_ok
        and structure_ok
        and not is_range
    )

    # --------------------------------------------------
    # Bear trend
    # --------------------------------------------------

    bear_trend = (
        c < ema_t
        and ema_f < ema_s
        and mdi > pdi
        and strength_ok
        and structure_ok
        and not is_range
    )

    regime_ok = (
        bull_trend
        or bear_trend
    )

    if bull_trend:
        direction = "BULL"
    elif bear_trend:
        direction = "BEAR"
    else:
        direction = "NONE"

    # ==================================================
    # ACTIVITY / VOLATILITY
    # ==================================================

    # Current candle is CLOSED.
    vol_now = vol.iloc[-1]

    # Previous 20 CLOSED candles.
    previous_volumes = vol.iloc[-21:-1]

    if len(previous_volumes) < 20:
        return None

    avg_vol = previous_volumes.mean()

    if (
        pd.isna(avg_vol)
        or avg_vol <= 0
    ):
        vol_ratio = None
    else:
        vol_ratio = (
            vol_now / avg_vol
        )

    vol_spike = (
        vol_ratio is not None
        and vol_ratio >= VOL_SPIKE_MULT
    )

    # --------------------------------------------------
    # Candle range vs ATR
    # --------------------------------------------------

    candle_range = (
        high.iloc[-1]
        - low.iloc[-1]
    )

    range_atr = (
        candle_range
        / atr_now
    )

    range_spike = (
        range_atr
        >= RANGE_SPIKE_ATR
    )

    # --------------------------------------------------
    # Final candidate
    # --------------------------------------------------

    volatility_ok = (
        vol_spike
        or range_spike
    )

    candidate = (
        regime_ok
        and volatility_ok
    )

    return {
        "symbol": symbol,
        "candidate": candidate,
        "direction": direction,

        "adx": round(
            float(adx_now),
            1,
        ),

        "chop": (
            round(
                float(chop_now),
                1,
            )
            if not pd.isna(chop_now)
            else None
        ),

        "vol_ratio": (
            round(
                float(vol_ratio),
                2,
            )
            if vol_ratio is not None
            else None
        ),

        "range_atr": round(
            float(range_atr),
            2,
        ),

        "close": float(c),
    }


# ======================================================
# STATE
# ======================================================

def load_state():

    if not os.path.exists(
        STATE_FILE
    ):
        return {}

    try:
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            state = json.load(f)

        if not isinstance(state, dict):
            return {}

        return state

    except (
        json.JSONDecodeError,
        OSError,
    ) as e:

        print(
            f"Warning: could not load state file: {e}"
        )

        return {}


def save_state(state):
    """
    Atomic state save.

    Writes to a temporary file first and then replaces
    scanner_state.json.

    This reduces the chance of corrupting the state file
    if the GitHub runner is interrupted during writing.
    """

    directory = (
        os.path.dirname(
            os.path.abspath(
                STATE_FILE
            )
        )
        or "."
    )

    fd, temp_path = tempfile.mkstemp(
        prefix=".scanner_state_",
        suffix=".tmp",
        dir=directory,
        text=True,
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                state,
                f,
                indent=2,
                sort_keys=True,
            )

        os.replace(
            temp_path,
            STATE_FILE,
        )

    except Exception:

        try:
            os.unlink(temp_path)
        except OSError:
            pass

        raise


def cleanup_state(state, now):
    """
    Remove very old entries so scanner_state.json does not
    grow forever.
    """

    max_age = (
        ALERT_COOLDOWN_HOURS
        * 3600
        * 10
    )

    cleaned = {}

    for symbol, timestamp in state.items():

        try:
            timestamp = float(timestamp)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            now - timestamp
            < max_age
        ):
            cleaned[symbol] = timestamp

    return cleaned


# ======================================================
# TELEGRAM
# ======================================================

def send_telegram(text):

    if (
        not TELEGRAM_TOKEN
        or not TELEGRAM_CHAT_ID
    ):
        print(
            "Telegram not configured "
            "(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
            "missing).\n"
            "Message would have been:\n"
            f"{text}"
        )

        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=20,
        )

        if not response.ok:

            print(
                "Telegram send failed:",
                response.status_code,
                response.text[:1000],
            )

            return False

        try:
            result = response.json()
        except ValueError:

            print(
                "Telegram returned invalid JSON:",
                response.text[:1000],
            )

            return False

        if not result.get(
            "ok",
            False,
        ):

            print(
                "Telegram API returned failure:",
                result,
            )

            return False

        return True

    except requests.RequestException as e:

        print(
            "Telegram request failed:",
            e,
        )

        return False


# ======================================================
# FORMATTING
# ======================================================

def format_price(price):

    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:,.4f}"

    if price >= 0.01:
        return f"{price:.5f}"

    if price >= 0.0001:
        return f"{price:.7f}"

    return f"{price:.10f}"


def build_telegram_message(fresh):

    lines = [
        (
            f"<b>SCC Futures Scanner</b> — "
            f"{len(fresh)} candidate(s) — {INTERVAL}"
        ),
        (
            "<i>Binance USDⓈ-M Perpetual Futures</i>"
        ),
        "",
    ]

    for r in fresh:

        if r["direction"] == "BULL":
            arrow = "🟢"
        else:
            arrow = "🔴"

        vol_text = (
            f"x{r['vol_ratio']}"
            if r["vol_ratio"] is not None
            else "n/a"
        )

        lines.append(
            f"{arrow} <b>{r['symbol']}</b>  "
            f"ADX {r['adx']}  "
            f"Chop {r['chop']}  "
            f"Vol {vol_text}  "
            f"Range {r['range_atr']}xATR  "
            f"@ {format_price(r['close'])}"
        )

    lines.append("")
    lines.append(
        "<i>Scanner only — "
        "confirm with SCC in TradingView.</i>"
    )

    return "\n".join(lines)


# ======================================================
# MAIN
# ======================================================

def main():

    start_time = time.time()

    state = load_state()

    now = time.time()

    state = cleanup_state(
        state,
        now,
    )

    # --------------------------------------------------
    # Get Futures universe
    # --------------------------------------------------

    symbols = get_top_symbols(
        TOP_N_BY_VOLUME
    )

    print(
        f"Scanning {len(symbols)} "
        f"Binance USDⓈ-M perpetual symbols "
        f"on {INTERVAL}..."
    )

    # --------------------------------------------------
    # Scan
    # --------------------------------------------------

    results = []

    failed_symbols = 0

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        try:

            df = get_klines(
                symbol,
                INTERVAL,
                KLINES_LIMIT,
            )

            result = analyze(
                symbol,
                df,
            )

            if (
                result is not None
                and result["candidate"]
            ):
                results.append(result)

        except Exception as e:

            failed_symbols += 1

            print(
                f"skip {symbol}: {e}"
            )

        # Small delay to avoid unnecessarily aggressive
        # request pacing.
        time.sleep(
            REQUEST_DELAY
        )

        if index % 25 == 0:
            print(
                f"Progress: "
                f"{index}/{len(symbols)}"
            )

    print(
        f"Scan complete. "
        f"Candidates: {len(results)}. "
        f"Failed symbols: {failed_symbols}."
    )

    # --------------------------------------------------
    # Cooldown
    # --------------------------------------------------

    fresh = []

    for result in results:

        symbol = result["symbol"]

        try:
            last_alert = float(
                state.get(
                    symbol,
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            last_alert = 0

        if (
            now - last_alert
            >= ALERT_COOLDOWN_HOURS * 3600
        ):

            fresh.append(result)

            state[symbol] = now

    # --------------------------------------------------
    # Telegram
    # --------------------------------------------------

    if fresh:

        message = build_telegram_message(
            fresh
        )

        sent = send_telegram(
            message
        )

        if sent:

            print(
                f"Sent {len(fresh)} "
                f"fresh candidate(s) to Telegram."
            )

        else:

            # IMPORTANT:
            # Do NOT permanently consume the cooldown
            # if Telegram failed.
            #
            # Otherwise a failed Telegram request could
            # suppress the next successful alert for 6h.

            for result in fresh:
                state.pop(
                    result["symbol"],
                    None,
                )

            print(
                "Telegram failed. "
                "Cooldown entries were rolled back."
            )

    else:

        print(
            "No fresh candidates this run."
        )

    # --------------------------------------------------
    # Save state
    # --------------------------------------------------

    save_state(state)

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Finished in {elapsed:.1f}s."
    )


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()