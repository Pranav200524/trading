"""
Options Trading Bot — Streamlit Dashboard
==========================================
Run with:  streamlit run dashboard/dashboard.py
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

# Prefer the slim dashboard DB when present (lets you deploy to Streamlit Cloud
# by committing only db/dashboard.db). Falls back to the full trading_bot.db for
# local dev where the export hasn't been generated.
_DB_DIR       = Path(__file__).parent / "db"
_DASHBOARD_DB = _DB_DIR / "dashboard.db"
_FULL_DB      = _DB_DIR / "trading_bot.db"
DB_PATH       = _DASHBOARD_DB if _DASHBOARD_DB.exists() else _FULL_DB

STATUS_COLORS = {
    "OPEN":             "#F5A623",
    "CLOSED_TARGET":    "#2ECC71",
    "CLOSED_STOPLOSS":  "#E74C3C",
    "CLOSED_EXPIRY":    "#8E8E93",
    "CLOSED_ROLLED":    "#3498DB",
}

# Display labels — what the user actually sees in the table.
STATUS_LABEL = {
    "OPEN":            "Open",
    "CLOSED_TARGET":   "Win",
    "CLOSED_STOPLOSS": "Loss",
    "CLOSED_EXPIRY":   "Expired",
    "CLOSED_ROLLED":   "Rolled",
}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_trades(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return _empty_trades_df()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """
            SELECT
                id, stock_name, option_symbol, option_type, mode, strategy_name,
                entry_time, entry_stock_price, entry_option_premium,
                exit_time, exit_stock_price, exit_option_premium,
                target_price, stoploss_price, expiry_date, status, pnl,
                cpr_pivot, cpr_bc, cpr_tc, cpr_r2, cpr_r3, cpr_s2, cpr_s3
            FROM trades_executed
            ORDER BY entry_time DESC
            """,
            conn
        )
        conn.close()
        return _enrich(df)
    except Exception as e:
        st.error(f"Database error: {e}")
        return _empty_trades_df()


@st.cache_data(ttl=30)
def load_ml_zones(db_path: Path, stock_name: str | None = None) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        if stock_name:
            df = pd.read_sql_query(
                "SELECT * FROM v_latest_zones WHERE stock_name = ? ORDER BY rank_order",
                conn, params=(stock_name,)
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM v_latest_zones ORDER BY stock_name, rank_order",
                conn
            )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_cpr_levels(db_path: Path, symbol: str, year: int, month: int) -> dict | None:
    """Compute CPR pivot levels from previous month's daily OHLC."""
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        if month == 1:
            prev_year, prev_month = year - 1, 12
        else:
            prev_year, prev_month = year, month - 1

        first_day = f"{prev_year}-{prev_month:02d}-01"
        if prev_month == 12:
            last_day = f"{prev_year}-12-31"
        else:
            last_day = f"{prev_year}-{prev_month + 1:02d}-01"

        row = conn.execute(
            """SELECT MAX(high) AS h, MIN(low) AS l,
                      (SELECT close FROM equity_daily_ohlc
                       WHERE symbol = ? AND date < ? ORDER BY date DESC LIMIT 1) AS c
               FROM equity_daily_ohlc
               WHERE symbol = ? AND date >= ? AND date < ?""",
            (symbol, last_day, symbol, first_day, last_day)
        ).fetchone()
        conn.close()

        if row is None or row[0] is None:
            return None

        h, l, c = row
        pivot = (h + l + c) / 3
        bc = (h + l) / 2
        tc = (pivot * 2) - bc
        r1 = (2 * pivot) - l
        r2 = pivot + (h - l)
        r3 = h + 2 * (pivot - l)
        s1 = (2 * pivot) - h
        s2 = pivot - (h - l)
        s3 = l - 2 * (h - pivot)
        tub = (r2 + r3) / 2
        tlb = (s2 + s3) / 2

        return {
            "Prev Month High": h, "Prev Month Low": l, "Prev Month Close": c,
            "Pivot": pivot, "BC": bc, "TC": tc,
            "R1": r1, "R2": r2, "R3": r3,
            "S1": s1, "S2": s2, "S3": s3,
            "TUB (PUT trigger)": tub, "TLB (CALL trigger)": tlb,
        }
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_symbols(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT DISTINCT stock_name FROM ml_sr_zones ORDER BY stock_name"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=60)
def load_daily_prices(db_path: Path, symbol: str, days: int = 365) -> pd.DataFrame:
    """Daily close prices for the most recent {days} trading days."""
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close
               FROM equity_daily_ohlc
               WHERE symbol = ?
               ORDER BY date DESC LIMIT ?""",
            conn, params=(symbol, days)
        )
        conn.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_daily_symbols(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM equity_daily_ohlc ORDER BY symbol"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Stored as UTC ISO timestamps — convert to IST for display.
    df["entry_time"] = (
        pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
          .dt.tz_convert("Asia/Kolkata")
    )
    df["exit_time"]  = (
        pd.to_datetime(df["exit_time"],  errors="coerce", utc=True)
          .dt.tz_convert("Asia/Kolkata")
    )
    df["duration_mins"] = (
        (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
    ).round().astype("Int64")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    return df


def _empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "stock_name", "option_symbol", "option_type", "mode", "strategy_name",
        "entry_time", "entry_stock_price", "entry_option_premium",
        "exit_time", "exit_stock_price", "exit_option_premium",
        "target_price", "stoploss_price", "expiry_date", "status", "pnl",
        "duration_mins"
    ])


# ── Sidebar Filters ────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filters")
    st.sidebar.markdown("---")

    stocks = ["All"] + sorted(df["stock_name"].dropna().unique().tolist())
    selected_stock = st.sidebar.selectbox("Underlying Asset", stocks)

    mode_options = ["All", "LIVE", "BACKTEST"]
    selected_mode = st.sidebar.radio("Execution Mode", mode_options)

    # Strategy filter — works for both LIVE and BACKTEST modes
    strategy_options = ["All"] + sorted(df["strategy_name"].dropna().unique().tolist())
    selected_strategy = st.sidebar.selectbox("Strategy", strategy_options)

    status_options = ["OPEN", "CLOSED_TARGET", "CLOSED_STOPLOSS",
                      "CLOSED_EXPIRY", "CLOSED_ROLLED"]
    selected_status = st.sidebar.multiselect(
        "Status", status_options, default=status_options
    )

    # ── Date range filter ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range")

    if not df.empty and df["entry_time"].notna().any():
        min_date = df["entry_time"].min().date()
        max_date = df["entry_time"].max().date()
    else:
        min_date = max_date = datetime.today().date()

    date_filter_on = st.sidebar.checkbox(
        "Apply date filter", value=False,
        help="When off, all trades across all dates are shown.",
    )

    date_basis = "Entry time"
    date_range = (min_date, max_date)
    if date_filter_on:
        date_basis = st.sidebar.radio(
            "Filter by", ["Entry time", "Exit time"], horizontal=True
        )
        date_range = st.sidebar.date_input(
            "From / To",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range",
        )
        if st.sidebar.button("Clear dates"):
            st.session_state.pop("date_range", None)
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"DB: `{DB_PATH.name}`")
    st.sidebar.caption("Auto-refreshes every 10 s")

    filtered = df.copy()
    if selected_stock != "All":
        filtered = filtered[filtered["stock_name"] == selected_stock]
    if selected_mode != "All":
        filtered = filtered[filtered["mode"] == selected_mode]
    if selected_strategy != "All":
        filtered = filtered[filtered["strategy_name"] == selected_strategy]
    if selected_status:
        filtered = filtered[filtered["status"].isin(selected_status)]

    # Apply date filter only if the checkbox is on
    if date_filter_on:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d_from, d_to = date_range
        else:
            d_from = d_to = date_range if not isinstance(date_range, tuple) else date_range[0]

        col = "entry_time" if date_basis == "Entry time" else "exit_time"
        if not filtered.empty and col in filtered.columns:
            start_ts = pd.Timestamp(d_from, tz="Asia/Kolkata")
            end_ts   = pd.Timestamp(d_to,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
            mask = filtered[col].between(start_ts, end_ts, inclusive="left")
            if col == "exit_time":
                mask = mask | filtered[col].isna()
            filtered = filtered[mask]

    return filtered


# ── KPI Summary Row ────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame) -> None:
    total      = len(df)
    open_pos   = df[df["status"] == "OPEN"]
    wins       = df[df["status"] == "CLOSED_TARGET"]
    losses     = df[df["status"] == "CLOSED_STOPLOSS"]
    rolled     = df[df["status"] == "CLOSED_ROLLED"]
    decided    = len(wins) + len(losses)          # trades with a real W/L outcome
    win_rate   = (len(wins) / decided * 100) if decided > 0 else 0.0
    avg_dur    = df[df["status"] != "OPEN"]["duration_mins"].mean() if total > 0 else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades", total)
    c2.metric("Wins", len(wins))
    c3.metric("Losses", len(losses))
    c4.metric("Win Rate", f"{win_rate:.1f}%",
              help=f"{len(wins)} wins / {decided} decided (excludes {len(rolled)} rolled, {len(open_pos)} open)")
    c5.metric("Rolled", len(rolled))
    c6.metric("Avg Duration", f"{avg_dur:.0f} min" if not pd.isna(avg_dur) else "—")


# ── Trade Table ────────────────────────────────────────────────────────────────

def render_trade_table(df: pd.DataFrame) -> None:
    st.markdown("### Trade History")

    if df.empty:
        st.info("No trades match the current filters.")
        return

    display_cols = {
        "stock_name":            "Stock",
        "option_symbol":         "Option Contract",
        "option_type":           "Type",
        "strategy_name":         "Strategy",
        "mode":                  "Mode",
        "status":                "Status",
        "entry_time":            "Entry Time",
        "entry_stock_price":     "Entry Spot",
        "entry_option_premium":  "Entry Premium",
        "exit_time":             "Exit Time",
        "exit_stock_price":      "Exit Spot",
        "exit_option_premium":   "Exit Premium",
        "pnl":                   "P&L (pts)",
        "duration_mins":         "Duration (min)",
        "expiry_date":           "Expiry",
    }

    view = df[list(display_cols.keys())].rename(columns=display_cols).copy()
    # Friendly status labels (Win / Loss / Open / Rolled / Expired)
    view["Status"] = view["Status"].map(STATUS_LABEL).fillna(view["Status"])

    label_to_raw = {v: k for k, v in STATUS_LABEL.items()}
    def colour_status(val: str) -> str:
        raw = label_to_raw.get(val, val)
        bg = STATUS_COLORS.get(raw, "#ffffff")
        return f"background-color: {bg}20; color: {bg}; font-weight: bold;"

    # Use Styler.map (pandas ≥2.1). Fall back to applymap on older pandas.
    style_map = getattr(view.style, "map", None) or view.style.applymap
    styled = style_map(colour_status, subset=["Status"])
    styled = styled.format({
        "Entry Time":     lambda x: x.strftime("%Y-%m-%d %H:%M IST") if pd.notna(x) else "—",
        "Exit Time":      lambda x: x.strftime("%Y-%m-%d %H:%M IST") if pd.notna(x) else "—",
        "Entry Spot":     "{:.2f}",
        "Entry Premium":  "{:.2f}",
        "Exit Spot":      lambda x: f"{x:.2f}" if pd.notna(x) else "—",
        "Exit Premium":   lambda x: f"{x:.2f}" if pd.notna(x) else "—",
        "P&L (pts)":      lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
        "Duration (min)": lambda x: f"{int(x)}" if pd.notna(x) else "—",
    })

    st.dataframe(styled, use_container_width=True, height=450)


# ── PnL Chart ─────────────────────────────────────────────────────────────────

def render_pnl_chart(df: pd.DataFrame) -> None:
    closed = df[df["status"] != "OPEN"].copy()
    if closed.empty:
        return

    closed = closed.sort_values("exit_time")
    closed["cumulative_pnl"] = closed["pnl"].cumsum()

    st.markdown("### Cumulative P&L")
    st.line_chart(
        closed.set_index("exit_time")["cumulative_pnl"],
        use_container_width=True,
        height=250,
    )


# ── ML Zones Panel ────────────────────────────────────────────────────────────

def render_ml_zones() -> None:
    symbols = load_symbols(DB_PATH)
    if not symbols:
        st.info("No ML zones in database. Run ml/sr_analysis.py first.")
        return

    selected = st.selectbox("Select Symbol", symbols, key="zone_symbol")
    df = load_ml_zones(DB_PATH, selected)

    if df.empty:
        st.caption(f"No ML zones found for {selected}.")
        return

    sup = df[df["zone_type"].isin(["support"])].reset_index(drop=True)
    res = df[df["zone_type"].isin(["resistance"])].reset_index(drop=True)
    rev = df[df["zone_type"] == "role_reversal"].reset_index(drop=True)

    col_s, col_r = st.columns(2)
    with col_s:
        st.markdown("**Support Zones**")
        if not sup.empty:
            st.dataframe(
                sup[["zone_price", "zone_low", "zone_high", "strength_score", "rank_order", "method"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.caption("None")
    with col_r:
        st.markdown("**Resistance Zones**")
        if not res.empty:
            st.dataframe(
                res[["zone_price", "zone_low", "zone_high", "strength_score", "rank_order", "method"]],
                use_container_width=True, hide_index=True
            )
        else:
            st.caption("None")

    if not rev.empty:
        st.markdown("**Role Reversal Zones**")
        st.dataframe(
            rev[["zone_price", "zone_low", "zone_high", "original_type", "flipped_to",
                 "strength_score", "rank_order"]],
            use_container_width=True, hide_index=True
        )


# ── CPR Pivot Levels Panel ────────────────────────────────────────────────────

def render_zone_visualizer() -> None:
    symbols = load_symbols(DB_PATH)
    if not symbols:
        st.info("No ML zones in database. Run `python3 ml/sr_analysis.py` first.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Symbol", symbols, key="zviz_symbol")
    with col2:
        lookback = st.selectbox("Lookback",
                                ["3M", "6M", "1Y", "2Y"], index=2, key="zviz_lookback")

    days_map = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    prices = load_daily_prices(DB_PATH, selected, days=days_map[lookback])
    zones  = load_ml_zones(DB_PATH, selected)

    if prices.empty:
        st.warning(f"No daily price data for {selected} in `equity_daily_ohlc`.")
        return

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=prices["date"],
        open=prices["open"], high=prices["high"],
        low=prices["low"],   close=prices["close"],
        name=selected,
        increasing_line_color="#2ECC71",
        decreasing_line_color="#E74C3C",
    ))

    # Zone bands as colored horizontal rectangles
    type_color = {
        "support":       "rgba(46, 204, 113, 0.18)",
        "resistance":    "rgba(231, 76, 60, 0.18)",
        "role_reversal": "rgba(52, 152, 219, 0.18)",
    }
    type_line = {
        "support":       "#2ECC71",
        "resistance":    "#E74C3C",
        "role_reversal": "#3498DB",
    }
    x0, x1 = prices["date"].iloc[0], prices["date"].iloc[-1]

    annotations = []
    for _, z in zones.iterrows():
        ztype = z["zone_type"]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x0, x1=x1,
            y0=z["zone_low"], y1=z["zone_high"],
            fillcolor=type_color.get(ztype, "rgba(150,150,150,0.18)"),
            line=dict(color=type_line.get(ztype, "#888"), width=1),
            layer="below",
        )
        label = f"{ztype.upper()} @ {z['zone_price']:.2f} (score={z['strength_score']:.1f})"
        if ztype == "role_reversal" and pd.notna(z.get("flipped_to")):
            label += f" → {z['flipped_to']}"
        annotations.append(dict(
            x=x1, y=z["zone_price"],
            xref="x", yref="y",
            text=label,
            showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(size=10, color=type_line.get(ztype, "#888")),
            bgcolor="rgba(0,0,0,0.0)",
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        annotations=annotations,
        title=f"{selected} — {len(zones)} ML zones",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabular zone breakdown below the chart
    if not zones.empty:
        st.markdown("**Zones**")
        cols_to_show = ["zone_type", "zone_price", "zone_low", "zone_high",
                        "strength_score", "touches", "rank_order", "method"]
        if "flipped_to" in zones.columns:
            cols_to_show.append("flipped_to")
        st.dataframe(
            zones[cols_to_show].sort_values(["zone_type", "rank_order"]),
            use_container_width=True, hide_index=True
        )


def render_cpr_levels() -> None:
    symbols = load_daily_symbols(DB_PATH)
    if not symbols:
        st.info("No daily OHLC data. Run `python3 ml/fetch_backtest_data.py --phase daily` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        selected = st.selectbox("Select Symbol", symbols, key="cpr_symbol")
    with col2:
        now = datetime.now()
        year = st.number_input("Year", min_value=2020, max_value=2030, value=now.year, key="cpr_year")
    with col3:
        month = st.number_input("Month", min_value=1, max_value=12, value=now.month, key="cpr_month")

    levels = load_cpr_levels(DB_PATH, selected, int(year), int(month))
    if levels is None:
        st.warning(f"No OHLC data for the previous month to compute CPR for {selected} ({year}-{month:02d}).")
        return

    st.markdown(f"### CPR Levels for {selected} — {year}-{month:02d}")
    st.caption("Computed from previous month's High/Low/Close")

    col_r, col_p, col_s = st.columns(3)
    with col_r:
        st.markdown("**Resistance Levels**")
        st.metric("R3", f"{levels['R3']:.2f}")
        st.metric("TUB (PUT trigger)", f"{levels['TUB (PUT trigger)']:.2f}")
        st.metric("R2", f"{levels['R2']:.2f}")
        st.metric("R1", f"{levels['R1']:.2f}")

    with col_p:
        st.markdown("**Central Pivot Range**")
        st.metric("TC (Top Central)", f"{levels['TC']:.2f}")
        st.metric("Pivot", f"{levels['Pivot']:.2f}")
        st.metric("BC (Bottom Central)", f"{levels['BC']:.2f}")
        st.markdown("---")
        st.caption(f"Prev Month: H={levels['Prev Month High']:.2f}  L={levels['Prev Month Low']:.2f}  C={levels['Prev Month Close']:.2f}")

    with col_s:
        st.markdown("**Support Levels**")
        st.metric("S1", f"{levels['S1']:.2f}")
        st.metric("S2", f"{levels['S2']:.2f}")
        st.metric("TLB (CALL trigger)", f"{levels['TLB (CALL trigger)']:.2f}")
        st.metric("S3", f"{levels['S3']:.2f}")

    all_levels = pd.DataFrame([
        {"Level": k, "Price": v}
        for k, v in levels.items()
        if k not in ("Prev Month High", "Prev Month Low", "Prev Month Close")
    ]).sort_values("Price", ascending=False).reset_index(drop=True)

    with st.expander("All Levels (sorted by price)"):
        st.dataframe(all_levels, use_container_width=True, hide_index=True)


# ── App Shell ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Options Trading Bot")
    st.caption("Real-time trade monitor & backtest analyser")
    st.markdown("---")

    raw_df   = load_trades(DB_PATH)
    filtered = render_sidebar(raw_df)

    render_kpis(filtered)
    st.markdown("---")

    tab_trades, tab_chart, tab_zviz, tab_zones, tab_cpr = st.tabs(
        ["Trades", "P&L Chart", "Zone Chart", "ML Zones", "CPR Pivot Levels"]
    )

    with tab_trades:
        render_trade_table(filtered)

    with tab_chart:
        render_pnl_chart(filtered)

    with tab_zviz:
        render_zone_visualizer()

    with tab_zones:
        render_ml_zones()

    with tab_cpr:
        render_cpr_levels()


if __name__ == "__main__":
    main()
