"""
03_dashboard.py
===============
Exoplanet Explorer — Dash + Plotly dashboard backed by PostgreSQL.

Usage:
    python 03_dashboard.py \
        --host localhost \
        --port 5432 \
        --dbname your_database \
        --user your_username

Environment variables (alternative to CLI flags):
    EXODB_HOST, EXODB_PORT, EXODB_NAME, EXODB_USER, EXODB_PASS

Requirements:
    pip install dash dash-bootstrap-components plotly psycopg2-binary pandas
"""

import argparse
import getpass
import os

import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables when deployed on Render,
# falls back to CLI args / interactive prompts for local development.
# ---------------------------------------------------------------------------
def get_db_url():
    # On Render, DATABASE_URL is set automatically when you attach a Postgres DB
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # Render provides postgres:// but SQLAlchemy needs postgresql://
        return database_url.replace("postgres://", "postgresql+psycopg2://", 1)

    # Local development: use CLI args or env vars
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default=os.getenv("EXODB_HOST", "localhost"))
    parser.add_argument("--port",     type=int, default=int(os.getenv("EXODB_PORT", 5432)))
    parser.add_argument("--dbname",   default=os.getenv("EXODB_NAME", ""))
    parser.add_argument("--user",     default=os.getenv("EXODB_USER", ""))
    parser.add_argument("--password", default=os.getenv("EXODB_PASS", None))
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args() 

    if not args.dbname:
        args.dbname = input("Database name: ")
    if not args.user:
        args.user = input("Username: ")
    if not args.password:
        args.password = getpass.getpass(f"Password for {args.user}@{args.host}: ")

    return (
        f"postgresql+psycopg2://{args.user}:{args.password}"
        f"@{args.host}:{args.port}/{args.dbname}"
    )

engine = create_engine(get_db_url(), pool_size=5, max_overflow=2)


def query(sql: str, params=None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
METHODS = query("SELECT DISTINCT discoverymethod FROM clean_exoplanets WHERE discoverymethod IS NOT NULL ORDER BY 1")["discoverymethod"].tolist()
YEAR_RANGE = query("SELECT MIN(disc_year) AS mn, MAX(disc_year) AS mx FROM clean_exoplanets WHERE disc_year IS NOT NULL")
YEAR_MIN = int(YEAR_RANGE["mn"].iloc[0])
YEAR_MAX = int(YEAR_RANGE["mx"].iloc[0])

METHOD_COLORS = px.colors.qualitative.Bold


def get_kpis():
    return query("SELECT * FROM exoplanet_summary").iloc[0]


def get_timeline(methods):
    where = _method_where(methods)
    return query(f"""
        SELECT disc_year, discoverymethod, COUNT(*) AS count
        FROM clean_exoplanets
        WHERE disc_year IS NOT NULL {where}
        GROUP BY disc_year, discoverymethod
        ORDER BY disc_year
    """)


def get_scatter(methods, year_range):
    where = _method_where(methods)
    return query(f"""
        SELECT pl_name, hostname, pl_rade, pl_bmasse, discoverymethod,
               disc_year, pl_eqt, in_habitable_zone
        FROM clean_exoplanets
        WHERE pl_rade IS NOT NULL AND pl_bmasse IS NOT NULL {where}
          AND disc_year BETWEEN %s AND %s
    """, (year_range[0], year_range[1]))


def get_orbital(methods, year_range):
    where = _method_where(methods)
    return query(f"""
        SELECT pl_name, pl_orbper, pl_orbsmax, discoverymethod, pl_rade
        FROM clean_exoplanets
        WHERE pl_orbper IS NOT NULL AND pl_orbsmax IS NOT NULL {where}
          AND disc_year BETWEEN %s AND %s
          AND pl_orbper > 0 AND pl_orbsmax > 0
    """, (year_range[0], year_range[1]))


def get_method_pie(methods):
    where = _method_where(methods)
    return query(f"""
        SELECT discoverymethod, COUNT(*) AS count
        FROM clean_exoplanets
        WHERE discoverymethod IS NOT NULL {where}
        GROUP BY discoverymethod
        ORDER BY count DESC
    """)


# Famous multi-planet systems to feature in the Hall of Fame grid
HALL_OF_FAME = [
    "KOI-351",      # 8 planets (Kepler-90) — most known in any exo-system
    "TRAPPIST-1",  # 7 planets, 3 in habitable zone
    "Kepler-11",   # 6 tightly-packed planets
    "55 Cnc",      # 5 planets including a super-Earth
    "Kepler-20",   # 6 planets
    "HR 8799",     # 4 directly imaged giant planets
]

def get_multi_planet_stars():
    """Return only stars with 3+ known planets, sorted by planet count desc."""
    return query("""
        SELECT hostname, COUNT(*) AS planet_count
        FROM clean_exoplanets
        WHERE hostname IS NOT NULL AND pl_orbsmax IS NOT NULL
        GROUP BY hostname
        HAVING COUNT(*) >= 3
        ORDER BY planet_count DESC, hostname
    """)


def get_system(hostname):
    return query("""
        SELECT pl_name, pl_orbsmax, pl_rade, pl_eqt, pl_orbper,
               dist_ly
        FROM clean_exoplanets
        WHERE hostname = %s AND pl_orbsmax IS NOT NULL
        ORDER BY pl_orbsmax
    """, (hostname,))


def get_distance_ly(hostname):
    """Return distance in light years for a given host star."""
    result = query("""
        SELECT dist_ly
        FROM clean_exoplanets
        WHERE hostname = %s AND dist_ly IS NOT NULL
        LIMIT 1
    """, (hostname,))
    if result.empty:
        return "Unknown"
    return f"{result['dist_ly'].iloc[0]:,.0f} ly"


def _method_where(methods):
    if not methods:
        return ""
    escaped = ", ".join(f"'{m}'" for m in methods)
    return f" AND discoverymethod IN ({escaped})"


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def kpi_card(title, value, icon, color, subtitle=None):
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.Div(icon, className="kpi-icon", style={"fontSize": "2rem"}),
            html.H2(f"{value:,}", className="kpi-value",
                    style={"color": color, "margin": "0.25rem 0", "fontWeight": 800}),
            html.P(title, className="kpi-label",
                   style={"margin": 0, "opacity": 0.7, "fontSize": "0.85rem"}),
            html.P(subtitle, style={"margin": "0.3rem 0 0", "fontSize": "0.72rem",
                   "color": "#f472b6", "lineHeight": "1.4"}) if subtitle else None,
        ])
    ], className="kpi-card"), md=3, sm=6, xs=12)


PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(color="#cbd5e1", family="'Courier New', monospace"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)

GRID_STYLE = dict(
    gridcolor="rgba(255,255,255,0.25)",
    zerolinecolor="rgba(255,255,255,0.45)",
    gridwidth=1,
)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap",
    ],
    title="🌌 Exoplanet Explorer",
)
server = app.server  # exposed for gunicorn

# ---------------------------------------------------------------------------
# Discovery method descriptions for the Background tab
# ---------------------------------------------------------------------------
METHODS_INFO = {
    "Transit": {
        "icon": "🔭", "color": "#38bdf8",
        "short": "Planet passes in front of its star",
        "detail": (
            "When a planet crosses (transits) in front of its host star, it blocks a tiny "
            "fraction of the star's light. Astronomers detect this as a periodic dip in "
            "brightness. The depth of the dip reveals the planet's size, and the interval "
            "between dips reveals its orbital period. The Kepler and TESS space telescopes "
            "have used this method to discover the vast majority of known exoplanets."
        ),
        "pros": "Detects small planets · Reveals planet size · Works at great distances",
        "cons": "Requires edge-on orbit · Many false positives · Confirms ~1 in 10 candidates",
        "example": "Kepler-90 system (8 planets)",
    },
    "Radial Velocity": {
        "icon": "〰️", "color": "#818cf8",
        "short": "Planet's gravity wobbles its star toward/away from Earth",
        "detail": (
            "A planet's gravity pulls on its host star, causing the star to wobble slightly "
            "toward and away from Earth. This wobble causes a measurable Doppler shift in the "
            "star's light — stretching to redder wavelengths when moving away, shifting to "
            "bluer wavelengths when moving toward us. The size of the wobble reveals the "
            "planet's minimum mass. This method confirmed the first exoplanet around a "
            "Sun-like star — 51 Pegasi b in 1995."
        ),
        "pros": "Confirms planet mass · Works for nearby bright stars · Proven technique",
        "cons": "Biased toward massive planets · Only gives minimum mass · Needs stable spectrograph",
        "example": "51 Pegasi b — first confirmed exoplanet around a Sun-like star (1995)",
    },
    "Imaging": {
        "icon": "📷", "color": "#34d399",
        "short": "Planet photographed directly next to its star",
        "detail": (
            "In rare cases, astronomers can directly photograph a planet by blocking out the "
            "blinding light of its host star using a coronagraph. This is extraordinarily "
            "difficult — like photographing a firefly next to a searchlight from miles away. "
            "Direct imaging works best for young, massive planets far from their stars, as "
            "these planets still emit their own infrared glow. The James Webb Space Telescope "
            "has dramatically improved direct imaging capabilities."
        ),
        "pros": "Can study planet atmosphere · Sees planets far from their stars",
        "cons": "Extremely difficult · Only works for large young distant planets · Very few detections",
        "example": "HR 8799 system (4 directly imaged giant planets)",
    },
    "Microlensing": {
        "icon": "🔍", "color": "#fbbf24",
        "short": "Planet's gravity bends light from a background star",
        "detail": (
            "When a star with a planet passes in front of a more distant background star, "
            "the gravity of the foreground star and its planet bends and magnifies the "
            "background star's light — like a cosmic magnifying glass. If a planet is "
            "present, it creates a brief secondary spike in brightness. Microlensing can "
            "detect planets at great distances, including planets that other methods miss."
        ),
        "pros": "Detects distant and low-mass planets · Sensitive to planets other methods miss",
        "cons": "Event never repeats · Cannot follow up · Requires rare star alignments",
        "example": "OGLE-2005-BLG-390Lb — cool super-Earth 21,500 light years away",
    },
    "Astrometry": {
        "icon": "📐", "color": "#f472b6",
        "short": "Planet wobbles its star's position sideways across the sky",
        "detail": (
            "Similar to radial velocity, astrometry detects a planet by measuring the tiny "
            "wobble it induces in its host star — but instead of measuring the wobble toward "
            "and away from us, it measures the wobble side to side across the sky. This "
            "requires extraordinarily precise positional measurements. The Gaia space "
            "telescope is expected to discover thousands of planets using this method."
        ),
        "pros": "Gives true planet mass · Complementary to radial velocity",
        "cons": "Needs extremely precise measurements · Only practical from space · Very few detections",
        "example": "Gaia space telescope expected to yield thousands of discoveries",
    },
    "Pulsar Timing": {
        "icon": "⏱️", "color": "#fb7185",
        "short": "Planet disturbs a pulsar's extraordinarily precise timing",
        "detail": (
            "Pulsars are rapidly rotating neutron stars that emit radio pulses with "
            "extraordinary regularity — more precise than atomic clocks. If a planet orbits "
            "a pulsar, its gravity causes tiny variations in the arrival times of the pulses. "
            "This method led to the very first confirmed exoplanet discovery in 1992 "
            "(PSR 1257+12), predating the 51 Pegasi b discovery by three years. However, "
            "pulsar planets are bathed in intense radiation and are unlikely to support life."
        ),
        "pros": "Extremely sensitive · Led to the very first exoplanet discovery",
        "cons": "Only works around pulsars · Intense radiation environment · Very rare",
        "example": "PSR 1257+12 — the very first confirmed exoplanets (1992)",
    },
}


kpis = get_kpis()
multi_stars_df = get_multi_planet_stars()
multi_stars = multi_stars_df["hostname"].tolist()
# Default to TRAPPIST-1 if available, else first in list
default_star = "TRAPPIST-1" if "TRAPPIST-1" in multi_stars else (multi_stars[0] if multi_stars else None)

# ---------------------------------------------------------------------------
# Background tab content
# ---------------------------------------------------------------------------
def make_background_tab():
    method_cards = []
    for method, info in METHODS_INFO.items():
        method_cards.append(
            dbc.Col(dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.Span(info["icon"], style={"fontSize": "1.5rem"}), width="auto"),
                        dbc.Col([
                            html.Div(method, style={
                                "fontFamily": "'Orbitron', sans-serif",
                                "fontSize": "0.85rem", "color": info["color"],
                                "fontWeight": 700, "letterSpacing": "0.05em",
                            }),
                            html.Div(info["short"], style={
                                "fontSize": "0.78rem", "color": "#94a3b8",
                                "fontFamily": "'Share Tech Mono', monospace",
                            }),
                        ]),
                    ], align="center", className="g-2"),
                    style={"background": "rgba(30,41,59,0.9)", "border": "none",
                           "borderLeft": f"3px solid {info['color']}",
                           "padding": "0.6rem 0.75rem"},
                ),
                dbc.CardBody([
                    html.P(info["detail"], style={
                        "fontSize": "0.85rem", "color": "#cbd5e1",
                        "lineHeight": "1.7", "marginBottom": "0.75rem",
                    }),
                    dbc.Row([
                        dbc.Col([
                            html.Div("✅ Strengths", style={
                                "fontSize": "0.72rem", "color": "#34d399",
                                "fontWeight": 700, "marginBottom": "0.2rem",
                                "fontFamily": "'Share Tech Mono', monospace",
                            }),
                            html.P(info["pros"], style={
                                "fontSize": "0.78rem", "color": "#94a3b8",
                                "lineHeight": "1.6", "margin": 0,
                            }),
                        ], md=6),
                        dbc.Col([
                            html.Div("⚠️ Limitations", style={
                                "fontSize": "0.72rem", "color": "#fbbf24",
                                "fontWeight": 700, "marginBottom": "0.2rem",
                                "fontFamily": "'Share Tech Mono', monospace",
                            }),
                            html.P(info["cons"], style={
                                "fontSize": "0.78rem", "color": "#94a3b8",
                                "lineHeight": "1.6", "margin": 0,
                            }),
                        ], md=6),
                    ], className="g-2"),
                    html.Hr(style={"borderColor": "rgba(255,255,255,0.07)", "margin": "0.75rem 0"}),
                    html.Div([
                        html.Span("🌟 Notable Example: ", style={
                            "fontSize": "0.75rem", "color": "#818cf8",
                            "fontWeight": 700,
                            "fontFamily": "'Share Tech Mono', monospace",
                        }),
                        html.Span(info["example"], style={
                            "fontSize": "0.75rem", "color": "#cbd5e1",
                            "fontFamily": "'Share Tech Mono', monospace",
                        }),
                    ]),
                ], style={"padding": "1rem"}),
            ], style={"height": "100%",
                      "border": "1px solid rgba(255,255,255,0.07)",
                      "background": "rgba(15,23,42,0.85)"}),
            md=6, className="mb-3")
        )

    return dbc.Container([
        # Intro section
        dbc.Row([
            dbc.Col([
                html.H3("The Search for Exoplanets", style={
                    "fontFamily": "'Orbitron', sans-serif",
                    "color": "#38bdf8", "fontWeight": 700,
                    "letterSpacing": "0.08em", "marginBottom": "1rem",
                }),
                html.P([
                    "The search for planets beyond our Solar System is one of the newest and most "
                    "exciting fields in modern astronomy. Although astronomers long suspected that "
                    "other stars must have planets, the technology to detect them simply did not "
                    "exist until the 1990s. The first confirmed exoplanet discoveries came in ",
                    html.Strong("1992", style={"color": "#f472b6"}),
                    " (around a pulsar) and ",
                    html.Strong("1995", style={"color": "#f472b6"}),
                    " (around a Sun-like star) — meaning this entire field of astronomy is less "
                    "than 35 years old. Today, thanks to dedicated space telescopes like Kepler "
                    "and TESS, we have confirmed over ",
                    html.Strong("5,700 exoplanets", style={"color": "#38bdf8"}),
                    " with thousands more candidates awaiting confirmation.",
                ], style={"fontSize": "0.95rem", "color": "#cbd5e1",
                          "lineHeight": "1.8", "marginBottom": "1rem"}),
                html.P([
                    "No single method can detect all types of planets. Each technique has its own "
                    "strengths, limitations, and blind spots — which is why astronomers use many "
                    "different approaches, often confirming a planet with multiple methods. "
                    "The six primary discovery methods used in this dataset are described below."
                ], style={"fontSize": "0.95rem", "color": "#94a3b8",
                          "lineHeight": "1.8", "marginBottom": "1.5rem",
                          "borderLeft": "3px solid #818cf8",
                          "paddingLeft": "1rem"}),

                # Naming convention note
                dbc.Alert([
                    html.Span("🔤  ", style={"fontSize": "1.1rem"}),
                    html.Strong("A note on planet naming: ",
                                style={"color": "#fbbf24", "fontFamily": "'Share Tech Mono', monospace"}),
                    "Exoplanets are lettered starting at ",
                    html.Strong("b", style={"color": "#38bdf8"}),
                    " (the host star is considered ",
                    html.Strong("a", style={"color": "#38bdf8"}),
                    "), but the letters reflect the ",
                    html.Strong("order of discovery", style={"color": "#f472b6"}),
                    ", not distance from the star. This means the innermost planet is not "
                    "always ",
                    html.Strong("b", style={"color": "#38bdf8"}),
                    ". For example, in the ",
                    html.Strong("55 Cancri", style={"color": "#38bdf8"}),
                    " system the innermost planet is ",
                    html.Strong("55 Cnc e", style={"color": "#38bdf8"}),
                    " — discovered after b, c, and d. In ",
                    html.Strong("HR 8799", style={"color": "#38bdf8"}),
                    ", planet ",
                    html.Strong("b", style={"color": "#38bdf8"}),
                    " is actually the ",
                    html.Em("outermost"),
                    " planet, imaged first because it was the easiest to see. "
                    "The orbital diagrams in this dashboard always plot planets by true "
                    "distance from their star, regardless of letter designation.",
                ], color="dark", style={
                    "backgroundColor": "rgba(30,41,59,0.8)",
                    "border": "1px solid rgba(251,191,36,0.3)",
                    "borderLeft": "4px solid #fbbf24",
                    "color": "#cbd5e1",
                    "fontSize": "0.88rem",
                    "lineHeight": "1.7",
                    "marginBottom": "2rem",
                    "borderRadius": "8px",
                }),
            ], md=12),
        ]),
        # Method cards
        html.H4("Discovery Methods", style={
            "fontFamily": "'Orbitron', sans-serif",
            "color": "#e2e8f0", "fontSize": "1rem",
            "letterSpacing": "0.08em", "marginBottom": "1rem",
        }),
        dbc.Row(method_cards, className="g-3"),
    ], fluid=True, style={"padding": "1.5rem 1rem 3rem"})


app.layout = html.Div([

    # ── Stars background canvas ────────────────────────────────────────────
    html.Canvas(id="starfield", style={
        "position": "fixed", "top": 0, "left": 0,
        "width": "100vw", "height": "100vh",
        "zIndex": -1, "pointerEvents": "none",
    }),

    # ── Header ─────────────────────────────────────────────────────────────
    html.Div([
        html.H1("🌌 EXOPLANET EXPLORER", style={
            "fontFamily": "'Orbitron', sans-serif",
            "fontWeight": 900, "letterSpacing": "0.15em",
            "background": "linear-gradient(90deg, #38bdf8, #818cf8, #f472b6)",
            "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
            "margin": 0, "fontSize": "clamp(1.4rem, 4vw, 2.5rem)",
        }),
        html.P("NASA Confirmed Exoplanets · Interactive Observatory",
               style={"color": "#94a3b8", "fontFamily": "'Share Tech Mono', monospace",
                      "fontSize": "0.95rem", "margin": "0.25rem 0 0 0", "letterSpacing": "0.1em"}),
    ], style={"textAlign": "center", "padding": "2rem 1rem 1rem"}),

    # ── Tabs ───────────────────────────────────────────────────────────────
    dbc.Tabs([
        dbc.Tab(label="📊 Dashboard", tab_id="tab-dashboard",
                label_style={"fontFamily": "'Share Tech Mono', monospace",
                             "fontSize": "0.9rem", "color": "#94a3b8"},
                active_label_style={"fontFamily": "'Share Tech Mono', monospace",
                                    "fontSize": "0.9rem", "color": "#38bdf8",
                                    "fontWeight": "700"}),
        dbc.Tab(label="📖 Background", tab_id="tab-background",
                label_style={"fontFamily": "'Share Tech Mono', monospace",
                             "fontSize": "0.9rem", "color": "#94a3b8"},
                active_label_style={"fontFamily": "'Share Tech Mono', monospace",
                                    "fontSize": "0.9rem", "color": "#38bdf8",
                                    "fontWeight": "700"}),
    ], id="main-tabs", active_tab="tab-dashboard",
       style={"borderBottom": "1px solid rgba(255,255,255,0.1)",
              "padding": "0 1rem",
              "backgroundColor": "rgba(15,23,42,0.6)"}),

    html.Div(id="tab-content"),
], style={"minHeight": "100vh"})


# Inject CSS and starfield JS via index_string (replaces html.Style / html.Script)
app.index_string = """
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
body { background: #020817 !important; color: #cbd5e1; }
.filter-label { font-family: 'Share Tech Mono', monospace; font-size: 0.92rem;
                letter-spacing: 0.08em; color: #e2e8f0; font-weight: 600; margin-bottom: 0.4rem; }
.chart-title  { font-family: 'Orbitron', sans-serif; font-size: 0.9rem;
                letter-spacing: 0.05em; margin-bottom: 0.5rem; color: #e2e8f0; }
.kpi-card     { background: rgba(30,41,59,0.8) !important;
                border: 1px solid rgba(255,255,255,0.07) !important;
                border-radius: 12px !important; text-align: center; backdrop-filter: blur(4px); }
.kpi-value    { font-family: 'Orbitron', sans-serif; }
.rc-slider-handle, .rc-slider-handle-dragging { background-color: #818cf8 !important; border-color: #818cf8 !important; }
.rc-slider-tooltip-inner { background-color: #1e293b !important; color: #e2e8f0 !important; font-size: 0.9rem !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.25) !important; box-shadow: none !important; padding: 4px 10px !important; }
.rc-slider-tooltip-arrow { border-top-color: #1e293b !important; }
.rc-slider-mark-text { color: #e2e8f0 !important; font-size: 0.82rem !important; font-weight: 500; }
.rc-slider-mark-text-active { color: #38bdf8 !important; }
/* Endpoint min/max input boxes */
.dash-slider-input.dash-range-slider-min-input,
.dash-slider-input.dash-range-slider-max-input,
.dash-range-slider-min-input,
.dash-range-slider-max-input {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 4px !important;
}
/* Slider tooltip content */
[id*="slider-tooltip"][id*="content"],
[id*="year-slider-tooltip"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 4px !important;
}
/* Tick marks on the scale */
.rc-slider-dot {
    display: block !important;
    border-color: #64748b !important;
    background-color: #64748b !important;
    width: 6px !important; height: 6px !important;
    bottom: -2px !important;
}
.rc-slider-dot-active {
    border-color: #818cf8 !important;
    background-color: #818cf8 !important;
}
.card         { background: rgba(15,23,42,0.85) !important;
                border: 1px solid rgba(255,255,255,0.07) !important;
                border-radius: 12px !important; backdrop-filter: blur(8px); }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
(function() {
    var canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;';
    document.body.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    var stars = [];
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        stars = Array.from({length: 180}, function() { return {
            x: Math.random() * canvas.width, y: Math.random() * canvas.height,
            r: Math.random() * 1.4 + 0.2, a: Math.random(),
            da: (Math.random() - 0.5) * 0.003
        }; });
    }
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        stars.forEach(function(s) {
            s.a = Math.max(0.05, Math.min(1, s.a + s.da));
            if (s.a <= 0.05 || s.a >= 1) s.da *= -1;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255,255,255,' + s.a + ')';
            ctx.fill();
        });
        requestAnimationFrame(draw);
    }
    window.addEventListener('resize', resize);
    resize(); draw();
})();
</script>
</body>
</html>
"""



# ---------------------------------------------------------------------------
# Dashboard tab content
# ---------------------------------------------------------------------------
def make_dashboard_content():
    kpis = get_kpis()
    return html.Div([

        # ── Global filters ──────────────────────────────────────────────
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Label("Discovery Method", className="filter-label"),
                    dcc.Dropdown(
                        id="method-filter",
                        options=[{"label": m, "value": m} for m in METHODS],
                        value=METHODS,
                        multi=True,
                        placeholder="All methods…",
                        style={"backgroundColor": "#1e293b", "color": "#cbd5e1"},
                    ),
                ], md=7),
                dbc.Col([
                    html.Label(f"Discovery Year  ({YEAR_MIN}–{YEAR_MAX})", className="filter-label"),
                    dcc.RangeSlider(
                        id="year-slider",
                        min=YEAR_MIN, max=YEAR_MAX, step=1,
                        value=[YEAR_MIN, YEAR_MAX],
                        marks={y: {"label": str(y), "style": {"color": "#e2e8f0", "fontSize": "0.82rem", "fontWeight": "500"}} for y in range(YEAR_MIN, YEAR_MAX + 1, 5) if y not in (YEAR_MIN, YEAR_MAX)},
                        tooltip={"placement": "bottom"},
                    ),
                ], md=5),
            ], className="g-3", style={"padding": "0.5rem 0 1.5rem"}),
        ], fluid=True),

        # ── KPI cards ───────────────────────────────────────────────────
        dbc.Container([
            dbc.Row([
                kpi_card("Total Planets",        kpis["total_planets"],       "🪐", "#38bdf8"),
                kpi_card("Host Stars",           kpis["total_stars"],         "⭐", "#fbbf24"),
                kpi_card("Multi-planet Systems", kpis["multi_planet_systems"],"🌍", "#34d399"),
                kpi_card("In Habitable Zone",    kpis["in_habitable_zone"],   "💧", "#f472b6",
                         subtitle="Equilibrium temp 200–320 K · where liquid water may exist"),
            ], className="g-3"),
        ], fluid=True, style={"padding": "0 1rem 1.5rem"}),

        # ── Row 1: Timeline + Pie ───────────────────────────────────────
        dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Discoveries Per Year", className="chart-title"),
                    dcc.Graph(id="timeline-chart", config={"displayModeBar": False},
                              style={"height": "320px"}),
                ])), md=8),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("By Discovery Method", className="chart-title"),
                    dcc.Graph(id="pie-chart", config={"displayModeBar": False},
                              style={"height": "320px"}),
                ])), md=4),
            ], className="g-3"),
        ], fluid=True, style={"padding": "0 1rem 1rem"}),

        # ── Row 2: Scatter + Orbital ────────────────────────────────────
        dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Mass vs. Radius", className="chart-title"),
                    html.P("Hover for details · Habitable-zone planets marked ✦ · sized by discovery method",
                           style={"fontSize": "0.82rem", "color": "#cbd5e1", "margin": "-0.25rem 0 0.5rem"}),
                    dcc.Graph(id="scatter-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ])), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Orbital Period vs. Semi-major Axis", className="chart-title"),
                    html.P("Log scale · Kepler's Third Law emerges naturally — planets farther from their star take longer to orbit",
                           style={"fontSize": "0.82rem", "color": "#cbd5e1", "margin": "-0.25rem 0 0.5rem"}),
                    dcc.Graph(id="orbital-chart", config={"displayModeBar": False},
                              style={"height": "380px"}),
                ])), md=6),
            ], className="g-3"),
        ], fluid=True, style={"padding": "0 1rem 1rem"}),

        # ── Row 3: Hall of Fame ─────────────────────────────────────────
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H5("🏆 Hall of Fame — Most Remarkable Multi-Planet Systems & Distance from Earth (light years)",
                                className="chart-title"), md=12),
            ], style={"padding": "0 0 0.5rem"}),
            dbc.Row(
                [dbc.Col(dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col(html.Span(name, style={
                                "fontFamily": "'Orbitron', sans-serif",
                                "fontSize": "0.75rem", "letterSpacing": "0.05em",
                                "color": "#38bdf8",
                            }), width="auto"),
                            dbc.Col(html.Span(id=f"hof-dist-{name.replace(' ','-')}",
                                style={"fontSize": "0.7rem", "color": "#94a3b8",
                                       "fontFamily": "'Share Tech Mono', monospace"}),
                                width="auto", className="ms-auto"),
                        ], align="center", className="g-0"),
                        style={
                            "background": "rgba(30,41,59,0.9)",
                            "border": "none", "padding": "0.4rem 0.75rem",
                        }),
                    dbc.CardBody(
                        dcc.Graph(id=f"hof-{name.replace(' ','-')}",
                                  config={"displayModeBar": False},
                                  style={"height": "180px"}),
                        style={"padding": "0.25rem"},
                    ),
                ], style={"border": "1px solid rgba(255,255,255,0.07)"}),
                md=4, sm=6, xs=12, className="mb-3")
                for name in HALL_OF_FAME],
            className="g-2"),
        ], fluid=True, style={"padding": "0 1rem 1rem"}),

        # ── Row 4: Multi-Planet Explorer ────────────────────────────────
        dbc.Container([
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("🔭 Multi-Planet System Explorer", className="chart-title"),
                            html.P([
                                "Orbital diagram showing confirmed planets (3+ per star). ",
                                html.Br(),
                                html.Strong("Concentric rings", style={"color": "#38bdf8"}),
                                " = orbital paths · ",
                                html.Strong("Dot size", style={"color": "#38bdf8"}),
                                " = planet radius · ",
                                html.Strong("Dot colour", style={"color": "#38bdf8"}),
                                " = surface temperature (blue=cool, red=hot)",
                            ], style={"fontSize": "0.82rem", "color": "#cbd5e1",
                                      "margin": "-0.25rem 0 0.5rem", "lineHeight": "1.6"}),
                        ], md=7),
                        dbc.Col(dcc.Dropdown(
                            id="star-dropdown",
                            options=[
                                {"label": f"{row.hostname}  ({row.planet_count} planets)",
                                 "value": row.hostname}
                                for row in multi_stars_df.itertuples()
                            ],
                            value=default_star,
                            clearable=False,
                            style={"backgroundColor": "#1e293b"},
                        ), md=5),
                    ], align="center"),
                    dcc.Graph(id="system-chart", config={"displayModeBar": False},
                              style={"height": "450px"}),
                ])), md=12),
            ], className="g-3"),
        ], fluid=True, style={"padding": "0 1rem 2rem"}),
    ])


# ---------------------------------------------------------------------------
# Tab switching callback
# ---------------------------------------------------------------------------
@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
)
def render_tab(active_tab):
    if active_tab == "tab-background":
        return make_background_tab()
    return make_dashboard_content()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("timeline-chart", "figure"),
    Input("method-filter", "value"),
    Input("year-slider", "value"),
)
def update_timeline(methods, year_range):
    df = get_timeline(methods)
    df = df[df["disc_year"].between(year_range[0], year_range[1])]
    fig = px.bar(df, x="disc_year", y="count", color="discoverymethod",
                 color_discrete_sequence=METHOD_COLORS,
                 labels={"disc_year": "Year", "count": "Planets", "discoverymethod": "Method"})
    fig.update_layout(**PLOT_LAYOUT, barmode="stack",
                      xaxis=dict(title="", **GRID_STYLE),
                      yaxis=dict(title="Discoveries", **GRID_STYLE))
    return fig


@callback(
    Output("pie-chart", "figure"),
    Input("method-filter", "value"),
    Input("year-slider", "value"),
)
def update_pie(methods, year_range):
    df = get_method_pie(methods)
    fig = px.pie(df, names="discoverymethod", values="count",
                 color_discrete_sequence=METHOD_COLORS, hole=0.45)
    fig.update_traces(textfont_color="#e2e8f0", marker_line_color="#020817",
                      marker_line_width=2)
    layout = {**PLOT_LAYOUT, "showlegend": True, "legend": dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10))}
    fig.update_layout(**layout)
    return fig


@callback(
    Output("scatter-chart", "figure"),
    Input("method-filter", "value"),
    Input("year-slider", "value"),
)
def update_scatter(methods, year_range):
    import numpy as np
    df = get_scatter(methods, year_range)

    df = df.dropna(subset=["pl_rade", "pl_bmasse"])
    df["symbol"] = df["in_habitable_zone"].map({True: "star", False: "circle"})
    fig = px.scatter(
        df, x="pl_bmasse", y="pl_rade",
        color="discoverymethod",
        color_discrete_sequence=METHOD_COLORS,
        symbol="in_habitable_zone",
        symbol_map={True: "star", False: "circle"},
        hover_name="pl_name",
        hover_data={"hostname": True, "pl_eqt": ":.0f", "disc_year": True,
                    "pl_bmasse": ":.2f", "pl_rade": ":.2f",
                    "in_habitable_zone": False, "discoverymethod": True},
        log_x=True, log_y=True,
        labels={"pl_bmasse": "Mass (Earth masses)", "pl_rade": "Radius (Earth radii)",
                "discoverymethod": "Method", "disc_year": "Discovery Year",
                "hostname": "Host Star", "pl_eqt": "Equil. Temp (K)"},
    )
    # Add planet class annotation bands
    for name, y0, y1, color in [
        ("Sub-Earths",     0.3,  0.8, "rgba(56,189,248,0.05)"),
        ("Super-Earths",   0.8,  1.6, "rgba(52,211,153,0.07)"),
        ("Mini-Neptunes",  1.6,  4.0, "rgba(251,191,36,0.05)"),
        ("Giant Planets",  4.0, 25.0, "rgba(244,114,182,0.05)"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=name,
                      annotation_position="right",
                      annotation_font=dict(size=9, color="#94a3b8"))
    fig.update_layout(**PLOT_LAYOUT,
                      xaxis=dict(title="Mass (Earth masses)", range=[-1, np.log10(5000)], **GRID_STYLE),
                      yaxis=dict(title="Radius (Earth radii)", range=[np.log10(0.3), np.log10(30)], **GRID_STYLE))
    return fig


@callback(
    Output("orbital-chart", "figure"),
    Input("method-filter", "value"),
    Input("year-slider", "value"),
)
def update_orbital(methods, year_range):
    df = get_orbital(methods, year_range)
    df["pl_rade"] = df["pl_rade"].fillna(1)
    fig = px.scatter(
        df, x="pl_orbsmax", y="pl_orbper",
        color="discoverymethod",
        color_discrete_sequence=METHOD_COLORS,
        size="pl_rade", size_max=16,
        hover_name="pl_name",
        log_x=True, log_y=True,
        labels={"pl_orbsmax": "Semi-major Axis (AU)",
                "pl_orbper": "Orbital Period (days)",
                "discoverymethod": "Method"},
    )
    fig.update_layout(**PLOT_LAYOUT,
                      xaxis=dict(title="Semi-major Axis (AU)", **GRID_STYLE),
                      yaxis=dict(title="Orbital Period (days)", **GRID_STYLE))
    return fig



# ---------------------------------------------------------------------------
# Shared helper — draws one system orbital diagram
# ---------------------------------------------------------------------------
import math as _math

def draw_system_fig(hostname, height_px=300, show_colorbar=True, show_labels=True):
    df = get_system(hostname)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**{**PLOT_LAYOUT, "margin": dict(l=5, r=5, t=5, b=5)})
        return fig

    fig = go.Figure()
    theta = [i * 2 * _math.pi / 360 for i in range(361)]

    for _, row in df.iterrows():
        a = row["pl_orbsmax"]
        fig.add_trace(go.Scatter(
            x=[a * _math.cos(t) for t in theta],
            y=[a * _math.sin(t) for t in theta],
            mode="lines",
            line=dict(color="rgba(100,180,255,0.45)", width=1.5),
            showlegend=False,
            name="Orbital path",
            hovertemplate="Orbital path<extra></extra>",
        ))

    rade   = df["pl_rade"].fillna(3)
    max_r  = rade.max() if rade.max() > 0 else 5
    sizes  = (rade / max_r * (20 if show_labels else 14) + 5).tolist()
    colors = df["pl_eqt"].fillna(300).tolist()

    fig.add_trace(go.Scatter(
        x=df["pl_orbsmax"].tolist(),
        y=[0] * len(df),
        mode="markers+text" if show_labels else "markers",
        customdata=[[n] for n in df["pl_name"].tolist()],
        marker=dict(
            size=sizes,
            color=colors,
            colorscale="RdYlBu_r",
            showscale=show_colorbar,
            cmin=200, cmax=2000,
            colorbar=dict(
                thickness=14, len=0.75,
                x=1.01, xanchor="left",
                tickfont=dict(color="#e2e8f0", size=12),
                title=dict(text="Surface Temp (°K)",
                           side="right",
                           font=dict(color="#e2e8f0", size=12)),
                bgcolor="rgba(15,23,42,0.8)",
                bordercolor="rgba(255,255,255,0.15)", borderwidth=1,
            ) if show_colorbar else None,
            line=dict(color="rgba(255,255,255,0.6)", width=1),
        ),
        name="Planets",
        text=df["pl_name"].tolist() if show_labels else None,
        textposition="top center",
        textfont=dict(size=8, color="#94a3b8"),
        hovertemplate="<b>%{customdata[0]}</b><br>Orbit: %{x:.4f} AU<extra></extra>",
    ))

    # Star
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=16 if show_labels else 10,
                    color="#fbbf24",
                    line=dict(color="#fef3c7", width=2)),
        showlegend=False,
        hovertemplate=f"<b>{hostname}</b><extra></extra>",
    ))

    max_a = df["pl_orbsmax"].max() * 1.35
    layout = {**PLOT_LAYOUT,
        "showlegend": False,
        "margin": dict(l=5, r=50 if show_colorbar else 5, t=5, b=5),
        "xaxis": dict(range=[-max_a, max_a], showgrid=False, zeroline=False, visible=False),
        "yaxis": dict(range=[-max_a * 0.55, max_a * 0.55],
                      showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
    }
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Hall of Fame callbacks — one per featured system
# ---------------------------------------------------------------------------
for _name in HALL_OF_FAME:
    def _make_hof_cb(n):
        @callback(
            Output(f"hof-{n.replace(' ','-')}", "figure"),
            Input("star-dropdown", "value"),   # dummy input to trigger on load
        )
        def _hof(_dummy, _n=n):
            return draw_system_fig(_n, height_px=180, show_colorbar=False, show_labels=False)
    _make_hof_cb(_name)


# ---------------------------------------------------------------------------
# Hall of Fame distance label callbacks
# ---------------------------------------------------------------------------
for _name in HALL_OF_FAME:
    def _make_dist_cb(n):
        @callback(
            Output(f"hof-dist-{n.replace(' ','-')}", "children"),
            Input("star-dropdown", "value"),  # dummy input to trigger on load
        )
        def _dist(_dummy, _n=n):
            return get_distance_ly(_n)
    _make_dist_cb(_name)


# ---------------------------------------------------------------------------
# Explorer dropdown callback
# ---------------------------------------------------------------------------
@callback(
    Output("system-chart", "figure"),
    Input("star-dropdown", "value"),
)
def update_system(hostname):
    if not hostname:
        return go.Figure()
    return draw_system_fig(hostname, height_px=300, show_colorbar=True, show_labels=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🌌  Exoplanet Explorer starting …")
    print("    Open http://127.0.0.1:8050 in your browser.\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))
