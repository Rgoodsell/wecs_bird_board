from __future__ import annotations

import html
import math
import re
import shutil
from typing import Callable

import altair as alt
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import HeatMap, MarkerCluster
from plotly.subplots import make_subplots
from streamlit_folium import st_folium
from streamlit_gsheets import GSheetsConnection

from species_info import (
    IUCN_COLORS,
    IUCN_CRITERIA,
    IUCN_DESCRIPTIONS,
    IUCN_LABELS,
    IUCN_RANK,
    SPECIES_TRAITS,
    UK_CONSERVATION_STATUS,
    UK_SPECIES,
    describe_iucn_criteria,
    get_iucn_criteria,
    get_iucn_status,
    get_species_info,
)

# UWE Bristol, Frenchay Campus (Coldharbour Lane, Bristol BS16 1QY) — the
# "Furthest Bird" award's origin point.
CAMPUS_COORDS = (51.5008, -2.5501)


def require_passcode() -> None:
    """Gate the whole app behind one shared class passcode (see
    .streamlit/secrets.toml.example) — halts the script here for anyone who
    hasn't entered it, so nothing below (data, live API calls) ever runs for
    them. A shared code, not per-student login: low friction for the class,
    but treat it as a low-stakes gate, not real access control — it will
    eventually get shared past the intended group.
    """
    if st.session_state.get("authenticated"):
        return
    st.title("🐦 WECS Bird Board 🐦")
    entered = st.text_input("Class passcode", type="password")
    if entered:
        if entered == st.secrets.get("APP_PASSCODE"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong passcode — try again.")
    st.stop()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def zoom_for_bounds(
    lat_min: float, lat_max: float, lng_min: float, lng_max: float,
    map_width_px: int, map_height_px: int, max_zoom: int = 16,
) -> int:
    """The Leaflet/Google Maps zoom level that fits a lat/lng bounding box
    inside a map of the given pixel size, computed directly rather than via
    Leaflet's own fitBounds(). fitBounds() needs to measure the map's actual
    rendered size, which it can't reliably do the instant a freshly-mounted
    Streamlit iframe embed loads — get that timing wrong and it silently
    zooms in on the bounding box's centre instead of framing it. Callers
    should pass a conservative (rather than exact) pixel size: an
    underestimate only zooms out a bit further than strictly necessary,
    while an overestimate can crop points back out of view.
    """
    world_dim = 256

    def lat_rad(lat: float) -> float:
        s = math.sin(math.radians(lat))
        return max(min(math.log((1 + s) / (1 - s)) / 2, math.pi), -math.pi) / 2

    def zoom_for_fraction(px: int, fraction: float) -> float:
        return math.log(px / world_dim / fraction, 2) if fraction > 0 else max_zoom

    lat_fraction = (lat_rad(lat_max) - lat_rad(lat_min)) / math.pi
    lng_span = lng_max - lng_min
    lng_fraction = (lng_span if lng_span >= 0 else lng_span + 360) / 360

    lat_zoom = zoom_for_fraction(map_height_px, lat_fraction)
    lng_zoom = zoom_for_fraction(map_width_px, lng_fraction)
    return max(0, min(math.floor(min(lat_zoom, lng_zoom)), max_zoom))

# BTO British List species count (Categories A, B and C — every species with
# an established, naturally-occurring or naturalised UK population).
UK_TOTAL_SPECIES = 636

# Paul Tol "Bright" qualitative palette
TOL_BRIGHT = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

# Award accents. These sit behind white text, so the lighter Tol swatches are
# darkened here to clear the 4.5:1 contrast threshold; every colour below is
# between 4.7:1 and 6.1:1 against white.
AWARD_ACCENTS = {
    "furthest": "#4477AA",
    "biggest": "#C43B50",
    "smallest": "#0F7B8A",
    "rarest": "#AA3377",
    "beakiest": "#228833",
}

CHART_AXIS_CONFIG = dict(
    labelFontSize=16,
    titleFontSize=19,
    labelColor="#000000",
    titleColor="#000000",
    labelFontWeight=500,
    titleFontWeight=600,
)

CHART_LEGEND_CONFIG = dict(
    labelFontSize=15,
    titleFontSize=16,
    labelColor="#000000",
    titleColor="#000000",
    labelFontWeight=500,
    titleFontWeight=600,
)

st.set_page_config(
    page_title="WECS Bird Board",
    page_icon="🐦",
    layout="wide",
)

require_passcode()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    h1, h2, h3, [data-testid="stCaptionContainer"] {
        text-align: center;
    }
    div[class*="st-key-metric-card-"], [data-testid="stMarkdownContainer"] {
        text-align: center;
    }
    div[class*="st-key-metric-card-"] [data-testid="stMetric"] {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    div[class*="st-key-metric-card-"] [data-testid="stMetricLabel"],
    div[class*="st-key-metric-card-"] [data-testid="stMetricValue"],
    div[class*="st-key-metric-card-"] [data-testid="stMetricDelta"] {
        justify-content: center;
    }
    div[class*="st-key-metric-card-"] [data-testid="stMetricLabel"] {
        /* The label's own grid container left-aligns its child by default —
           centering the container isn't enough once the text inside it is
           narrower than the container itself. */
        justify-items: center;
    }
    div[class*="st-key-metric-card-"] [data-testid="stMetricLabel"] p {
        font-weight: 800;
        text-align: center;
        font-size: 1.05rem;
        width: 100%;
    }
    div[class*="st-key-metric-card-"] {
        border-top: 4px solid transparent;
    }
    div[class*="st-key-metric-card-species"] { border-top-color: #4477AA; }
    div[class*="st-key-metric-card-observations"] { border-top-color: #EE6677; }
    div[class*="st-key-metric-card-observers"] { border-top-color: #228833; }
    div[class*="st-key-metric-card-new-species"] { border-top-color: #CCBB44; }
    div[class*="st-key-metric-card-locations"] { border-top-color: #66CCEE; }
    div[class*="st-key-metric-card-busiest-spot"] { border-top-color: #AA3377; }
    div[class*="st-key-metric-card-this-week"] { border-top-color: #228833; }
    div[class*="st-key-species-card-"] {
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        padding: 0.5rem;
        gap: 0.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[class*="st-key-species-card-"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-species-card-rank1-"],
    div[class*="st-key-species-card-rank2-"],
    div[class*="st-key-species-card-rank3-"] {
        padding: 0.85rem;
        position: relative;
        overflow: visible;
    }
    div[class*="st-key-species-card-rank1-"] {
        border: 4px solid #D4AF37;
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.2);
    }
    div[class*="st-key-species-card-rank2-"] {
        border: 4px solid #A8A9AD;
        box-shadow: 0 0 0 3px rgba(168, 169, 173, 0.2);
    }
    div[class*="st-key-species-card-rank3-"] {
        border: 4px solid #CD7F32;
        box-shadow: 0 0 0 3px rgba(205, 127, 50, 0.2);
    }
    @keyframes wobble {
        0%, 100% { transform: translateY(-4px) rotate(0deg); }
        20% { transform: translateY(-4px) rotate(-4deg); }
        40% { transform: translateY(-4px) rotate(4deg); }
        60% { transform: translateY(-4px) rotate(-3deg); }
        80% { transform: translateY(-4px) rotate(3deg); }
    }
    div[class*="st-key-species-card-rank1-"]:hover {
        animation: wobble 0.6s ease-in-out;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-species-podium"] {
        gap: 0.5rem;
    }
    /* Drop each card by the height its plinth gives away to the tallest one,
       so the three plinths line up on a common floor. */
    div[class*="st-key-species-card-rank2-"] { margin-top: 28px; }
    div[class*="st-key-species-card-rank3-"] { margin-top: 52px; }
    div[class*="st-key-species-podium"] [data-testid="stColumn"] > div {
        gap: 0;
    }
    div[class*="st-key-plinth-rank"] {
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px 10px 0 0;
        box-shadow: inset 0 -6px 0 rgba(0, 0, 0, 0.12);
    }
    div[class*="st-key-plinth-rank"] p {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
    }
    div[class*="st-key-plinth-rank1"] { min-height: 96px; background-color: #D4AF37; }
    div[class*="st-key-plinth-rank2"] { min-height: 68px; background-color: #A8A9AD; }
    div[class*="st-key-plinth-rank3"] { min-height: 44px; background-color: #CD7F32; }
    div[class*="st-key-observer-podium"] {
        gap: 0.3rem;
    }
    div[class*="st-key-observer-podium"] [data-testid="stColumn"] > div {
        gap: 0;
    }
    div[class*="st-key-observer-podium-rank"] {
        min-height: 64px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        padding: 0.2rem;
        gap: 0.1rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[class*="st-key-observer-podium-rank"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-observer-podium-rank1-"] {
        border: 3px solid #D4AF37;
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.2);
    }
    div[class*="st-key-observer-podium-rank2-"] {
        border: 3px solid #A8A9AD;
        box-shadow: 0 0 0 3px rgba(168, 169, 173, 0.2);
        margin-top: 18px;
    }
    div[class*="st-key-observer-podium-rank3-"] {
        border: 3px solid #CD7F32;
        box-shadow: 0 0 0 3px rgba(205, 127, 50, 0.2);
        margin-top: 34px;
    }
    div[class*="st-key-observer-podium-rank1-"]:hover {
        animation: wobble 0.6s ease-in-out;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-observer-plinth-rank"] {
        box-sizing: border-box;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px 8px 0 0;
        box-shadow: inset 0 -4px 0 rgba(0, 0, 0, 0.12);
    }
    div[class*="st-key-observer-plinth-rank"] p {
        font-size: 1.3rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
    }
    div[class*="st-key-observer-plinth-rank1"] { min-height: 74px; background-color: #D4AF37; }
    div[class*="st-key-observer-plinth-rank2"] { min-height: 56px; background-color: #A8A9AD; }
    div[class*="st-key-observer-plinth-rank3"] { min-height: 40px; background-color: #CD7F32; }
    div[class*="st-key-species-cards-grid"] {
        gap: 0.5rem;
    }
    div[class*="st-key-species-card-"] [data-testid="stElementContainer"]:has(img),
    div[class*="st-key-photo-"] [data-testid="stElementContainer"]:has(img) {
        height: 130px !important;
    }
    div[class*="st-key-photo-"] {
        min-height: 280px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-radius: 12px;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        padding: 0.4rem;
        gap: 0.25rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[class*="st-key-photo-"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-photo-award-furthest"] { border-top-color: #4477AA; border-bottom-color: #4477AA; }
    div[class*="st-key-photo-award-biggest"] { border-top-color: #C43B50; border-bottom-color: #C43B50; }
    div[class*="st-key-photo-award-smallest"] { border-top-color: #0F7B8A; border-bottom-color: #0F7B8A; }
    div[class*="st-key-photo-award-rarest"] { border-top-color: #AA3377; border-bottom-color: #AA3377; }
    div[class*="st-key-photo-award-beakiest"] { border-top-color: #228833; border-bottom-color: #228833; }
    div[class*="st-key-extra-"] {
        min-height: 280px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-radius: 12px;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        padding: 0.6rem 0.8rem;
        gap: 0.25rem;
        text-align: center;
        transition: box-shadow 0.15s ease;
    }
    /* No translateY lift here (unlike the other hover-card variants above):
       a CSS transform on this element would make it the containing block for
       its descendants' `position: fixed` — which is exactly how Streamlit's
       chart fullscreen overlay is positioned, so expanding one of the award
       comparison charts would clip/scale it to this card's own small box
       instead of the viewport. */
    div[class*="st-key-extra-"]:hover {
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-extra-award-furthest"] { border-top-color: #4477AA; border-bottom-color: #4477AA; }
    div[class*="st-key-extra-award-biggest"] { border-top-color: #C43B50; border-bottom-color: #C43B50; }
    div[class*="st-key-extra-award-smallest"] { border-top-color: #0F7B8A; border-bottom-color: #0F7B8A; }
    div[class*="st-key-extra-award-rarest"] { border-top-color: #AA3377; border-bottom-color: #AA3377; }
    div[class*="st-key-extra-award-beakiest"] { border-top-color: #228833; border-bottom-color: #228833; }
    div[class*="st-key-extra-"] [data-testid="stElementContainer"]:has(iframe) {
        height: 190px !important;
    }
    div[class*="st-key-segment-"] {
        min-height: 280px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border-radius: 12px;
        border-top: 10px solid transparent;
        border-bottom: 10px solid transparent;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        padding: 1rem 1.25rem;
        gap: 0.6rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[class*="st-key-segment-"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    div[class*="st-key-segment-award-furthest"] { border-top-color: #4477AA; border-bottom-color: #4477AA; }
    div[class*="st-key-segment-award-biggest"] { border-top-color: #C43B50; border-bottom-color: #C43B50; }
    div[class*="st-key-segment-award-smallest"] { border-top-color: #0F7B8A; border-bottom-color: #0F7B8A; }
    div[class*="st-key-segment-award-rarest"] { border-top-color: #AA3377; border-bottom-color: #AA3377; }
    div[class*="st-key-segment-award-beakiest"] { border-top-color: #228833; border-bottom-color: #228833; }
    div[class*="st-key-map-frame"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }
    div[class*="st-key-map-frame"] iframe {
        display: block;
    }
    div[class*="st-key-chart-narrow-"] {
        max-width: 85%;
        margin: 0 auto;
    }
    /* st.plotly_chart resizes its figure to fill whatever box it's given,
       even with use_container_width=False — so the waffle's cell size is
       controlled by pinning that box to a fixed, square pixel size here,
       rather than by the figure's own (ignored) width/height. */
    div[class*="st-key-chart-narrow-waffle"] {
        max-width: 440px;
        margin: 0 auto;
    }
    div[class*="st-key-chart-narrow-waffle"] [data-testid="stElementContainer"] {
        width: 440px;
        height: 440px;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        justify-content: center;
    }
    button[data-testid="stTab"] {
        height: auto;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        background-color: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.15);
    }
    button[data-testid="stTab"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: #3D5A73;
        border-color: #3D5A73;
    }
    button[data-testid="stTab"][aria-selected="true"] p {
        color: white;
    }
    div[data-baseweb="tab-list"] > div {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_observations() -> pd.DataFrame:
    """The single source of truth for every tab in the app. Reads straight
    from the Google Sheet (the same one the upload tab writes to), not the
    local data/wecs_birds.csv — that file is only ever a one-time seed copy
    now. Reading it here instead would mean the rest of the app never sees
    what's actually been uploaded, and — on Streamlit Cloud specifically —
    would keep showing whatever's baked into the git repo forever, since
    the local filesystem there doesn't persist across reboots/redeploys.
    ttl=0 disables the connection's own internal cache, so the only caching
    in play is this function's own — cleared explicitly (see
    load_observations.clear()) right after a successful upload, which is
    what makes a fresh upload actually show up elsewhere without it.
    """
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(ttl=0)
    df = df.rename(
        columns={
            "Common Name": "common_name",
            "Scientific Name": "scientific_name",
            "Count": "count",
            "Location": "location",
            "Latitude": "lat",
            "Longitude": "lng",
            "Date": "date",
            "Name": "observer",
        }
    )
    # Combined exports from different observers don't agree on date format
    # (ISO "2026-03-06" alongside "06/03/2026"), so each value is parsed
    # individually rather than against one fixed format.
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    # eBird records presence-only checklists as "X" rather than a number.
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(1).astype(int)
    return df[
        ["common_name", "scientific_name", "count", "date", "location", "lat", "lng", "observer"]
    ]


# Columns load_observations actually reads — an upload missing any of these
# would silently break (or crash) the rest of the app, so they're required.
# Note: "Name" is deliberately not required here — eBird's own "My eBird
# Data" personal export doesn't include an observer-name column at all (it's
# a personal export, so eBird doesn't need to tell you your own name), so it
# never appears on a real upload. It has to come from the username field
# instead — see the upload tab, which requires one when the file has no
# "Name" column of its own.
REQUIRED_UPLOAD_COLUMNS = [
    "Common Name", "Scientific Name", "Count", "Location",
    "Latitude", "Longitude", "Date",
]
# Generous headroom for a real eBird export (weeks/months of one person's
# checklists), while still bounding a mistaken or abusive giant upload.
MAX_UPLOAD_ROWS = 2000

# A "Location" is treated as a home address if it starts with a house
# number, and either (a) has a comma-separated region after it — e.g.
# "28 Ferndale Drive, Scotland" — the pattern the original class dataset's
# fake addresses happened to follow — or (b) simply ends in a common
# English street-type word — e.g. "42 Test Lane", which has no trailing
# region at all. This is a best-effort heuristic, not a real address
# detector: a house-numbered location with neither a region suffix nor a
# recognisable street word (a Welsh street name on its own, say) can still
# slip through. There's no fully reliable way to spot an arbitrary address
# in free text without a geocoding lookup.
_UPLOAD_ADDRESS_LEADING_NUMBER = re.compile(r"^\d+\s+(.+)$")
_UPLOAD_STREET_SUFFIX_WORDS = {
    "street", "road", "drive", "lane", "avenue", "close", "way", "court",
    "place", "crescent", "terrace", "walk", "row", "square", "mews",
    "gardens", "grove", "boulevard", "gate", "hill", "farm",
}


def _looks_like_address(location: str) -> bool:
    match = _UPLOAD_ADDRESS_LEADING_NUMBER.match(location.strip())
    if not match:
        return False
    rest = match.group(1)
    if "," in rest:
        return True
    words = rest.split()
    last_word = re.sub(r"\W", "", words[-1]).lower() if words else ""
    return last_word in _UPLOAD_STREET_SUFFIX_WORDS


def validate_upload_schema(df: pd.DataFrame) -> list[str]:
    """Fatal, whole-file problems — wrong columns, empty, or way too big —
    checked before anything else runs. A non-empty result means the file is
    unusable as a whole, not just missing a few rows."""
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}"]

    errors = []
    if len(df) == 0:
        errors.append("The file has no rows.")
    if len(df) > MAX_UPLOAD_ROWS:
        errors.append(f"That's {len(df)} rows — please upload at most {MAX_UPLOAD_ROWS} at a time.")
    return errors


def clean_upload_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drops rows that can't be used — missing/invalid coordinates, or an
    unparseable Date — rather than rejecting the whole file for them. Real
    eBird exports routinely mix a few incomplete rows (an incidental
    sighting with no precise location, say) in with otherwise-good data, so
    one bad row shouldn't block everything else. Returns the filtered
    DataFrame and a list of human-readable notes about what got dropped and
    why (empty if nothing was)."""
    df = df.copy()
    notes = []

    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lng = pd.to_numeric(df["Longitude"], errors="coerce")
    valid_coords = lat.between(-90, 90) & lng.between(-180, 180)
    if not valid_coords.all():
        notes.append(
            f"{int((~valid_coords).sum())} row(s) skipped — missing or invalid "
            "Latitude/Longitude."
        )
        df = df[valid_coords]

    dates = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    if dates.isna().any():
        notes.append(f"{int(dates.isna().sum())} row(s) skipped — unparseable Date.")
        df = df[dates.notna()]

    return df, notes


def scrub_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors scripts/scrub_locations.py: a "Location" that looks like a
    home address (a house number followed by a street name) gets collapsed
    to the row's County instead — without a County to fall back on, the
    address-like value is dropped entirely rather than shown as-is. Applied
    to the whole combined dataset after a merge (see clean_dataset), not
    just newly-uploaded rows, so a previously un-scrubbed row — or a later
    improvement to the address heuristic — gets caught too, not only
    whatever happened to come in on this particular upload."""
    df = df.copy()

    def scrub(row):
        location = row.get("Location")
        if not isinstance(location, str) or not _looks_like_address(location):
            return location
        county = row.get("County")
        return county.strip() if isinstance(county, str) and county.strip() else None

    df["Location"] = df.apply(scrub, axis=1)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """The full data-hygiene pass run over the combined dataset after new
    rows are merged in — currently just location scrubbing, but the natural
    place to add further whole-file cleaning steps later."""
    return scrub_locations(df)


def trend_label(current: int, previous: int, comparison: str):
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff} {comparison}"


def build_line_figure(x_vals, y_vals, color: str, x_title: str, y_title: str, x_range, y_max) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=7, color=color),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(t=20, b=10, l=10, r=10),
        font=dict(family="Space Grotesk, sans-serif", size=15, color="#000000"),
        xaxis=dict(title=x_title, range=x_range, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title=y_title, range=[0, y_max], gridcolor="rgba(0,0,0,0.08)"),
    )
    return fig


def build_waffle_figure(
    filled_cells: int, total_cells: int, color: str, empty_color: str = "#E1E1E1",
    cols: int = 10, cell_px: int = 34,
) -> go.Figure:
    """A grid of squares, one per unit of total_cells — filled_cells of them
    in colour, the rest in a neutral grey, filled row by row from the bottom
    up. total_cells doesn't need to fill the grid exactly; the last row is
    simply left short. A fixed pixel size (rather than use_container_width)
    keeps the squares actually square and a consistent size regardless of
    the column width they're dropped into."""
    rows = -(-total_cells // cols)  # ceil
    xs = [i % cols for i in range(total_cells)]
    ys = [i // cols for i in range(total_cells)]
    colors = [color if i < filled_cells else empty_color for i in range(total_cells)]
    margin = 10
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(symbol="square", size=cell_px - 6, color=colors, line=dict(width=1, color="#FFFFFF")),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        width=cols * cell_px + margin * 2,
        height=rows * cell_px + margin * 2,
        autosize=False,
        margin=dict(t=margin, b=margin, l=margin, r=margin),
        xaxis=dict(visible=False, range=[-0.6, cols - 0.4], fixedrange=True),
        yaxis=dict(
            visible=False, range=[-0.6, rows - 0.4], scaleanchor="x", fixedrange=True
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_static_line_chart(
    placeholder, df: pd.DataFrame, x_col: str, y_col: str, color: str, x_title: str,
    y_title: str, key: str,
) -> None:
    x_range = [df[x_col].min().isoformat(), df[x_col].max().isoformat()]
    y_max = df[y_col].max() * 1.1
    fig = build_line_figure(df[x_col], df[y_col], color, x_title, y_title, x_range, y_max)
    placeholder.plotly_chart(fig, use_container_width=True, key=key)


MEDALS = {0: "🥇", 1: "🥈", 2: "🥉"}

# Silver-left, gold-raised-in-the-middle, bronze-right — shared by the
# species and observer podiums.
PODIUM_COLUMN = {1: 2, 2: 1, 3: 3}


def render_species_card(card: dict, key: str, medal: str = "") -> None:
    with st.container(border=True, key=key):
        name = (
            f"{medal} {card['common_name']} {medal}" if medal else card["common_name"]
        )
        st.markdown(f"**{name}**")
        if card["image_url"]:
            st.markdown(
                f'<img src="{card["image_url"]}" '
                'style="width:100%; height:130px; object-fit:cover; '
                'object-position:center top; border-radius:8px; '
                'display:block; margin:0 auto;">',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No image available")
        st.caption(f"*{card['scientific_name']}*")
        st.write(f"Observations: {card['observations']}")
        bg, fg = IUCN_COLORS.get(card["iucn_code"], ("#EEEEEE", "#666666"))
        status_text = (
            f"{card['iucn_label']} ({card['iucn_code']})"
            if card["iucn_label"]
            else "Unknown"
        )
        st.markdown(
            f'<span style="background-color:{bg}; color:{fg}; '
            "padding:0.2rem 0.6rem; border-radius:999px; "
            f'font-size:0.85rem; font-weight:600;">{status_text}</span>',
            unsafe_allow_html=True,
        )


def render_award_row(
    index: int,
    key: str,
    icon: str,
    title: str,
    accent: str,
    common_name: str,
    scientific_name: str,
    metric_value: str,
    metric_label: str,
    observer: str,
    date: pd.Timestamp,
    location: str,
    render_extra: Callable[[], None],
    value_colors: tuple[str, str] | None = None,
    extra_stat: str | None = None,
) -> None:
    info = get_species_info(common_name, scientific_name)

    def render_photo():
        with st.container(key=f"photo-{key}"):
            if info["image_url"]:
                st.markdown(
                    f'<img src="{info["image_url"]}" '
                    'style="width:100%; height:130px; object-fit:cover; '
                    'object-position:center top; border-radius:8px; '
                    'display:block; margin:0 auto;">',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No image available")
            st.markdown(
                f'<p style="font-size:1.3rem; font-weight:700; margin:0;">'
                f'{html.escape(common_name)}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="font-size:0.95rem; font-style:italic; '
                f'color:gray; margin:0;">{html.escape(scientific_name)}</p>',
                unsafe_allow_html=True,
            )

    def render_segment():
        with st.container(key=f"segment-{key}"):
            st.markdown(
                f'<p style="text-transform:uppercase; letter-spacing:0.06em; '
                f'font-size:1.4rem; font-weight:800; color:{accent}; margin:0;">'
                f'{title}</p>',
                unsafe_allow_html=True,
            )
            if value_colors:
                bg, fg = value_colors
                st.markdown(
                    f'<span style="display:inline-block; background-color:{bg}; '
                    f'color:{fg}; padding:0.3rem 1rem; border-radius:999px; '
                    f'font-size:1.5rem; font-weight:800; line-height:1.25;">'
                    f"{metric_value}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span style="font-size:2.2rem; font-weight:800; '
                    f'color:#1A1A1A; line-height:1.25;">{metric_value}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<p style="font-size:1.05rem; color:gray; margin:0;">'
                f'{metric_label}</p>',
                unsafe_allow_html=True,
            )
            if extra_stat:
                st.markdown(
                    f'<p style="font-size:1.05rem; font-weight:700; '
                    f'color:#1A1A1A; margin:0.4rem 0 0 0;">{extra_stat}</p>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span style="display:inline-block; margin-top:0.6rem; '
                f'background-color:{accent}; color:#FFFFFF; padding:0.3rem 0.75rem; '
                'border-radius:999px; font-size:1rem; font-weight:700; '
                f'line-height:1.25;">Spotted by {html.escape(observer)}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"{date:%d %b %Y} · {location}")

    def render_extra_panel():
        with st.container(key=f"extra-{key}"):
            render_extra()

    if index % 2 == 0:
        photo_col, segment_col, extra_col = st.columns(
            [1, 1.5, 1.7], gap="medium", vertical_alignment="center"
        )
    else:
        extra_col, segment_col, photo_col = st.columns(
            [1.7, 1.5, 1], gap="medium", vertical_alignment="center"
        )
    with photo_col:
        render_photo()
    with segment_col:
        render_segment()
    with extra_col:
        render_extra_panel()


st.title("🐦 WECS Bird Board 🐦")
st.caption("The bird is the word")

observations = load_observations()

total_species = observations["common_name"].nunique()
total_observations = len(observations)
total_observers = observations["observer"].nunique()

today = pd.Timestamp.now().normalize()
this_month = observations[observations["date"].dt.to_period("M") == today.to_period("M")]

# The Species/Observations/Observers cards show season-to-date totals, so
# their trend badge is this month's raw activity — how many observations
# were logged, and how many distinct species/observers appeared in them —
# rather than a month-over-month snapshot comparison. The old comparison
# netted to a misleading "+0" whenever this month's and last month's
# activity levels happened to match, even when none of it was the same
# species or observations.
species_trend = f"+{this_month['common_name'].nunique()} new this month"
observations_trend = f"+{len(this_month)} new this month"
observers_trend = f"+{this_month['observer'].nunique()} new this month"

# "New Species" specifically means species new to the season — the first
# time they've ever been logged — so it's tracked separately via each
# species' first-ever appearance date, not just "not seen last month" (which
# would wrongly flag a species that skipped a month and reappeared).
first_seen_species = observations.groupby("common_name")["date"].min()
new_species_count = int(
    (first_seen_species.dt.to_period("M") == today.to_period("M")).sum()
)
previous_new_species_count = int(
    (first_seen_species.dt.to_period("M") == (today.to_period("M") - 1)).sum()
)
new_species_trend = trend_label(
    new_species_count, previous_new_species_count, "vs last month"
)

top_species = (
    observations["common_name"]
    .value_counts()
    .head(10)
    .rename_axis("common_name")
    .reset_index(name="observations")
)
scientific_names = observations.drop_duplicates("common_name").set_index(
    "common_name"
)["scientific_name"]

overview_tab, map_tab, species_tab, leaderboard_tab, awards_tab, upload_tab = st.tabs(
    ["📊 Overview", "🗺️ Map", "🏆 Top Species", "🥇 Leaderboard", "🏅 Awards", "📤 Add Data"]
)

with overview_tab:
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True, key="metric-card-species"):
            st.metric("Species", total_species, delta=species_trend)
    with col2:
        with st.container(border=True, key="metric-card-observations"):
            st.metric("Observations", total_observations, delta=observations_trend)
    with col3:
        with st.container(border=True, key="metric-card-observers"):
            st.metric("Observers", total_observers, delta=observers_trend)
    with col4:
        with st.container(border=True, key="metric-card-new-species"):
            st.metric("New Species", new_species_count, delta=new_species_trend)

    st.divider()

    trend_col, richness_col = st.columns(2)

    with trend_col:
        st.subheader("Observations Over Time")
        weekly_counts = (
            observations.set_index("date")
            .resample("W")
            .size()
            .rename("observations")
            .reset_index()
        )
        with st.container(key="chart-narrow-trend"):
            trend_placeholder = st.empty()

    with richness_col:
        st.subheader("Cumulative Species Richness")
        first_seen = (
            observations.groupby("common_name")["date"]
            .min()
            .rename("date")
            .reset_index()
        )
        richness = (
            first_seen.groupby("date")
            .size()
            .cumsum()
            .rename("species_richness")
            .reset_index()
        )
        with st.container(key="chart-narrow-richness"):
            richness_placeholder = st.empty()

    render_static_line_chart(
        trend_placeholder, weekly_counts, "date", "observations", TOL_BRIGHT[0],
        "Week", "Observations", key="trend-chart-static",
    )
    render_static_line_chart(
        richness_placeholder, richness, "date", "species_richness", TOL_BRIGHT[2],
        "Date", "Cumulative Species", key="richness-chart-static",
    )

    st.divider()

    uk_species_pct = total_species / UK_TOTAL_SPECIES
    st.subheader("UK Species Coverage")
    with st.container(key="chart-narrow-waffle"):
        st.plotly_chart(
            build_waffle_figure(
                total_species, UK_TOTAL_SPECIES, TOL_BRIGHT[1], cols=26, cell_px=16
            ),
            use_container_width=True,
            key="uk-species-waffle",
        )
    st.caption(
        f"{total_species} of {UK_TOTAL_SPECIES} species on the BTO British List "
        f"(Categories A, B and C) observed so far — {uk_species_pct:.1%}. "
        "Each square represents 1 species."
    )

with map_tab:
    st.header("Observation Locations")

    location_counts = observations["location"].value_counts()
    busiest_location = location_counts.idxmax()
    busiest_count = int(location_counts.max())
    recent_count = len(
        observations[
            observations["date"] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        ]
    )

    map_col1, map_col2, map_col3 = st.columns(3)
    with map_col1:
        with st.container(border=True, key="metric-card-locations"):
            st.metric("Locations", observations["location"].nunique())
    with map_col2:
        with st.container(border=True, key="metric-card-busiest-spot"):
            st.metric("Busiest Spot", busiest_location, f"{busiest_count} sightings")
    with map_col3:
        with st.container(border=True, key="metric-card-this-week"):
            st.metric("This Week", recent_count)

    st.divider()

    map_lat_min, map_lat_max = observations["lat"].min(), observations["lat"].max()
    map_lng_min, map_lng_max = observations["lng"].min(), observations["lng"].max()
    map_center = [(map_lat_min + map_lat_max) / 2, (map_lng_min + map_lng_max) / 2]
    # A conservative (narrower-than-real) width estimate for the map's
    # actual rendered size — see zoom_for_bounds — since use_container_width
    # means the true pixel width isn't known here; height is exact, from the
    # st_folium(height=500) call below.
    map_zoom = zoom_for_bounds(
        map_lat_min, map_lat_max, map_lng_min, map_lng_max,
        map_width_px=320, map_height_px=500, max_zoom=16,
    )
    # Esri's World_Light_Gray_Base only has tiles up to zoom 16 — capping the
    # map here (not just the tile layer) stops MarkerCluster's zoom-to-bounds
    # click handler from zooming past that into "tile not available" grey
    # squares when a cluster's markers share (near-)identical coordinates,
    # e.g. multiple observers logging the same spot. It spiderfies instead.
    observation_map = folium.Map(
        location=map_center, zoom_start=map_zoom, tiles=None, max_zoom=16
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
            "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Base map",
        control=False,
        max_zoom=16,
    ).add_to(observation_map)

    markers_group = folium.FeatureGroup(name="📍 Markers", show=True)
    marker_cluster = MarkerCluster(disableClusteringAtZoom=16).add_to(markers_group)
    for _, row in observations.iterrows():
        # Folium/Leaflet popups render as HTML by default, same risk as the
        # unsafe_allow_html card text above — escape for the same reason.
        popup_text = f"{html.escape(row['common_name'])} — {html.escape(row['observer'])}"
        folium.Marker(location=[row["lat"], row["lng"]], popup=popup_text).add_to(
            marker_cluster
        )
    markers_group.add_to(observation_map)

    heatmap_group = folium.FeatureGroup(name="🔥 Heatmap", show=False)
    HeatMap(observations[["lat", "lng"]].values.tolist()).add_to(heatmap_group)
    heatmap_group.add_to(observation_map)

    folium.LayerControl(collapsed=False).add_to(observation_map)
    observation_map.get_root().html.add_child(
        folium.Element(
            """
            <style>
            .leaflet-control-layers {
                font-family: 'Space Grotesk', sans-serif;
                border: none !important;
                border-radius: 12px !important;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) !important;
                padding: 0.75rem 1rem !important;
            }
            .leaflet-control-layers-list label {
                display: flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 0.95rem;
                font-weight: 600;
                color: #333;
                padding: 0.2rem 0;
                cursor: pointer;
            }
            .leaflet-control-layers-selector {
                width: 15px;
                height: 15px;
                cursor: pointer;
            }
            </style>
            """
        )
    )

    # Constrain the map's width with columns rather than CSS: the folium iframe
    # sizes itself when it is created, and a later CSS resize leaves Leaflet
    # rendering tiles for the old dimensions.
    _left_pad, map_col, _right_pad = st.columns([1, 5, 1])
    with map_col:
        with st.container(key="map-frame"):
            st_folium(observation_map, use_container_width=True, height=500)

with species_tab:
    st.header("Top 8 Species")
    st.caption("Images and conservation status via Wikipedia and GBIF.")

    with st.spinner("Fetching species info..."):
        cards = [
            {
                "common_name": row.common_name,
                "scientific_name": scientific_names.get(row.common_name, "—"),
                "observations": row.observations,
                **get_species_info(
                    row.common_name, scientific_names.get(row.common_name, "")
                ),
            }
            for row in top_species.itertuples()
        ]

    podium, rest = cards[:3], cards[3:8]

    # A real podium: silver on the left, gold raised in the middle, bronze on
    # the right. Plinth heights and colours live in the CSS above, alongside the
    # matching card offsets that put all three plinths on a common floor.
    with st.container(key="species-podium"):
        podium_cols = st.columns(5)
        for rank, card in enumerate(podium, start=1):
            with podium_cols[PODIUM_COLUMN[rank]]:
                render_species_card(
                    card,
                    key=f"species-card-rank{rank}-{card['common_name']}",
                    medal=MEDALS[rank - 1],
                )
                # The plinth is the container itself rather than a fixed-height
                # div inside one: Streamlit measures element containers from
                # their text and would leave the block overflowing its own box.
                with st.container(key=f"plinth-rank{rank}-species"):
                    st.markdown(f"**{rank}**")

    if "balloons_shown" not in st.session_state:
        st.session_state.balloons_shown = False

    st.markdown(
        '<style>div[class*="st-key-balloon-trigger"] { display: none; }</style>',
        unsafe_allow_html=True,
    )
    if st.button("🎈", key="balloon-trigger") and not st.session_state.balloons_shown:
        st.session_state.balloons_shown = True
        st.balloons()

    balloons_shown_js = "true" if st.session_state.balloons_shown else "false"
    components.html(
        f"""
        <script>
        // Streamlit reruns the whole script on any interaction anywhere in
        // the app, and each rerun spawns a fresh iframe for this component.
        // Old iframes aren't always torn down immediately (still-pending
        // setInterval retries keep them alive), so a stale one can re-wire a
        // "not shown yet" handler onto a card after a newer iframe already
        // fired the balloons — the "shown" flag it captured at creation time
        // never sees the update. Tracking the flag on window.parent instead
        // of in each iframe's own closure gives every instance, old or new,
        // the same live answer, so a stale instance can't reintroduce a
        // hover trigger (or fire it twice) after the fact.
        const win = window.parent;
        if ({balloons_shown_js}) {{
            win.__wecsBalloonsShown = true;
        }}
        function wireBalloonHover() {{
            const doc = win.document;
            const cards = doc.querySelectorAll(
                'div[class*="st-key-species-card-rank"]'
            );
            if (!cards.length) return false;
            cards.forEach(function (card) {{
                if (card._balloonHandler) {{
                    card.removeEventListener("mouseenter", card._balloonHandler);
                    card._balloonHandler = null;
                }}
                if (!win.__wecsBalloonsShown) {{
                    card._balloonHandler = function () {{
                        if (win.__wecsBalloonsShown) return;
                        win.__wecsBalloonsShown = true;
                        const btn = doc.querySelector(
                            'div[class*="st-key-balloon-trigger"] button'
                        );
                        if (btn) btn.click();
                    }};
                    card.addEventListener("mouseenter", card._balloonHandler);
                }}
            }});
            return true;
        }}
        if (!wireBalloonHover()) {{
            const interval = setInterval(function () {{
                if (win.__wecsBalloonsShown || wireBalloonHover()) {{
                    clearInterval(interval);
                }}
            }}, 300);
        }}
        </script>
        """,
        height=0,
    )

    cols_per_row = 5
    with st.container(key="species-cards-grid"):
        for row_start in range(0, len(rest), cols_per_row):
            row_cards = rest[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, card in zip(cols, row_cards):
                with col:
                    render_species_card(
                        card, key=f"species-card-{card['common_name']}"
                    )

    st.divider()

    st.subheader("Top Species by Observation Count")
    species_encoding = dict(
        x=alt.X(
            "common_name",
            sort="-y",
            title="Species",
            axis=alt.Axis(labelAngle=-40, labelLimit=160, labelOverlap=False),
        ),
        y=alt.Y("observations", title="Observations"),
        color=alt.Color(
            "common_name", scale=alt.Scale(range=TOL_BRIGHT), legend=None
        ),
    )
    species_dots = alt.Chart(top_species).mark_circle(size=200).encode(**species_encoding)
    species_stems = alt.Chart(top_species).mark_bar(size=5).encode(**species_encoding)
    species_chart = (
        (species_dots + species_stems)
        .properties(height=450)
        .configure_axis(
            **{**CHART_AXIS_CONFIG, "labelFontSize": 18, "titleFontSize": 21}
        )
        .configure_legend(**CHART_LEGEND_CONFIG)
    )
    with st.container(key="chart-narrow-species"):
        st.altair_chart(species_chart, use_container_width=True)

@st.cache_data
def build_leaderboard_figure(observations: pd.DataFrame, observer_names: list[str]) -> go.Figure:
    """The animated race chart + resting cumulative-richness chart. Expensive
    (TWEEN_STEPS x ~one frame per week, each with a bar trace and one line
    trace per observer) and only actually needs to change when observations
    or the top-5 observer list changes — cached so it isn't rebuilt from
    scratch on every rerun, which Streamlit triggers on any interaction
    anywhere in the app, not just ones on this tab.
    """
    # Running weekly totals per observer. The last week's row equals each
    # observer's season total, so this drives both the resting charts and the
    # animated replay from one source.
    week_ends = observations.set_index("date").resample("W").size().index

    race = pd.DataFrame(
        [
            {
                "date": week_end,
                "observer": name,
                "species": seen["common_name"].nunique(),
                "observations": len(seen),
            }
            for week_end in week_ends
            for name in observer_names
            for seen in [
                observations[
                    (observations["date"] <= week_end)
                    & (observations["observer"] == name)
                ]
            ]
        ]
    )

    observer_colors = dict(zip(observer_names, TOL_BRIGHT))
    species_max = int(race["species"].max())

    weeks = list(week_ends)
    weekly = {week_end: group.set_index("observer") for week_end, group in race.groupby("date")}

    # Rank position per week, 1 = most species. Bars are drawn at these numeric
    # positions rather than on a category axis, which is what lets them slide
    # past each other instead of snapping to a new slot.
    weekly_ranks = {
        week_end: {
            name: position + 1
            for position, name in enumerate(
                sorted(
                    observer_names,
                    key=lambda n: (-weekly[week_end].loc[n, "species"], n),
                )
            )
        }
        for week_end in weeks
    }

    # gganimate/tweenr equivalent: rather than one frame per week, interpolate
    # extra frames in between and ease the fraction so motion accelerates out of
    # one week and settles into the next.
    TWEEN_STEPS = 5

    def smoothstep(fraction: float) -> float:
        return fraction * fraction * (3 - 2 * fraction)

    def race_traces(week_index: int, progress: float):
        """One tweened frame: bar lengths, rank positions and the leading edge
        of each line all interpolated between two weekly snapshots."""
        week_a = weeks[week_index]
        week_b = weeks[min(week_index + 1, len(weeks) - 1)]

        def blend(start, end):
            return start + (end - start) * progress

        values = [
            blend(weekly[week_a].loc[n, "species"], weekly[week_b].loc[n, "species"])
            for n in observer_names
        ]
        positions = [
            blend(weekly_ranks[week_a][n], weekly_ranks[week_b][n])
            for n in observer_names
        ]
        bars = go.Bar(
            x=values,
            y=positions,
            orientation="h",
            width=0.62,
            marker_color=[observer_colors[n] for n in observer_names],
            text=[
                f"{n}  {round(v)}" for n, v in zip(observer_names, values)
            ],
            textposition="outside",
            textfont=dict(size=14, color="#000000"),
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
        )

        history_weeks = weeks[: week_index + 1]
        tip_date = week_a + (week_b - week_a) * progress
        lines = [
            go.Scatter(
                x=history_weeks + [tip_date],
                y=[weekly[w].loc[name, "species"] for w in history_weeks]
                + [
                    blend(
                        weekly[week_a].loc[name, "species"],
                        weekly[week_b].loc[name, "species"],
                    )
                ],
                mode="lines+markers",
                name=name,
                line=dict(color=observer_colors[name], width=2.5),
                marker=dict(size=5),
                hovertemplate=f"{name}<br>%{{y}} species<extra></extra>",
            )
            for name in observer_names
        ]
        return [bars] + lines

    first_traces = race_traces(0, 0.0)
    final_traces = race_traces(len(weeks) - 1, 0.0)

    leaderboard_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top Observers", "Cumulative Species Richness"),
        horizontal_spacing=0.12,
    )
    for trace in first_traces[:1]:
        leaderboard_fig.add_trace(trace, row=1, col=1)
    for trace in first_traces[1:]:
        leaderboard_fig.add_trace(trace, row=1, col=2)

    # TWEEN_STEPS frames per week, plus a final resting frame. Every frame keeps
    # the same trace shapes so Plotly can interpolate between them too.
    trace_indices = list(range(len(observer_names) + 1))
    leaderboard_fig.frames = [
        go.Frame(
            name=f"{week_index}-{step}",
            data=race_traces(week_index, smoothstep(step / TWEEN_STEPS)),
            traces=trace_indices,
        )
        for week_index in range(len(weeks) - 1)
        for step in range(TWEEN_STEPS)
    ] + [
        go.Frame(
            name=f"{len(weeks) - 1}-0",
            data=final_traces,
            traces=trace_indices,
        )
    ]

    # Transport controls sit under the plots, beside the scrubber, so they stay
    # clear of the subplot titles.
    play_button = dict(
        type="buttons",
        direction="left",
        showactive=False,
        x=0,
        xanchor="left",
        y=-0.30,
        yanchor="top",
        pad=dict(t=0, r=10),
        buttons=[
            dict(
                label="▶ Play the season",
                method="animate",
                args=[
                    None,
                    dict(
                        # The easing already lives in the interpolated frames, so
                        # Plotly just needs to hand off between them linearly.
                        frame=dict(duration=55, redraw=True),
                        transition=dict(duration=55, easing="linear"),
                        fromcurrent=True,
                        mode="immediate",
                    ),
                ],
            ),
            dict(
                label="❚❚ Pause",
                method="animate",
                args=[
                    [None],
                    dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                    ),
                ],
            ),
        ],
    )

    # One scrub stop per week, not per tween frame, so dragging stays usable.
    week_slider = dict(
        active=len(weeks) - 1,
        x=0.33,
        len=0.67,
        y=-0.26,
        yanchor="top",
        currentvalue=dict(prefix="Week ending ", font=dict(size=15, color="#000000")),
        font=dict(size=13, color="#000000"),
        steps=[
            dict(
                label=f"{week_end:%d %b}",
                method="animate",
                args=[
                    [f"{week_index}-0"],
                    dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                    ),
                ],
            )
            for week_index, week_end in enumerate(weeks)
        ],
    )

    leaderboard_fig.update_layout(
        height=540,
        margin=dict(t=60, b=170, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", size=16, color="#000000"),
        updatemenus=[play_button],
        sliders=[week_slider],
        legend=dict(title_text="Observer", font=dict(size=15, color="#000000")),
        bargap=0.35,
    )
    # Rest on the finished season; the slider starts at the last week to match.
    for trace_index, trace in enumerate(final_traces):
        leaderboard_fig.data[trace_index].update(trace)

    axis_title_font = dict(size=19, color="#000000")
    axis_tick_font = dict(size=16, color="#000000")
    leaderboard_fig.update_xaxes(
        title_text="Species Seen",
        # Extra headroom so the name label sitting outside each bar has space.
        range=[0, species_max * 1.65],
        title_font=axis_title_font,
        tickfont=axis_tick_font,
        gridcolor="rgba(0,0,0,0.08)",
        row=1,
        col=1,
    )
    # Rank 1 at the top. Names ride on the bars themselves rather than the axis,
    # so a label travels with its bar as the order changes.
    leaderboard_fig.update_yaxes(
        range=[len(observer_names) + 0.6, 0.4],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )
    leaderboard_fig.update_xaxes(
        title_text="Week",
        # Pinned, or the time axis re-fits to each frame's partial data and the
        # lines appear to crawl on the spot instead of advancing.
        range=[week_ends.min().isoformat(), week_ends.max().isoformat()],
        title_font=axis_title_font,
        tickfont=axis_tick_font,
        gridcolor="rgba(0,0,0,0.08)",
        row=1,
        col=2,
    )
    leaderboard_fig.update_yaxes(
        title_text="Cumulative Species Richness",
        range=[0, species_max * 1.08],
        title_font=axis_title_font,
        tickfont=axis_tick_font,
        gridcolor="rgba(0,0,0,0.08)",
        row=1,
        col=2,
    )
    for annotation in leaderboard_fig.layout.annotations:
        annotation.font = dict(
            size=22, color="#000000", family="Space Grotesk, sans-serif"
        )
    return leaderboard_fig


with leaderboard_tab:
    st.header("Observer Leaderboard")

    # Leaderboard is capped to the top 5 observers by season species count —
    # everything downstream (race chart, podium) only ever sees this list.
    final_species_counts = observations.groupby("observer")["common_name"].nunique()
    observer_names = sorted(
        final_species_counts.index,
        key=lambda n: (-final_species_counts[n], n),
    )[:5]

    with st.container(key="observer-podium"):
        observer_podium_cols = st.columns(5)
        for rank, name in enumerate(observer_names[:3], start=1):
            with observer_podium_cols[PODIUM_COLUMN[rank]]:
                with st.container(border=True, key=f"observer-podium-rank{rank}-{name}"):
                    st.markdown(f"**{MEDALS[rank - 1]} {name}**")
                    st.caption(f"{final_species_counts[name]} species")
                with st.container(key=f"observer-plinth-rank{rank}"):
                    st.markdown(f"**{rank}**")

    leaderboard_fig = build_leaderboard_figure(observations, observer_names)
    st.plotly_chart(leaderboard_fig, use_container_width=True)

    st.divider()

    st.subheader("All Observers")
    all_observer_names = sorted(
        final_species_counts.index,
        key=lambda n: (-final_species_counts[n], n),
    )
    all_observers_df = pd.DataFrame(
        {
            "observer": all_observer_names,
            "species": [final_species_counts[n] for n in all_observer_names],
        }
    )
    all_observers_chart = (
        alt.Chart(all_observers_df)
        .mark_bar()
        .encode(
            x=alt.X("species:Q", title="Species Seen"),
            y=alt.Y("observer:N", sort="-x", title=None),
            color=alt.Color(
                "observer:N",
                scale=alt.Scale(domain=all_observer_names, range=TOL_BRIGHT),
                legend=None,
            ),
        )
        .properties(height=36 * len(all_observer_names) + 40)
        .configure_axis(**CHART_AXIS_CONFIG)
    )
    st.altair_chart(all_observers_chart, use_container_width=True)

    components.html(
        """
        <script>
        function wireLeaderboardAutoplay() {
            const doc = window.parent.document;
            const tabs = [...doc.querySelectorAll('button[data-testid="stTab"]')];
            const tab = tabs.find(function (t) {
                return t.textContent.includes("Leaderboard");
            });
            if (!tab) return false;
            if (tab._autoplayHandler) {
                tab.removeEventListener("click", tab._autoplayHandler);
            }
            tab._autoplayHandler = function () {
                let tries = 0;
                const waitForChart = setInterval(function () {
                    tries += 1;
                    const gd = doc.querySelector(".js-plotly-plot");
                    const Plotly = window.parent.Plotly;
                    if (gd && gd.data && Plotly) {
                        clearInterval(waitForChart);
                        // A .then() chain here doesn't reliably fire — the
                        // promise crosses from the parent window's Plotly
                        // into this iframe's realm, so a plain timeout
                        // sequences the two calls instead.
                        Plotly.animate(gd, ["0-0"], {
                            frame: {duration: 0, redraw: true},
                            mode: "immediate",
                        });
                        setTimeout(function () {
                            Plotly.animate(gd, null, {
                                frame: {duration: 55, redraw: true},
                                transition: {duration: 55, easing: "linear"},
                                fromcurrent: true,
                                mode: "immediate",
                            });
                        }, 60);
                    } else if (tries > 20) {
                        clearInterval(waitForChart);
                    }
                }, 150);
            };
            tab.addEventListener("click", tab._autoplayHandler);
            return true;
        }
        if (!wireLeaderboardAutoplay()) {
            const setupInterval = setInterval(function () {
                if (wireLeaderboardAutoplay()) clearInterval(setupInterval);
            }, 300);
        }
        </script>
        """,
        height=0,
    )

@st.cache_data
def compute_award_winners(observations: pd.DataFrame) -> dict:
    """Picks the Furthest/Biggest/Smallest/Rarest/Beakiest award winners and
    everything the cards need to display them. Cached for the same reason as
    build_leaderboard_figure: this reruns in full on any interaction anywhere
    in the app (not just on this tab), and none of it changes unless the
    observations data itself does — including the rarity loop, which calls
    the (separately, 24h-cached) IUCN API once per observed species.
    """
    obs_with_distance = observations.copy()
    obs_with_distance["distance_km"] = haversine_km(
        CAMPUS_COORDS[0],
        CAMPUS_COORDS[1],
        obs_with_distance["lat"],
        obs_with_distance["lng"],
    )
    furthest = obs_with_distance.sort_values(
        ["distance_km", "common_name"], ascending=[False, True]
    ).iloc[0]

    # The AVONET reference sample (data/AVONET_traits.csv), narrowed to the
    # BOU British List — this is where each observed species' trait values
    # come from, not a comparison backdrop in its own right (the comparison
    # charts only ever plot species actually observed this season — see
    # observed_traits). Keyed by scientific name, not common name — see
    # _load_species_traits(). Left unfiltered, this would be the full
    # 11,000+-species worldwide AVONET sample, which includes the occasional
    # value — like a kiwi's near-vestigial wing — that would blow the chart
    # scale out to nonsense if an observed species ever joined against it by
    # coincidence of scientific name — see _load_uk_species().
    all_traits = pd.DataFrame.from_dict(SPECIES_TRAITS, orient="index").rename_axis(
        "scientific_name"
    ).reset_index()
    all_traits = all_traits[all_traits["scientific_name"].isin(UK_SPECIES)]
    # "1 : X" display value — wing length as a multiple of beak length. Lower
    # means the beak takes up more of the wing.
    all_traits["wing_beak_ratio"] = (
        all_traits["wing_length_cm"] * 10 / all_traits["beak_mm"]
    )
    # Beak length as a % of wing length — the same relationship as the ratio
    # above (reused here rather than re-deriving it, since beak_mm divided
    # directly by wing_length_cm without converting units gave a meaningless
    # number), but inverted so it *increases* with beakiness, and scaled up
    # from a sub-0.1 fraction to a legible percentage. The comparison chart
    # plots this one instead: every other award's chart has its winner as
    # the tallest bar, and plotting the ratio above would instead put the
    # beakiest bird's bar at the bottom, reading as though it had lost.
    all_traits["beak_prominence"] = 100 / all_traits["wing_beak_ratio"]

    # Narrowed to species actually observed this season — this is what picks
    # the award winners, and what the comparison charts plot (rather than the
    # wider AVONET/BOU reference sample in all_traits, which is only used as
    # a lookup here). Common names in the export aren't consistent (the same
    # species can appear under more than one common-name spelling), so this
    # dedupes on the (common, scientific) pair first — to avoid losing rows
    # by deduping on common name alone — then again on scientific name alone
    # once merged, so a species logged under two common-name spellings
    # doesn't plot as two separate bars.
    observed_species = observations.drop_duplicates(["common_name", "scientific_name"])[
        ["common_name", "scientific_name"]
    ]
    observed_traits = observed_species.merge(all_traits, on="scientific_name").drop_duplicates(
        subset="scientific_name"
    )
    biggest = observed_traits.loc[observed_traits["wing_length_cm"].idxmax()]
    smallest = observed_traits.loc[observed_traits["wing_length_cm"].idxmin()]
    beakiest = observed_traits.loc[observed_traits["beak_prominence"].idxmax()]

    # "X cm/mm bigger/smaller than average" call-outs on the Biggest/
    # Smallest/Beakiest cards — computed against the sample's actual
    # measurement (wing_length_cm, beak_mm), not whatever derived value_col
    # the chart happens to plot (beak_prominence, for Beakiest, is a
    # wing-relative ratio rather than a raw beak length, so it needs its
    # own average).
    wing_length_avg = observed_traits["wing_length_cm"].mean()
    beak_mm_avg = observed_traits["beak_mm"].mean()
    biggest_diff_cm = biggest["wing_length_cm"] - wing_length_avg
    smallest_diff_cm = wing_length_avg - smallest["wing_length_cm"]
    beakiest_diff_mm = beakiest["beak_mm"] - beak_mm_avg

    # Ranked by global IUCN Red List category (rarer/more threatened first —
    # see IUCN_RANK), with each species' global range size as a tie-breaker
    # (smaller range first, from the AVONET trait sample) since almost every
    # species on a UK patch list is globally "Least Concern" and would
    # otherwise tie constantly.
    rarity_rows = []
    for row in observed_species.itertuples():
        info = get_iucn_status(row.scientific_name)
        code = info.get("code")
        if code:
            rarity_rows.append(
                {
                    "common_name": row.common_name,
                    "scientific_name": row.scientific_name,
                    "iucn_code": code,
                    "iucn_taxon_id": info.get("iucn_taxon_id"),
                    "rank": IUCN_RANK.get(code, len(IUCN_RANK)),
                }
            )
    rarity = pd.DataFrame(rarity_rows).merge(
        all_traits[["scientific_name", "range_size_km2"]],
        on="scientific_name",
        how="left",
    )
    rarest = rarity.sort_values(["rank", "range_size_km2", "common_name"]).iloc[0]
    rarest_range_stat = (
        f"Global range: {rarest['range_size_km2']:,.0f} km²"
        if pd.notna(rarest.get("range_size_km2"))
        else None
    )

    return dict(
        furthest=furthest,
        observed_traits=observed_traits,
        biggest=biggest,
        smallest=smallest,
        beakiest=beakiest,
        biggest_diff_cm=biggest_diff_cm,
        smallest_diff_cm=smallest_diff_cm,
        beakiest_diff_mm=beakiest_diff_mm,
        rarest=rarest,
        rarest_range_stat=rarest_range_stat,
    )


with awards_tab:
    st.header("Species Awards")
    st.caption("Special sightings from this season")

    def first_sighting(common_name: str) -> dict:
        matches = observations[observations["common_name"] == common_name]
        row = matches.sort_values("date").iloc[0]
        return {"observer": row["observer"], "date": row["date"], "location": row["location"]}

    winners = compute_award_winners(observations)
    furthest = winners["furthest"]
    observed_traits = winners["observed_traits"]
    biggest = winners["biggest"]
    smallest = winners["smallest"]
    beakiest = winners["beakiest"]
    biggest_diff_cm = winners["biggest_diff_cm"]
    smallest_diff_cm = winners["smallest_diff_cm"]
    beakiest_diff_mm = winners["beakiest_diff_mm"]
    rarest = winners["rarest"]
    rarest_range_stat = winners["rarest_range_stat"]

    def render_comparison_chart(
        value_col: str, highlight_name: str, accent: str, y_title: str,
    ) -> None:
        """Lollipop chart of every species observed this season on value_col,
        with the award winner picked out in colour and a dashed rule at the
        sample average."""
        avg = observed_traits[value_col].mean()
        chart_df = observed_traits[["scientific_name", "common_name", value_col]].copy()
        chart_df["Highlighted"] = np.where(
            chart_df["scientific_name"] == highlight_name, "This bird", "Sample"
        )
        # First-sighting observer per species, for the tooltip — a species
        # can have several observers across the season, so this matches the
        # same "who gets credit" rule the award cards themselves use
        # (first_sighting above).
        first_observer_by_species = (
            observations.sort_values("date")
            .drop_duplicates("common_name")
            .set_index("common_name")["observer"]
        )
        chart_df["observer"] = chart_df["common_name"].map(first_observer_by_species)
        base = alt.Chart(chart_df).encode(
            x=alt.X(
                "scientific_name:N",
                sort="-y",
                title=None,
                axis=alt.Axis(labels=False, ticks=False, domain=False),
            ),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            color=alt.Color(
                "Highlighted:N",
                scale=alt.Scale(
                    domain=["This bird", "Sample"], range=[accent, "#CBCBCB"]
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("common_name:N", title="Species"),
                alt.Tooltip("scientific_name:N", title="Scientific name"),
                alt.Tooltip("observer:N", title="Spotted by"),
                alt.Tooltip(f"{value_col}:Q", title=y_title, format=".1f"),
            ],
        )
        stems = base.mark_bar(size=5)
        dots = base.mark_circle(size=90)
        avg_rule = (
            alt.Chart(pd.DataFrame({"avg": [avg]}))
            .mark_rule(strokeDash=[6, 4], color="#666666", size=1.5)
            .encode(y="avg:Q")
        )
        chart = (
            (stems + dots + avg_rule)
            .properties(height=190)
            .configure_axis(labelFontSize=11, titleFontSize=12, grid=False)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Sample average:({avg:.1f})")

    def render_furthest_extra() -> None:
        campus = list(CAMPUS_COORDS)
        spot = [furthest["lat"], furthest["lng"]]
        midpoint = [(campus[0] + spot[0]) / 2, (campus[1] + spot[1]) / 2]
        # A conservative (narrower-than-real) width estimate — see
        # zoom_for_bounds — since this card's actual rendered width isn't
        # known here; height is exact, from the st_folium(height=190) call
        # below.
        mini_zoom = zoom_for_bounds(
            min(campus[0], spot[0]), max(campus[0], spot[0]),
            min(campus[1], spot[1]), max(campus[1], spot[1]),
            map_width_px=220, map_height_px=190, max_zoom=16,
        )
        fmap = folium.Map(location=midpoint, zoom_start=mini_zoom, tiles=None, max_zoom=16)
        folium.TileLayer(
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Tiles &copy; Esri",
            control=False,
            max_zoom=16,
        ).add_to(fmap)
        folium.CircleMarker(
            campus, radius=6, color="#333333", fill=True, fill_opacity=1,
            tooltip="UWE Frenchay Campus",
        ).add_to(fmap)
        folium.CircleMarker(
            spot, radius=6, color=AWARD_ACCENTS["furthest"], fill=True,
            fill_opacity=1, tooltip=furthest["common_name"],
        ).add_to(fmap)
        folium.PolyLine(
            [campus, spot], color=AWARD_ACCENTS["furthest"], weight=3,
            dash_array="8,6",
        ).add_to(fmap)
        st_folium(
            fmap, use_container_width=True, height=190, key="award-furthest-map"
        )
        st.caption(f"📍 {furthest['location']}")

    def render_rarest_extra() -> None:
        st.markdown(f"**📋 IUCN {IUCN_LABELS[rarest['iucn_code']]} criteria**")
        st.write(IUCN_CRITERIA[rarest["iucn_code"]])
        # Only resolves to anything if an IUCN_API_TOKEN is configured (see
        # .streamlit/secrets.toml.example) — a no-op enhancement otherwise.
        assessment = get_iucn_criteria(rarest.get("iucn_taxon_id"))
        main_criterion = describe_iucn_criteria(assessment.get("criteria"))
        if main_criterion:
            st.caption(f"Assessed as threatened due to {main_criterion}.")

    awards = [
        dict(
            key="award-furthest",
            icon="🧭",
            title="Furthest Bird",
            accent=AWARD_ACCENTS["furthest"],
            common_name=furthest["common_name"],
            scientific_name=furthest["scientific_name"],
            metric_value=f"{furthest['distance_km']:.1f} km",
            metric_label="from Frenchay Campus",
            observer=furthest["observer"],
            date=furthest["date"],
            location=furthest["location"],
            render_extra=render_furthest_extra,
        ),
        dict(
            key="award-biggest",
            icon="🦅",
            title="Biggest Bird",
            accent=AWARD_ACCENTS["biggest"],
            common_name=biggest["common_name"],
            scientific_name=biggest["scientific_name"],
            metric_value=f"{biggest['wing_length_cm']:g} cm",
            metric_label="wing length",
            extra_stat=f"{biggest_diff_cm:.1f} cm bigger than the average bird in our sample",
            render_extra=lambda: render_comparison_chart(
                "wing_length_cm", biggest["scientific_name"], AWARD_ACCENTS["biggest"],
                "Wing Length (cm)",
            ),
            **first_sighting(biggest["common_name"]),
        ),
        dict(
            key="award-smallest",
            icon="🐤",
            title="Smallest Bird",
            accent=AWARD_ACCENTS["smallest"],
            common_name=smallest["common_name"],
            scientific_name=smallest["scientific_name"],
            metric_value=f"{smallest['wing_length_cm']:g} cm",
            metric_label="wing length",
            extra_stat=f"{smallest_diff_cm:.1f} cm smaller than the average bird in our sample",
            render_extra=lambda: render_comparison_chart(
                "wing_length_cm", smallest["scientific_name"], AWARD_ACCENTS["smallest"],
                "Wing Length (cm)",
            ),
            **first_sighting(smallest["common_name"]),
        ),
        dict(
            key="award-rarest",
            icon="🚨",
            title="Rarest Bird",
            accent=AWARD_ACCENTS["rarest"],
            common_name=rarest["common_name"],
            scientific_name=rarest["scientific_name"],
            metric_value=IUCN_LABELS[rarest["iucn_code"]],
            metric_label=IUCN_DESCRIPTIONS[rarest["iucn_code"]],
            value_colors=IUCN_COLORS.get(rarest["iucn_code"]),
            extra_stat=rarest_range_stat,
            render_extra=render_rarest_extra,
            **first_sighting(rarest["common_name"]),
        ),
        dict(
            key="award-beakiest",
            icon="🦜",
            title="Beakiest Bird",
            accent=AWARD_ACCENTS["beakiest"],
            common_name=beakiest["common_name"],
            scientific_name=beakiest["scientific_name"],
            metric_value=f"1 : {beakiest['wing_beak_ratio']:.1f}",
            metric_label="beak length to wing length",
            extra_stat=f"{beakiest_diff_mm:.1f} mm bigger beak than the average bird in our sample",
            render_extra=lambda: render_comparison_chart(
                "beak_prominence", beakiest["scientific_name"], AWARD_ACCENTS["beakiest"],
                "Beak length (% of wing length)",
            ),
            **first_sighting(beakiest["common_name"]),
        ),
    ]

    for i, award in enumerate(awards):
        render_award_row(i, **award)
        if i < len(awards) - 1:
            st.write("")


with upload_tab:
    st.header("Add Your Sightings")
    st.caption(
        "Upload your eBird CSV export (eBird → My eBird Data → Download My "
        "Data) in the same format as the class dataset. eBird's own export "
        "doesn't include your name — enter it below so your sightings are "
        "credited on the leaderboard and awards."
    )

    username = st.text_input(
        "Your name",
        help=(
            "Required unless your file already has a \"Name\" column of its "
            "own (e.g. someone already combined several people's exports). "
            "If given, this replaces that column throughout your file."
        ),
    )
    uploaded_file = st.file_uploader("eBird CSV export", type="csv")

    if uploaded_file is not None:
        try:
            new_rows = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeDecodeError):
            st.error("Couldn't read that file — make sure it's a valid CSV export from eBird.")
            new_rows = None

        if new_rows is not None:
            errors = validate_upload_schema(new_rows)
            if "Name" not in new_rows.columns and not username.strip():
                errors.append(
                    "Your file doesn't have a \"Name\" column, so please "
                    "enter your name above to credit your sightings."
                )
            if errors:
                st.error("This file can't be added yet:\n\n" + "\n".join(f"- {e}" for e in errors))
            else:
                new_rows, row_notes = clean_upload_rows(new_rows)
                for note in row_notes:
                    st.warning(note)

                if username.strip():
                    new_rows["Name"] = username.strip()

                # Submission ID is eBird's own unique checklist identifier —
                # the natural way to spot "I already uploaded this file" (or
                # part of it) without asking anyone to remember what they've
                # submitted before.
                conn = st.connection("gsheets", type=GSheetsConnection)
                existing = df_conn = conn.read(ttl=0)

                duplicate_count = 0
                if "Submission ID" in new_rows.columns and "Submission ID" in existing.columns:
                    is_duplicate = new_rows["Submission ID"].isin(
                        set(existing["Submission ID"].dropna())
                    )
                    duplicate_count = int(is_duplicate.sum())
                    new_rows = new_rows[~is_duplicate]

                status = f"Found {len(new_rows)} new observation(s) to add"
                if duplicate_count:
                    status += f" ({duplicate_count} already in the dashboard, skipped)"
                st.write(status + ".")

                if len(new_rows) > 0:
                    preview_cols = [
                        c for c in ["Common Name", "Date", "Location", "Name"]
                        if c in new_rows.columns
                    ]
                    st.dataframe(new_rows[preview_cols].head(10), use_container_width=True)

                    if st.button("Add to the dashboard", type="primary"):
                        # 1) combine first, 2) then clean the whole result —
                        # not just the new rows — so the scrub (and any
                        # future cleaning step) always runs against the
                        # dataset as it will actually be saved.
                        combined = pd.concat([existing, new_rows], ignore_index=True)
                        combined = clean_dataset(combined)
                        conn.update(data=combined)

                        load_observations.clear()
                        st.success(
                            f"Added {len(new_rows)} observation(s)! "
                            "Switch tabs to see them reflected."
                        )
                elif duplicate_count:
                    st.info("Every row in this file is already in the dashboard — nothing to add.")
                else:
                    st.error("No usable rows were left in this file after the checks above.")
