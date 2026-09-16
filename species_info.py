from __future__ import annotations

import os
import re

import pandas as pd
import requests
import streamlit as st

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_IUCN_URL = "https://api.gbif.org/v1/species/{usage_key}/iucnRedListCategory"
REQUEST_TIMEOUT = 5
HEADERS = {"User-Agent": "ClassBiodiversityDashboard/1.0 (educational project)"}

DATA_DIR = "data"
TRAITS_PATH = os.path.join(DATA_DIR, "AVONET_traits.csv")
BOU_LIST_PATH = os.path.join(DATA_DIR, "BOU_British_List_2025.csv")
UK_STATUS_PATH = os.path.join(DATA_DIR, "uk_conservation_status.csv")
SPECIES_CACHE_PATH = os.path.join(DATA_DIR, "species_info_cache.csv")
SPECIES_CACHE_COLUMNS = [
    "common_name",
    "scientific_name",
    "image_url",
    "extract",
    "iucn_code",
    "iucn_label",
]

IUCN_LABELS = {
    "EX": "Extinct",
    "EW": "Extinct in the Wild",
    "CR": "Critically Endangered",
    "EN": "Endangered",
    "VU": "Vulnerable",
    "NT": "Near Threatened",
    "LC": "Least Concern",
    "DD": "Data Deficient",
    "NE": "Not Evaluated",
}

IUCN_DESCRIPTIONS = {
    "EX": "Confirmed extinct",
    "EW": "Survives only in captivity",
    "CR": "Extremely high extinction risk",
    "EN": "Very high extinction risk",
    "VU": "High extinction risk",
    "NT": "Close to qualifying as threatened",
    "LC": "Widespread and abundant",
    "DD": "Not enough data to assess",
    "NE": "Not yet assessed",
}

# IUCN Red List Categories and Criteria (v3.1) definitions.
IUCN_CRITERIA = {
    "EX": "There is no reasonable doubt that the last individual has died.",
    "EW": (
        "Known only to survive in cultivation, in captivity, or as a "
        "naturalised population well outside its past range."
    ),
    "CR": "Facing an extremely high risk of extinction in the wild.",
    "EN": "Facing a very high risk of extinction in the wild.",
    "VU": "Facing a high risk of extinction in the wild.",
    "NT": (
        "Close to qualifying for, or likely to qualify for, a threatened "
        "category in the near future."
    ),
    "LC": (
        "Widespread and abundant — does not qualify for Near Threatened or "
        "any threatened category."
    ),
    "DD": (
        "Inadequate information to make a direct or indirect assessment of "
        "extinction risk."
    ),
    "NE": "Has not yet been evaluated against the criteria.",
}

# Rarity ranking for the "Rarest Bird" award — lower is rarer. DD and NE mean
# "unknown", not "not at risk", so they're ranked as uninformative rather
# than safe, alongside the extinct categories which can never actually apply
# to an observed bird but are included for a complete ordering.
IUCN_RANK = {
    "EX": 0, "EW": 1, "CR": 2, "EN": 3, "VU": 4, "NT": 5, "LC": 6, "DD": 7, "NE": 8,
}

# IUCN Red List Criteria (v3.1) — what a species was actually assessed
# against to land in its category, as opposed to IUCN_CRITERIA above (which
# only explains the category itself). A full code like "A4abcde" or
# "B2ab(iii)" also carries sub-criteria letters (the evidence used) and
# roman-numeral qualifiers, but describe_iucn_criteria() below only ever
# looks up the main criterion (the leading letter + number, e.g. "A4") —
# more than that is more technical detail than a class dashboard needs.
# This is reference data, not per-species data — it means the same thing
# for every species, so it lives here as a lookup table rather than
# something added per-row to a traits CSV.
IUCN_CRITERION_MEANINGS = {
    "A1": (
        "A population reduction over 10 years where the drivers have stopped or are reversible"
    ),
    "A2": (
        "A population reduction over 10 years where the drivers have not stopped or may not be reversible"
    ),
    "A3": "a population reduction projected or suspected over the next 10 years",
    "A4": (
        "a population reduction observed or suspected, where the causes may not have ceased, be understood, or may not be reversible"
    ),
    "B1": "a small Extent of Occurrence, combined with fragmentation, decline, or fluctuation",
    "B2": "a small Area of Occupancy, combined with fragmentation, decline, or fluctuation",
    "C1": "a small, declining population with a specified rate of decline",
    "C2": (
        "a small, declining population whose structure is fragmented or "
        "concentrated in one subpopulation"
    ),
    "D": "a very small or geographically restricted population",
    "D1": "a very small population (fewer than 1,000 mature individuals)",
    "D2": (
        "a population with a very restricted area of occupancy or number of "
        "locations, making it prone to the effects of human activity or "
        "chance events"
    ),
    "E": "a quantitative extinction-risk analysis (e.g. Population Viability Analysis)",
}

_IUCN_MAIN_CRITERION_RE = re.compile(r"[A-E]\d?")


def describe_iucn_criteria(criteria: str) -> str | None:
    """The plain-English meaning of a raw IUCN Red List criteria string's
    main criterion only — e.g. "A4abcde" -> IUCN_CRITERION_MEANINGS["A4"],
    ignoring the "abcde" sub-criteria letters, any "(...)" roman-numeral
    qualifiers, and any further criteria after a ";" or "+". None if
    criteria is empty or doesn't start with a recognised criterion.
    """
    if not criteria:
        return None
    match = _IUCN_MAIN_CRITERION_RE.match(criteria.strip())
    return IUCN_CRITERION_MEANINGS.get(match.group()) if match else None

# Official IUCN Red List category colours: {code: (background, text)}
IUCN_COLORS = {
    "EX": ("#000000", "#FFFFFF"),
    "EW": ("#542344", "#FFFFFF"),
    "CR": ("#D81E05", "#FFFFFF"),
    "EN": ("#FC7F3F", "#FFFFFF"),
    "VU": ("#F9E814", "#000000"),
    "NT": ("#CCE226", "#000000"),
    "LC": ("#60C659", "#FFFFFF"),
    "DD": ("#D1D1C6", "#000000"),
    "NE": ("#FFFFFF", "#000000"),
}


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_wikipedia_summary(title: str) -> dict:
    try:
        response = requests.get(
            WIKIPEDIA_SUMMARY_URL.format(title=title.replace(" ", "_")),
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return {}
    return {
        "image_url": data.get("thumbnail", {}).get("source"),
        "extract": data.get("extract"),
    }


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_iucn_status(scientific_name: str) -> dict:
    try:
        match_response = requests.get(
            GBIF_MATCH_URL,
            params={"name": scientific_name},
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )
        match_response.raise_for_status()
        usage_key = match_response.json().get("usageKey")
        if usage_key is None:
            return {}

        status_response = requests.get(
            GBIF_IUCN_URL.format(usage_key=usage_key),
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
        )
        status_response.raise_for_status()
        payload = status_response.json()
        code = payload.get("code")
    except requests.RequestException:
        return {}
    if not code:
        return {}
    return {
        "code": code,
        "label": IUCN_LABELS.get(code, code),
        # The IUCN Red List's own taxon ID (SIS ID) for this species — GBIF
        # passes it straight through. Needed to query the official IUCN Red
        # List API directly (get_iucn_criteria below), since that API is
        # keyed by SIS ID rather than scientific name matching.
        "iucn_taxon_id": payload.get("iucnTaxonID"),
    }


IUCN_REDLIST_API_BASE = "https://api.iucnredlist.org/api/v4"


def _iucn_redlist_headers() -> dict | None:
    """Bearer-token headers for the official IUCN Red List API (separate
    from — and far more detailed than — the GBIF passthrough above), or
    None if no token is configured. Requires an [IUCN_API_TOKEN] entry in
    .streamlit/secrets.toml (see .streamlit/secrets.toml.example) — get one
    free at https://api.iucnredlist.org/users/sign_up. Never hardcode the
    token here.
    """
    token = st.secrets.get("IUCN_API_TOKEN")
    if not token:
        return None
    return {**HEADERS, "Authorization": f"Bearer {token}"}


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_iucn_criteria(iucn_taxon_id: str) -> dict:
    """The specific Red List criteria a species was assessed under (e.g.
    "B2ab(iii)"), from the official IUCN Red List API — richer than the
    bare category GBIF returns. Requires an API token; returns {} silently
    if none is configured, so this is a pure enhancement, not a hard
    dependency.

    NOTE: the exact field name for the criteria string in the assessment
    payload isn't nailed down against a live response yet (no token was
    available while writing this) — this tries a few plausible names and
    also returns the raw assessment dict under "raw" so it's easy to check
    which key actually holds it and adjust CRITERIA_FIELD_CANDIDATES below.
    """
    headers = _iucn_redlist_headers()
    if not headers or not iucn_taxon_id:
        return {}
    try:
        taxon_response = requests.get(
            f"{IUCN_REDLIST_API_BASE}/taxa/sis/{iucn_taxon_id}",
            timeout=REQUEST_TIMEOUT,
            headers=headers,
        )
        taxon_response.raise_for_status()
        assessments = taxon_response.json().get("assessments", [])
        latest = next((a for a in assessments if a.get("latest")), None) or (
            assessments[0] if assessments else None
        )
        assessment_id = latest.get("assessment_id") if latest else None
        if not assessment_id:
            return {}

        assessment_response = requests.get(
            f"{IUCN_REDLIST_API_BASE}/assessment/{assessment_id}",
            timeout=REQUEST_TIMEOUT,
            headers=headers,
        )
        assessment_response.raise_for_status()
        assessment = assessment_response.json()
    except requests.RequestException:
        return {}

    criteria_field_candidates = ("criteria", "red_list_criteria", "redListCriteria")
    criteria = next(
        (assessment[f] for f in criteria_field_candidates if assessment.get(f)), None
    )
    return {"criteria": criteria, "assessment_id": assessment_id, "raw": assessment}


# AVONET's reference taxonomy predates a couple of splits/reclassifications
# that the observations data (eBird/Clements taxonomy) already reflects —
# patch those species' AVONET names so they still join correctly.
AVONET_SCIENTIFIC_NAME_ALIASES = {
    "Corvus monedula": "Coloeus monedula",  # Eurasian Jackdaw
    "Alexandrinus krameri": "Psittacula krameri",  # Ring-necked/Rose-ringed Parakeet
}


@st.cache_data
def _load_species_traits() -> dict:
    """Measurements from AVONET (Tobias et al. 2022, "AVONET: morphological,
    ecological and geographical data for all birds"), read from
    data/AVONET_traits.csv. Used for the "Biggest", "Smallest" and
    "Beakiest" awards.

    Keyed by scientific name rather than common name: common names in
    observation exports aren't consistent (e.g. "Common Woodpigeon" vs
    "Common Wood-Pigeon" for the same species), while scientific names are
    unambiguous.

    beak_mm is AVONET's Beak.Length_Culmen specifically — the standard bill
    length measurement, taken from the base of the skull to the tip — as
    opposed to Beak.Length_Nares, Beak.Width or Beak.Depth.

    wing_length_cm is AVONET's Wing.Length (mm, converted to cm here). This
    is the folded wing (chord) length used in ringing, not wingtip-to-wingtip
    wingspan — AVONET doesn't include true wingspan at all — so the "Biggest"
    /"Smallest Bird" awards and their UI labels are framed as wing length,
    not wingspan.

    range_size_km2 is AVONET's Range.Size — each species' global Extent of
    Occurrence in km2, from BirdLife International's range maps. Used only
    as a "Rarest Bird" tie-breaker (smaller range first) when two species
    share the same IUCN category, so it's allowed to be missing (unlike
    beak_mm/wing_length_cm) rather than dropping the whole row.
    """
    if not os.path.exists(TRAITS_PATH):
        return {}
    df = pd.read_csv(
        TRAITS_PATH,
        usecols=["Species1", "Beak.Length_Culmen", "Wing.Length", "Range.Size"],
    )
    df["Species1"] = df["Species1"].replace(AVONET_SCIENTIFIC_NAME_ALIASES)
    df = df.rename(
        columns={
            "Species1": "scientific_name",
            "Beak.Length_Culmen": "beak_mm",
            "Range.Size": "range_size_km2",
        }
    )
    df["wing_length_cm"] = df["Wing.Length"] / 10
    df = (
        df.drop(columns=["Wing.Length"])
        .dropna(subset=["beak_mm", "wing_length_cm"])
        .drop_duplicates(subset="scientific_name")
    )
    return df.set_index("scientific_name").to_dict(orient="index")


@st.cache_data
def _load_uk_conservation_status() -> dict:
    """UK conservation status (BTO/RSPB "Birds of Conservation Concern"
    Red/Amber/Green lists), read from data/uk_conservation_status.csv. Used
    for the "Rarest" award, since species here are otherwise all globally
    IUCN "Least Concern".
    """
    if not os.path.exists(UK_STATUS_PATH):
        return {}
    df = pd.read_csv(UK_STATUS_PATH)
    return dict(zip(df["common_name"], df["status"]))


@st.cache_data
def _load_uk_species() -> set:
    """Scientific names on the official BOU British List (Categories A, B
    and C — species with an established, naturally-occurring or naturalised
    UK population), read from data/BOU_British_List_2025.csv. Used to keep
    the trait-award comparison charts scoped to birds that could plausibly
    turn up in the UK, rather than the full worldwide AVONET sample
    (11,000+ species — including things like kiwis, whose near-vestigial
    0.01cm wing length blows the beak-to-wing-length chart's scale out to
    nonsense if they're left in).
    """
    if not os.path.exists(BOU_LIST_PATH):
        return set()
    df = pd.read_csv(BOU_LIST_PATH)
    is_abc = df["Category"].astype(str).str.contains(r"[ABC]", regex=True)
    return set(df.loc[is_abc, "Scientific name"].dropna())


SPECIES_TRAITS = _load_species_traits()
UK_SPECIES = _load_uk_species()
UK_CONSERVATION_STATUS = _load_uk_conservation_status()


@st.cache_data
def _load_species_cache() -> pd.DataFrame:
    if os.path.exists(SPECIES_CACHE_PATH):
        return pd.read_csv(SPECIES_CACHE_PATH)
    return pd.DataFrame(columns=SPECIES_CACHE_COLUMNS)


def _cached_row_to_info(row: pd.Series) -> dict:
    return {
        "image_url": row["image_url"] if pd.notna(row["image_url"]) else None,
        "extract": row["extract"] if pd.notna(row["extract"]) else None,
        "iucn_code": row["iucn_code"] if pd.notna(row["iucn_code"]) else None,
        "iucn_label": row["iucn_label"] if pd.notna(row["iucn_label"]) else None,
    }


def get_species_info(common_name: str, scientific_name: str) -> dict:
    """Image + IUCN status for one species, backed by a persistent CSV cache
    (data/species_info_cache.csv) instead of re-fetching from Wikipedia/GBIF
    on every run. A species already in the cache is read straight from disk;
    a new one is fetched live once, then appended so it never needs fetching
    again — including across app restarts, unlike an in-memory cache.
    """
    cache = _load_species_cache()
    match = cache[cache["common_name"] == common_name]
    if not match.empty:
        return _cached_row_to_info(match.iloc[0])

    wiki = get_wikipedia_summary(common_name)
    if not wiki.get("image_url") and scientific_name:
        wiki = get_wikipedia_summary(scientific_name) or wiki
    iucn = get_iucn_status(scientific_name)
    info = {
        "image_url": wiki.get("image_url"),
        "extract": wiki.get("extract"),
        "iucn_code": iucn.get("code"),
        "iucn_label": iucn.get("label"),
    }

    # Only persist a genuine hit — an empty result from a transient network
    # blip should be retried next run, not cached as a permanent miss.
    if info["image_url"] or info["iucn_code"]:
        new_row = {
            "common_name": common_name,
            "scientific_name": scientific_name,
            **info,
        }
        updated = pd.concat([cache, pd.DataFrame([new_row])], ignore_index=True)
        updated.to_csv(SPECIES_CACHE_PATH, index=False)
        _load_species_cache.clear()

    return info
