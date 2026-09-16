"""Strip precise home addresses out of data/wecs_birds.csv's Location
column before the dashboard goes live.

The eBird export mixes genuine, already-generic location names (park
names, made-up patch nicknames like "duck zone") with a handful of values
that are clearly someone's home address — a house number and street
name, e.g. "28 Ferndale Drive, Scotland" — because eBird falls back to
the address when a personal location hasn't been given a proper name.

Those get replaced with the row's own County (already in the eBird
export, e.g. "Glasgow" or "Bridgend") rather than collapsed all the way
down to the bare country/region name ("Scotland", "Wales") — enough to
de-identify (no house number, no street) without being so coarse it's
useless for anything that groups sightings by area.

Only touches the Location text column. Latitude/Longitude for those same
rows still pinpoint the exact spot on the map — this script prints a
warning about that rather than silently coarsening coordinates nobody
asked it to touch.

Safe to re-run: it always starts from data/wecs_birds.csv.bak (the
untouched original) if one exists, rather than re-scrubbing its own
previous output, so tightening this rule later or fixing a bad first
pass doesn't compound anything.

Run from the project root:
    python3 scripts/scrub_locations.py
"""

import re
import shutil
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/wecs_birds.csv")
BACKUP_PATH = DATA_PATH.with_suffix(".csv.bak")

# A location is treated as a home address if it starts with a house
# number, and either (a) has a comma-separated region after it — e.g.
# "28 Ferndale Drive, Scotland" — or (b) simply ends in a common English
# street-type word — e.g. "42 Test Lane", which has no trailing region at
# all (this is what a real uploaded eBird export's fallback location tends
# to look like, as opposed to this repo's original hand-made test data,
# which always had a trailing region). Anything else (park names, made-up
# patch nicknames, bare region names) is left alone. Kept in sync with
# app.py's _looks_like_address — the app runs the same check on every
# upload, this script is for a one-off pass over the whole file.
ADDRESS_LEADING_NUMBER = re.compile(r"^\d+\s+(.+)$")
STREET_SUFFIX_WORDS = {
    "street", "road", "drive", "lane", "avenue", "close", "way", "court",
    "place", "crescent", "terrace", "walk", "row", "square", "mews",
    "gardens", "grove", "boulevard", "gate", "hill", "farm",
}


def is_address(value) -> bool:
    if not isinstance(value, str):
        return False
    match = ADDRESS_LEADING_NUMBER.match(value.strip())
    if not match:
        return False
    rest = match.group(1)
    if "," in rest:
        return True
    words = rest.split()
    last_word = re.sub(r"\W", "", words[-1]).lower() if words else ""
    return last_word in STREET_SUFFIX_WORDS


def main() -> None:
    source_path = BACKUP_PATH if BACKUP_PATH.exists() else DATA_PATH
    df = pd.read_csv(source_path)

    address_mask = df["Location"].apply(is_address)
    if not address_mask.any():
        print(f"No address-like locations found in {source_path} — nothing to change.")
        return

    original = df["Location"].copy()
    df.loc[address_mask, "Location"] = df.loc[address_mask, "County"]

    print(f"Scrubbing {address_mask.sum()} row(s) (source: {source_path}):")
    for orig, new in sorted(set(zip(original[address_mask], df.loc[address_mask, "Location"]))):
        print(f"  {orig!r} -> {new!r}")

    if not BACKUP_PATH.exists():
        shutil.copy(DATA_PATH, BACKUP_PATH)
        print(f"\nBacked up original to {BACKUP_PATH}")

    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {DATA_PATH}")

    print(
        "\nNote: Latitude/Longitude for these rows are still the exact "
        "coordinates of those addresses — the Location text is scrubbed but "
        "the map markers aren't. Let me know if you want those coarsened "
        "too (e.g. rounded, or snapped to a region centroid)."
    )


if __name__ == "__main__":
    main()
