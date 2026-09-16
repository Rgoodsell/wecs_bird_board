"""Sketch: scrape wingspan/length/weight trait data from the RSPB Bird A-Z,
driven by data/BOU_British_List_2025.csv as the master species list.

STATUS: draft, not run. Selectors below are a best guess at RSPB's page
structure (parsed from markdown-rendered content, not raw HTML) and almost
certainly need adjusting once you actually inspect a live page — view-source
a couple of RSPB bird pages and fix parse_key_facts() before trusting output.

Known gaps to sort out before this is usable:
  - RSPB's URL slugs don't always match a simple slugify() of the BOU
    vernacular name (e.g. it may be "robin" not "european-robin", or use a
    different common name entirely). RSPB_SLUG_OVERRIDES below is a manual
    patch table for exactly that — expect to grow it a lot after the first
    run reports its misses.
  - RSPB pages don't reliably publish bill/beak length — that's BTO
    BirdFacts' data, not RSPB's. This script only covers what RSPB actually
    has (length, wingspan, weight); beak_mm needs a separate scraper against
    BTO BirdFacts.
  - Be a good citizen: check https://www.rspb.org.uk/robots.txt before
    running this for real, keep REQUEST_DELAY_SECONDS conservative, and
    identify the script with a real contact in USER_AGENT.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BOU_LIST_PATH = Path("data/BOU_British_List_2025.csv")
OUTPUT_PATH = Path("data/rspb_traits_raw.csv")

RSPB_BASE_URL = "https://www.rspb.org.uk/birds-and-wildlife/birds-and-wildlife-guides/bird-a-z/{slug}"
REQUEST_DELAY_SECONDS = 2.0  # be polite — this is a small class project, not a crawler
REQUEST_TIMEOUT = 10
USER_AGENT = "WECS-Bird-Board-Research/0.1 (UWE Bristol class project; contact: <fill in email>)"

# BOU vernacular name -> RSPB URL slug, for the cases a plain slugify() gets
# wrong. Populate this as failures come back from the first real run rather
# than guessing all of them up front.
RSPB_SLUG_OVERRIDES: dict[str, str] = {
    "Woodpigeon": "wood-pigeon",
    "Carrion Crow": "crow",
    # ...
}


@dataclass
class TraitRow:
    common_name: str
    scientific_name: str
    length_cm: float | None
    wingspan_cm: float | None
    weight_g: float | None
    source_url: str


def slugify(common_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", common_name.lower()).strip("-")


def rspb_slug_for(common_name: str) -> str:
    return RSPB_SLUG_OVERRIDES.get(common_name, slugify(common_name))


def load_bou_species(path: Path) -> list[tuple[str, str]]:
    """Returns [(common_name, scientific_name), ...] for Category A/B/C
    species — the ones that actually occur/have occurred in Britain, as
    opposed to escapes or historical records only. Adjust the category
    filter once you've looked at what's actually in the Category column for
    this file."""
    species = []
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            category = row.get("Category", "")
            if not any(c in category for c in "ABC"):
                continue
            name = row["British (English) vernacular name"].strip()
            sci = row["Scientific name"].strip()
            if name:
                species.append((name, sci))
    return species


def parse_midpoint(range_text: str) -> float | None:
    """"80-98cm" -> 89.0, "38cm" -> 38.0, "" -> None. RSPB's own formatting
    is inconsistent between species (units, dashes, ranges vs single
    figures) — this needs to be more forgiving once you've seen real
    examples fail it."""
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", range_text)]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def parse_key_facts(soup: BeautifulSoup) -> dict[str, str]:
    """RSPB bird pages have a 'Key facts' / stats block (length, wingspan,
    weight, conservation status, population, etc). This is a placeholder:
    it just looks for a label like 'Wingspan' anywhere on the page and
    grabs nearby text, rather than a real CSS selector — the actual DOM
    (class names, whether it's a <dl>, a table, or JSON in a <script>
    tag) needs checking on a live page before this is trustworthy."""
    facts: dict[str, str] = {}
    for label in ("Length", "Wingspan", "Weight"):
        el = soup.find(string=re.compile(rf"^\s*{label}\s*$", re.I))
        if el and el.find_next(string=True):
            facts[label] = el.find_next(string=True).strip()
    return facts


def fetch_species_traits(common_name: str, scientific_name: str, session: requests.Session) -> TraitRow:
    slug = rspb_slug_for(common_name)
    url = RSPB_BASE_URL.format(slug=slug)
    response = session.get(url, timeout=REQUEST_TIMEOUT)

    if response.status_code == 404:
        # Slug guess was wrong — needs an RSPB_SLUG_OVERRIDES entry.
        return TraitRow(common_name, scientific_name, None, None, None, url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    facts = parse_key_facts(soup)

    return TraitRow(
        common_name=common_name,
        scientific_name=scientific_name,
        length_cm=parse_midpoint(facts.get("Length", "")),
        wingspan_cm=parse_midpoint(facts.get("Wingspan", "")),
        weight_g=parse_midpoint(facts.get("Weight", "")),
        source_url=url,
    )


def main() -> None:
    species_list = load_bou_species(BOU_LIST_PATH)
    print(f"{len(species_list)} species loaded from {BOU_LIST_PATH}")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    rows: list[TraitRow] = []
    misses: list[str] = []

    for common_name, scientific_name in species_list:
        row = fetch_species_traits(common_name, scientific_name, session)
        if row.wingspan_cm is None:
            misses.append(common_name)
        rows.append(row)
        time.sleep(REQUEST_DELAY_SECONDS)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["common_name", "scientific_name", "length_cm", "wingspan_cm", "weight_g", "source_url"])
        for row in rows:
            writer.writerow([row.common_name, row.scientific_name, row.length_cm, row.wingspan_cm, row.weight_g, row.source_url])

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    if misses:
        print(f"{len(misses)} species had no wingspan found — check slugs/selectors: {misses}")


if __name__ == "__main__":
    main()
