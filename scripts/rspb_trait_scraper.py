"""Scape trait data from RSPB & BTO using BOU_British_List_2025.csv as the master species list.

Known gaps to sort out before this is usable:
  - RSPB's URL slugs don't always match a simple slugify() of the BOU
    vernacular name (e.g. it may be "robin" not "european-robin", or use a
    different common name entirely). RSPB_SLUG_OVERRIDES below is a manual
    patch table for exactly that — expect to grow it a lot after the first
    run reports its misses.
  - Be a good citizen: check https://www.rspb.org.uk/robots.txt before
    running this for real, keep REQUEST_DELAY_SECONDS conservative, and
    identify the script with a real contact in USER_AGENT.
"""

# ---- Packages
import requests
from bs4 import BeautifulSoup
import pathlib as pb
import pandas as pd
import json
import html
# ------

# --- set up paths for reading and saving
BOU_LIST_PATH = pb.Path("data/BOU_British_List_2025.csv")
OUTPUT_PATH = pb.Path("data/rspb_traits_raw.csv")
# ---

# ---- Functions
# Unescape text
def unescape_blob(text):
    text = html.unescape(text)      # handles &quot; &amp; etc.
    text = text.replace("&q;", '"')  # fallback for the non-standard form
    return text

# walk scraped trait blocks recursively and get key info
def find_key_recursive(obj , target, path = ""):
    if isinstance(obj , dict): # If obj is a dict
        for k, v in obj.items(): # if a dict loop through key value pairs
            new_path = f"{path}.{k}" if path else k # build names of location
            if k == target: # If in the right location
                yield new_path, v # print the path and value extracted
            yield from find_key_recursive(v , target, new_path) # recursively search through the new path
    elif isinstance(obj,list): # If a list instead of a dict
        for i, item in enumerate(obj): # loop across items 
            yield from find_key_recursive(item , target , f"{path}.{i}") # recursively search
# -----

# --- first get pages that bird traits are stored on from the .xml sitemap


# -------- Start scraping
base_URL = 'https://www.rspb.org.uk/birds-and-wildlife/'

# Set a header
HEADERS = {
    "User-Agent": (
        "RSPB-trait-scraper/1.0 "
        "(student engagement project, non-commercial; contact: robert.goodsell@uwe.ac.uk)"
    )
}

birds = ["woodpigeon", "robin" , "skylark"]
check_url = []
rows = []
for bird in birds:
    check_url = base_URL+ bird
    print(check_url)

    page = requests.get(check_url, headers=HEADERS, timeout=30)

    soup = BeautifulSoup(page.content, "html.parser") # get the page
    info = soup.find_all("script") # get the 

    # Find the info needed
    info_list = []
    for script in info:
        raw = script.string 
        if not raw or "specifications" not in raw:
            continue
        info_list.append(raw)


    obj = unescape_blob(info_list[0])
    parsed = json.loads(obj)

    traits = []
    for path, specs in find_key_recursive(parsed, "specifications"):
        traits.append(specs)

    row = {"common_name": path.split('.')[0], **traits[0]}
    rows.append(row)

df_out = pd.DataFrame(rows)  # build the DataFrame once, after the loop