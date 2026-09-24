"""
fetch_training_set.py

REPLACES slice_openfoodfacts.py.

Why: the bulk TSV export doesn't include nova_group by default -- Open
Food Facts' own export documentation confirms computed fields like
nova_group are opt-in "extra_fields" on custom exports, not present in
the standard dump. Verified live: the plain bulk file has 163 columns,
zero of them NOVA-related.

Fix: pull labeled training rows directly from Open Food Facts' live
search API (search.openfoodfacts.org -- the "Search-a-licious" service,
not the older /cgi/search.pl). This is the same live API the FastAPI
backend already calls at inference time (Stage 6), so this script
reuses a skill you need anyway rather than introducing a new one.

Also resolves PRD standing risk "India-only training is unviable": this
script does NOT filter by country for training data at all (there
simply aren't enough India-tagged + NOVA-labeled rows to hit the
~200/class minimum). India stays a live-demo story, not a training
filter, exactly as investigated.

Usage:
    python fetch_training_set.py
"""

import json
import os
import time
import urllib.parse
import urllib.request

TARGET_PER_CLASS = 5000      # 20k rows total; API caps paging at 10000/class
PAGE_SIZE = 200               # confirmed working ceiling for this API
OUT_PATH = "data/raw/openfoodfacts_training_set.csv"
USER_AGENT = "Chewsy-CourseProject/1.0 (student project)"

# Leakage guard: never request Nutri-Score/NutriScore-derived fields.
# (Old export named this nutrition_grade_fr; this API's equivalent
# fields are nutrition_grades / nutriscore_grade / nutriscore_score /
# nutriscore_data -- same leakage risk, new names, excluded here too.)
FORBIDDEN_LEAKAGE_FIELDS = {
    "nutrition_grades", "nutriscore_grade", "nutriscore_score", "nutriscore_data",
}


FIELDS = (
    "code,product_name,brands,categories_tags,ingredients_tags,"
    "additives_n,ingredients_n,unknown_ingredients_n,nutriments"
)  # explicitly excludes nutrition_grades/nutriscore_* -- the leakage guard


def fetch_page(nova_class: int, page: int) -> dict:
    query = urllib.parse.urlencode({
        "q": f"nova_groups:{nova_class}",
        "page": page,
        "page_size": PAGE_SIZE,
        "fields": FIELDS,
    })
    url = f"https://search.openfoodfacts.org/search?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def flatten_hit(hit: dict, nova_class: int) -> dict:
    nutriments = hit.get("nutriments") or {}
    ingredients_tags = hit.get("ingredients_tags") or []
    categories_tags = hit.get("categories_tags") or []

    row = {
        "code": hit.get("code"),
        "product_name": hit.get("product_name"),
        "brands": hit.get("brands"),
        "categories_tags": ",".join(categories_tags),
        # ingredients_tags are pre-normalized ingredient tokens
        # (e.g. "en:sugar", "en:palm-oil") -- joined into a
        # space-separated pseudo-text field for TF-IDF in Stage 3.
        # This is arguably cleaner than raw free text: it's already
        # tokenized and de-duplicated by OFF's own pipeline.
        "ingredients_pseudo_text": " ".join(
            t.split(":", 1)[-1].replace("-", " ") for t in ingredients_tags
        ),
        "additives_n": hit.get("additives_n"),
        "ingredients_n": hit.get("ingredients_n"),
        "unknown_ingredients_n": hit.get("unknown_ingredients_n"),
        "nova_group": nova_class,  # ground truth from the query itself
        "energy_100g": nutriments.get("energy-kcal_100g"),
        "fat_100g": nutriments.get("fat_100g"),
        "saturated_fat_100g": nutriments.get("saturated-fat_100g"),
        "carbohydrates_100g": nutriments.get("carbohydrates_100g"),
        "sugars_100g": nutriments.get("sugars_100g"),
        "fiber_100g": nutriments.get("fiber_100g"),
        "proteins_100g": nutriments.get("proteins_100g"),
        "salt_100g": nutriments.get("salt_100g"),
        "sodium_100g": nutriments.get("sodium_100g"),
    }

    # Leakage guard: since FIELDS above never requests nutrition_grades/
    # nutriscore_*, they should never appear in a hit. Assert this rather
    # than silently trust it -- catches an API/param change immediately
    # instead of leaking Nutri-Score into training data unnoticed.
    assert not FORBIDDEN_LEAKAGE_FIELDS & set(hit.keys()), (
        f"Leakage field present in API response: "
        f"{FORBIDDEN_LEAKAGE_FIELDS & set(hit.keys())}"
    )
    return row


def fetch_class(nova_class: int) -> list[dict]:
    rows = []
    page = 1
    while len(rows) < TARGET_PER_CLASS:
        data = fetch_page(nova_class, page)
        hits = data.get("hits", [])
        if not hits:
            break  # ran out of results before hitting target
        rows.extend(flatten_hit(h, nova_class) for h in hits)
        print(f"  NOVA {nova_class}: page {page} -> {len(rows)} rows so far")
        page += 1
        time.sleep(0.2)  # polite pacing, not aggressive
    return rows[:TARGET_PER_CLASS]


def main():
    all_rows = []
    for nova_class in (1, 2, 3, 4):
        print(f"Fetching NOVA class {nova_class}...")
        all_rows.extend(fetch_class(nova_class))

    import csv
    os.makedirs("data/raw", exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal rows: {len(all_rows)}")
    from collections import Counter
    print("Class balance:", Counter(r["nova_group"] for r in all_rows))
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
