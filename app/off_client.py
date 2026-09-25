"""Stage 6.1 -- live Open Food Facts client (PRD 6.1).

``fetch_product(barcode)`` hits the public product API with a hard 5s
timeout and converts every failure mode into a clean, typed exception the
FastAPI layer maps to 404/503 -- a hung or flaky upstream must never crash
the live demo (Hard rule 5).

Hard rule 11 -- inference leakage guard: the product response carries
``nova_group`` (OFF's own stored classification) plus the Nutri-Score
field family. Both are STRIPPED here and re-asserted downstream
(``src/clean.py::normalize_live_row``, ``app/main.py``): the model must
compute the verdict from raw nutrients/ingredients only, never read OFF's
label -- even when one exists. This mirrors ``FORBIDDEN_LEAKAGE_FIELDS``
assertion style in ``fetch_training_set.py`` (PRD 2.5 / 6.1).
"""

from __future__ import annotations

import requests

from src.clean import FORBIDDEN_LEAKAGE_FIELDS

API_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
TIMEOUT = 5.0  # Hard rule 5 -- never let a hung request kill the demo
USER_AGENT = "Chewsy-CourseProject/1.0 (student project)"

# Hard rule 11: OFF's stored NOVA label + the Nutri-Score family must never
# enter the feature row or the API response.
FORBIDDEN_RESPONSE_FIELDS = FORBIDDEN_LEAKAGE_FIELDS | {"nova_group"}


class ProductNotFound(Exception):
    """Barcode unknown to Open Food Facts -> HTTP 404."""


class OffAPIUnavailable(Exception):
    """Timeout / 5xx / malformed payload -> HTTP 503."""


def strip_forbidden(product: dict) -> dict:
    """Remove Hard rule 11 fields from a raw product payload.

    Returns a shallow copy; asserts nothing forbidden survives.
    """
    cleaned = {k: v for k, v in product.items() if k not in FORBIDDEN_RESPONSE_FIELDS}
    leaked = FORBIDDEN_RESPONSE_FIELDS & set(cleaned)
    assert not leaked, f"forbidden field(s) survived stripping: {leaked}"
    return cleaned


def fetch_product(barcode: str) -> dict:
    """Fetch one product by barcode; returns the payload with forbidden
    fields stripped.

    Raises:
        ProductNotFound: barcode not found (API ``status != 1``) -> 404.
        OffAPIUnavailable: timeout, connection error, 5xx, or non-JSON
            body -> 503.
    """
    url = API_URL.format(barcode=barcode)
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise OffAPIUnavailable(f"Open Food Facts unreachable: {exc}") from exc

    if resp.status_code == 404:
        raise ProductNotFound(f"barcode {barcode} not found (HTTP 404)")
    if resp.status_code != 200:
        # covers 429 rate-limits, 5xx, redirects — anything that isn't a
        # clean "not found" is an availability problem, never a 404 message
        # (a 429 body often carries {"status": 0}, which must NOT read as
        # "barcode not found" during a rate-limited live demo)
        raise OffAPIUnavailable(
            f"Open Food Facts returned HTTP {resp.status_code}"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise OffAPIUnavailable("Open Food Facts returned a non-JSON body") from exc

    if not isinstance(data, dict):
        raise OffAPIUnavailable("unexpected payload shape from Open Food Facts")
    if data.get("status") == 0:
        raise ProductNotFound(f"barcode {barcode} not found in Open Food Facts")
    product = data.get("product")
    if data.get("status") != 1 or not isinstance(product, dict):
        # guards against list/None/missing `product` escaping as a 500
        raise OffAPIUnavailable("unexpected payload shape from Open Food Facts")

    return strip_forbidden(dict(product))


def extract_feature_input(barcode: str, product: dict) -> dict:
    """Raw OFF product JSON -> one row in the training input space.

    Field mapping is a deliberate mirror of Stage 1's ``flatten_hit()``
    (``src/fetch_training_set.py``) so live rows look like training rows:
    ``energy-kcal_100g`` (kcal, NOT the kJ ``energy_100g``), comma-joined
    ``categories_tags``, ingredient tags joined into the pseudo-text field.
    The row is still RAW here -- ``src/clean.py::normalize_live_row`` applies
    the Stage 2 normalizations next.
    """
    nutriments = product.get("nutriments") or {}
    ingredients_tags = product.get("ingredients_tags") or []
    if ingredients_tags:
        pseudo_text = " ".join(
            t.split(":", 1)[-1].replace("-", " ") for t in ingredients_tags
        )
    else:
        # some products only ship free-text ingredients
        pseudo_text = product.get("ingredients_text") or ""

    brands = product.get("brands")
    # Train/serve parity: the search API gave Stage 1 a list-repr
    # ("['Nutella', 'Ferrero']") and clean() keeps the FIRST entry, while
    # the product API gives a plain string ("Nutella, Ferrero") that the
    # list-repr parser leaves whole -> "nutella, ferrero", which is absent
    # from the FrequencyEncoder vocab (203/19,998 training rows carry a
    # comma; 'nutella' is in the vocab, 'nutella, ferrero' is not). Take
    # the first segment here so both sources normalize identically.
    if isinstance(brands, str) and "," in brands:
        brands = brands.split(",")[0].strip()

    row = {
        "code": str(barcode),
        "product_name": product.get("product_name") or "",
        "brands": brands,
        "categories_tags": ",".join(product.get("categories_tags") or []),
        "ingredients_pseudo_text": pseudo_text,
        "additives_n": product.get("additives_n"),
        "ingredients_n": product.get("ingredients_n"),
        "unknown_ingredients_n": product.get("unknown_ingredients_n"),
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
    # same assertion pattern as flatten_hit() -- catches a mapping bug that
    # would smuggle OFF's label into the frame the model scores
    leaked = FORBIDDEN_RESPONSE_FIELDS & set(row)
    assert not leaked, f"forbidden field(s) in feature input: {leaked}"
    assert "nova_group" not in row, "Hard rule 11: nova_group must not be extracted"
    return row
