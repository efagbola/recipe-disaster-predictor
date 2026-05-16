from __future__ import annotations

import argparse
import ast
import math
import os
import re
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from clean_data import extract_number, normalize_ingredient, parse_ingredient_list

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
RAW_RECIPES_PATH = ROOT.parent / "allrecipes_all.csv"
RAW_INGREDIENTS_PATH = ROOT.parent / "recipes_ingredients_long.csv"
REPORT_DIR = ROOT / "report" / "final_model_v3"


INGREDIENT_GROUP_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "meat_poultry": (
        r"chicken", r"turkey", r"beef", r"pork", r"bacon", r"ham", r"sausage", r"lamb", r"veal",
        r"prosciutto", r"pepperoni", r"chorizo", r"steak",
    ),
    "seafood": (
        r"salmon", r"tuna", r"shrimp", r"fish", r"cod", r"crab", r"lobster", r"clam", r"oyster",
        r"anchovy", r"sardine", r"trout", r"tilapia", r"scallop",
    ),
    "dairy": (
        r"milk", r"cream", r"cheese", r"butter", r"yogurt", r"sour cream", r"mozzarella",
        r"parmesan", r"cheddar", r"feta", r"ricotta",
    ),
    "eggs": (r"\begg\b", r"\beggs\b", r"egg white", r"egg yolk"),
    "vegetables": (
        r"onion", r"garlic", r"tomato", r"pepper", r"potato", r"carrot", r"celery", r"spinach",
        r"broccoli", r"mushroom", r"zucchini", r"lettuce", r"corn", r"cabbage", r"cucumber",
        r"squash", r"bean", r"pea",
    ),
    "fruit": (
        r"apple", r"banana", r"berry", r"orange", r"lemon", r"lime", r"pineapple", r"mango",
        r"peach", r"pear", r"raisin", r"cranberry", r"avocado", r"coconut",
    ),
    "grains_starches": (
        r"flour", r"rice", r"pasta", r"noodle", r"bread", r"tortilla", r"oat", r"quinoa",
        r"cornmeal", r"cereal", r"breadcrumb", r"cracker", r"dough", r"pizza crust",
    ),
    "legumes_soy": (r"bean", r"chickpea", r"lentil", r"pea", r"tofu", r"soy"),
    "nuts_seeds": (
        r"almond", r"walnut", r"pecan", r"peanut", r"cashew", r"pistachio", r"sesame", r"chia",
        r"flax", r"sunflower seed", r"pumpkin seed",
    ),
    "herbs_spices_seasonings": (
        r"basil", r"oregano", r"parsley", r"cilantro", r"thyme", r"rosemary", r"cumin",
        r"paprika", r"pepper", r"cinnamon", r"nutmeg", r"ginger", r"turmeric", r"coriander",
        r"dill", r"seasoning", r"salt",
    ),
    "oils_fats": (r"oil", r"olive oil", r"vegetable oil", r"canola", r"shortening", r"lard", r"mayonnaise"),
    "sweeteners_chocolate": (r"sugar", r"honey", r"syrup", r"molasses", r"jam", r"jelly", r"chocolate", r"cocoa"),
    "sauces_condiments": (
        r"sauce", r"soy sauce", r"ketchup", r"mustard", r"vinegar", r"dressing", r"salsa",
        r"hot sauce", r"worcestershire", r"barbecue", r"bbq", r"miso", r"pesto",
    ),
    "baking_additives": (r"baking powder", r"baking soda", r"yeast", r"vanilla", r"cornstarch", r"gelatin", r"pectin"),
}

PREP_METHOD_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "prep_bake": (r"\bbake\b", r"\broast\b"),
    "prep_fry": (r"\bfry\b", r"air fry", r"\bsaute\b", r"\bsear\b"),
    "prep_boil": (r"\bboil\b", r"\bsimmer\b", r"\bpoach\b"),
    "prep_grill": (r"\bgrill\b",),
    "prep_mix": (r"\bmix\b", r"\bstir\b", r"\bwhisk\b", r"\bcombine\b", r"\bfold\b"),
    "prep_chill": (r"\bchill\b", r"\brefrigerate\b", r"\bfreeze\b"),
}

CURATED_QUANTITY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "qty_flour": (r"flour",),
    "qty_sugar": (r"sugar", r"honey", r"syrup", r"molasses"),
    "qty_salt": (r"\bsalt\b",),
    "qty_butter": (r"butter",),
    "qty_oil": (r"oil",),
    "qty_garlic": (r"garlic",),
    "qty_onion": (r"onion",),
    "qty_cheese": (r"cheese",),
    "qty_milk": (r"milk", r"cream", r"yogurt"),
    "qty_egg": (r"\begg\b", r"egg white", r"egg yolk"),
}

QUANTITY_BASE_FEATURES = [
    "qty_known_amount_items",
    "qty_known_amount_share",
    "qty_total_normalized",
    "qty_mean_normalized",
    "qty_max_normalized",
    "qty_std_normalized",
    "qty_unique_unit_count",
    "qty_volume_tsp_total",
    "qty_weight_oz_total",
    "qty_count_total",
    "qty_other_total",
    "qty_cup_count",
    "qty_tbsp_count",
    "qty_tsp_count",
    "qty_oz_count",
    "qty_lb_count",
]

CURATED_QUANTITY_FEATURES = list(CURATED_QUANTITY_PATTERNS.keys()) + [
    "ratio_sugar_to_flour",
    "ratio_salt_to_flour",
    "ratio_butter_to_flour",
    "ratio_oil_to_flour",
]

QUANTITY_INTERACTION_FEATURES = [
    "inter_total_qty_x_ingredient_count",
    "inter_total_qty_x_prep_bake",
    "inter_total_qty_x_prep_fry",
    "inter_total_qty_x_prep_boil",
    "inter_total_qty_x_prep_mix",
    "inter_qty_sugar_x_prep_bake",
    "inter_qty_oil_x_prep_fry",
    "inter_qty_cheese_x_prep_bake",
]

FEATURE_SET_SPECS: Dict[str, Dict[str, bool]] = {
    "content_only_core": {
        "include_reliability": False,
        "include_quantity": False,
        "include_interactions": False,
    },
    "content_plus_reliability": {
        "include_reliability": True,
        "include_quantity": False,
        "include_interactions": False,
    },
    "content_plus_quantity": {
        "include_reliability": False,
        "include_quantity": True,
        "include_interactions": False,
    },
    "content_quantity_plus_reliability": {
        "include_reliability": True,
        "include_quantity": True,
        "include_interactions": False,
    },
    "content_quantity_interactions": {
        "include_reliability": False,
        "include_quantity": True,
        "include_interactions": True,
    },
    "full_content_quantity_interactions_plus_reliability": {
        "include_reliability": True,
        "include_quantity": True,
        "include_interactions": True,
    },
}


@dataclass
class RunConfig:
    mode: str
    seeds: List[int]
    test_size: float
    text_max_features: int
    text_svd_components: int
    top_ingredients: int
    top_categories: int
    top_cuisines: int
    quick_name: str
    category_min_support: int = 30
    threshold_grid: Tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.05, 1.00, 0.05))
    classification_targets: Tuple[str, ...] = (
        "low_rating_bottom30_raw",
        "low_rating_bottom20_raw",
        "low_rating_bottom30_bayes_global_m25",
        "low_rating_bottom30_bayes_category_m25",
        "low_rating_bottom30_category_relative",
    )
    regression_targets: Tuple[str, ...] = (
        "continuous_base_rating",
        "continuous_bayes_global_m25",
        "continuous_category_residual",
    )
    rating_count_filters: Tuple[int, ...] = (0, 25)


def get_run_config(mode: str) -> RunConfig:
    if mode == "full":
        return RunConfig(
            mode="full",
            seeds=[11, 22, 33, 44, 55],
            test_size=0.20,
            text_max_features=2000,
            text_svd_components=120,
            top_ingredients=180,
            top_categories=25,
            top_cuisines=20,
            quick_name="full",
        )
    return RunConfig(
        mode="quick",
        seeds=[42],
        test_size=0.20,
        text_max_features=500,
        text_svd_components=35,
        top_ingredients=90,
        top_categories=18,
        top_cuisines=12,
        quick_name="quick",
    )


def ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_literal_list(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [text]


def parse_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    iso = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", text.upper())
    if iso:
        days = float(iso.group(1) or 0)
        hours = float(iso.group(2) or 0)
        minutes = float(iso.group(3) or 0)
        return 1440 * days + 60 * hours + minutes
    hours_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs|h)\b", text.lower())
    minutes_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minute|minutes|min|mins|m)\b", text.lower())
    total = 0.0
    if hours_match:
        total += 60 * float(hours_match.group(1))
    if minutes_match:
        total += float(minutes_match.group(1))
    return total if total else float(extract_number(text) or np.nan)


def safe_divide(a: pd.Series | np.ndarray | float, b: pd.Series | np.ndarray | float) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full_like(a_arr, np.nan, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr) & (b_arr != 0)
    out[mask] = a_arr[mask] / b_arr[mask]
    return out


def flatten_text(parts: Iterable[Any]) -> str:
    joined: List[str] = []
    for part in parts:
        if isinstance(part, list):
            joined.extend(str(x).strip() for x in part if str(x).strip())
        elif pd.notna(part):
            text = str(part).strip()
            if text:
                joined.append(text)
    return " ".join(joined)


def clean_label(text: Any) -> str:
    if pd.isna(text):
        return ""
    value = str(text).strip().lower()
    return re.sub(r"\s+", " ", value)


def parse_fraction_text(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    text = text.replace("-", " ")
    parts = [part for part in text.split() if part]
    total = 0.0
    found = False
    for part in parts:
        if "/" in part:
            nums = part.split("/")
            if len(nums) == 2 and nums[0].isdigit() and nums[1].isdigit() and int(nums[1]) != 0:
                total += int(nums[0]) / int(nums[1])
                found = True
        else:
            try:
                total += float(part)
                found = True
            except ValueError:
                return None
    return total if found else None


def parse_quantity_value(raw_ingredient: Any) -> Tuple[float, str]:
    if pd.isna(raw_ingredient):
        return np.nan, "none"
    text = clean_label(raw_ingredient)
    if not text:
        return np.nan, "none"
    match = re.match(r"^(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)\s*(.*)$", text)
    amount = np.nan
    remainder = text
    if match:
        parsed = parse_fraction_text(match.group(1))
        if parsed is not None:
            amount = float(parsed)
            remainder = match.group(2).strip()
    unit_aliases = {
        "teaspoons": "tsp", "teaspoon": "tsp", "tsp": "tsp", "tsps": "tsp",
        "tablespoons": "tbsp", "tablespoon": "tbsp", "tbsp": "tbsp", "tbsps": "tbsp",
        "cups": "cup", "cup": "cup",
        "ounces": "oz", "ounce": "oz", "oz": "oz",
        "pounds": "lb", "pound": "lb", "lbs": "lb", "lb": "lb",
        "grams": "g", "gram": "g", "g": "g",
        "kilograms": "kg", "kilogram": "kg", "kg": "kg",
        "milliliters": "ml", "milliliter": "ml", "ml": "ml",
        "liters": "l", "liter": "l", "l": "l",
        "cloves": "clove", "clove": "clove",
        "slices": "slice", "slice": "slice",
        "cans": "can", "can": "can",
        "packages": "package", "package": "package", "packets": "package", "packet": "package",
        "sticks": "stick", "stick": "stick",
        "pints": "pint", "pint": "pint",
        "quarts": "quart", "quart": "quart",
    }
    unit = "none"
    if remainder:
        first = remainder.split()[0]
        unit = unit_aliases.get(first, "none")
    if not np.isfinite(amount):
        return np.nan, unit
    conversions = {
        "tsp": ("volume", 1.0),
        "tbsp": ("volume", 3.0),
        "cup": ("volume", 48.0),
        "ml": ("volume", 0.202884),
        "l": ("volume", 202.884),
        "pint": ("volume", 96.0),
        "quart": ("volume", 192.0),
        "oz": ("weight", 1.0),
        "lb": ("weight", 16.0),
        "g": ("weight", 0.035274),
        "kg": ("weight", 35.274),
        "clove": ("count", 1.0),
        "slice": ("count", 1.0),
        "can": ("count", 1.0),
        "package": ("count", 1.0),
        "stick": ("count", 1.0),
        "none": ("other", 1.0),
    }
    family, multiplier = conversions.get(unit, ("other", 1.0))
    return amount * multiplier, family


def detect_true_rating_source(raw_df: pd.DataFrame) -> Dict[str, Any]:
    rating_columns = [column for column in raw_df.columns if any(token in column.lower() for token in ("star", "rating", "review"))]
    star_count_columns = [column for column in rating_columns if re.search(r"(one|two|three|four|five|1|2|3|4|5).*star", column.lower())]
    if star_count_columns:
        return {
            "true_rating_available": True,
            "rating_source_used": "true_rating_reconstructed",
            "reconstructed_column": "true_rating",
            "notes": f"Detected potential star-count columns: {', '.join(star_count_columns)}",
        }
    return {
        "true_rating_available": False,
        "rating_source_used": "rating_value",
        "reconstructed_column": None,
        "notes": "No individual star-count distribution columns were detected in the raw data; exact true-rating reconstruction was not possible.",
    }


def pick_first(values: List[str], fallback: str = "unknown") -> str:
    cleaned = [clean_label(value) for value in values if clean_label(value)]
    return cleaned[0] if cleaned else fallback


def add_ingredient_group_features(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for recipe_id, frame in long_df.groupby("recipe_id", sort=False):
        recipe_row: Dict[str, Any] = {"recipe_id": recipe_id}
        text_values = frame["ingredient_canonical"].fillna("").astype(str).str.lower()
        for group_name, patterns in INGREDIENT_GROUP_PATTERNS.items():
            regex = "|".join(f"(?:{pattern})" for pattern in patterns)
            match_mask = text_values.str.contains(regex, regex=True, na=False)
            recipe_row[f"inggrp_{group_name}_count"] = int(match_mask.sum())
            recipe_row[f"inggrp_{group_name}_present"] = int(match_mask.any())
        rows.append(recipe_row)
    return pd.DataFrame(rows)


def add_quantity_features(long_df: pd.DataFrame) -> pd.DataFrame:
    prepared = long_df.copy()
    parsed = prepared["ingredient_raw"].apply(parse_quantity_value)
    prepared["quantity_normalized"] = [item[0] for item in parsed]
    prepared["quantity_family"] = [item[1] for item in parsed]

    for feature_name, patterns in CURATED_QUANTITY_PATTERNS.items():
        regex = "|".join(f"(?:{pattern})" for pattern in patterns)
        mask = prepared["ingredient_canonical"].fillna("").astype(str).str.contains(regex, regex=True, na=False)
        prepared[feature_name] = np.where(mask, prepared["quantity_normalized"], 0.0)

    rows: List[Dict[str, Any]] = []
    for recipe_id, frame in prepared.groupby("recipe_id", sort=False):
        normalized = pd.to_numeric(frame["quantity_normalized"], errors="coerce")
        valid = normalized[np.isfinite(normalized)]
        unit_counts = frame["quantity_family"].value_counts(dropna=False)
        recipe_row: Dict[str, Any] = {
            "recipe_id": recipe_id,
            "qty_known_amount_items": int(valid.shape[0]),
            "qty_known_amount_share": float(valid.shape[0] / max(len(frame), 1)),
            "qty_total_normalized": float(valid.sum()) if len(valid) else 0.0,
            "qty_mean_normalized": float(valid.mean()) if len(valid) else 0.0,
            "qty_max_normalized": float(valid.max()) if len(valid) else 0.0,
            "qty_std_normalized": float(valid.std(ddof=0)) if len(valid) else 0.0,
            "qty_unique_unit_count": float(frame["quantity_family"].replace("none", np.nan).nunique(dropna=True)),
            "qty_volume_tsp_total": float(normalized[frame["quantity_family"] == "volume"].sum(skipna=True)),
            "qty_weight_oz_total": float(normalized[frame["quantity_family"] == "weight"].sum(skipna=True)),
            "qty_count_total": float(normalized[frame["quantity_family"] == "count"].sum(skipna=True)),
            "qty_other_total": float(normalized[frame["quantity_family"] == "other"].sum(skipna=True)),
            "qty_cup_count": float(frame["ingredient_raw"].fillna("").astype(str).str.contains(r"\bcup", case=False, regex=True).sum()),
            "qty_tbsp_count": float(frame["ingredient_raw"].fillna("").astype(str).str.contains(r"\btablespoon|\btbsp", case=False, regex=True).sum()),
            "qty_tsp_count": float(frame["ingredient_raw"].fillna("").astype(str).str.contains(r"\bteaspoon|\btsp", case=False, regex=True).sum()),
            "qty_oz_count": float(frame["ingredient_raw"].fillna("").astype(str).str.contains(r"\boz\b|ounce", case=False, regex=True).sum()),
            "qty_lb_count": float(frame["ingredient_raw"].fillna("").astype(str).str.contains(r"\blb\b|pound", case=False, regex=True).sum()),
        }
        for feature_name in CURATED_QUANTITY_PATTERNS:
            recipe_row[feature_name] = float(pd.to_numeric(frame[feature_name], errors="coerce").sum(skipna=True))
        rows.append(recipe_row)

    quantity_df = pd.DataFrame(rows)
    quantity_df["ratio_sugar_to_flour"] = safe_divide(quantity_df["qty_sugar"], quantity_df["qty_flour"])
    quantity_df["ratio_salt_to_flour"] = safe_divide(quantity_df["qty_salt"], quantity_df["qty_flour"])
    quantity_df["ratio_butter_to_flour"] = safe_divide(quantity_df["qty_butter"], quantity_df["qty_flour"])
    quantity_df["ratio_oil_to_flour"] = safe_divide(quantity_df["qty_oil"], quantity_df["qty_flour"])
    return quantity_df


def build_master_dataset(config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    raw_df = pd.read_csv(RAW_RECIPES_PATH).reset_index(drop=False).rename(columns={"index": "recipe_id"})
    long_df = pd.read_csv(RAW_INGREDIENTS_PATH)

    rating_info = detect_true_rating_source(raw_df)
    raw_df["rating_value_num"] = pd.to_numeric(raw_df["rating_value"], errors="coerce")
    raw_df["rating_count_num"] = pd.to_numeric(raw_df["rating_count"], errors="coerce")
    raw_df["review_count_num"] = pd.to_numeric(raw_df["review_count"], errors="coerce")
    raw_df["base_rating"] = raw_df["rating_value_num"]

    raw_df["category_list"] = raw_df["category"].apply(safe_literal_list)
    raw_df["cuisine_list"] = raw_df["cuisine"].apply(safe_literal_list)
    raw_df["ingredients_list"] = raw_df["ingredients"].apply(parse_ingredient_list)
    raw_df["directions_list"] = raw_df["directions"].apply(safe_literal_list)

    raw_df["primary_category"] = raw_df.apply(
        lambda row: pick_first(row["category_list"], clean_label(row.get("source_category")) or "unknown"),
        axis=1,
    )
    raw_df["primary_cuisine"] = raw_df["cuisine_list"].apply(lambda values: pick_first(values, "unknown"))
    raw_df["source_category_clean"] = raw_df["source_category"].apply(lambda value: clean_label(value) or "unknown")

    for source, target in (
        ("nutrition_calories", "calories"),
        ("nutrition_carbs", "carbs"),
        ("nutrition_fat", "fat"),
        ("nutrition_protein", "protein"),
    ):
        raw_df[target] = raw_df[source].apply(extract_number)
        raw_df[f"log_{target}"] = np.log1p(pd.to_numeric(raw_df[target], errors="coerce").clip(lower=0))

    raw_df["prep_minutes"] = raw_df["prep_time"].apply(parse_minutes)
    raw_df["cook_minutes"] = raw_df["cook_time"].apply(parse_minutes)
    raw_df["total_minutes"] = raw_df["total_time"].apply(parse_minutes)
    raw_df["yield_num"] = raw_df["yield"].apply(extract_number)

    raw_df["ingredient_count_num"] = raw_df["ingredients_list"].apply(len).astype(float)
    raw_df["direction_step_count"] = raw_df["directions_list"].apply(len).astype(float)
    raw_df["title_length"] = raw_df["title"].fillna("").astype(str).str.len().astype(float)
    raw_df["description_length"] = raw_df["description"].fillna("").astype(str).str.len().astype(float)
    raw_df["text_word_count"] = (
        raw_df["title"].fillna("").astype(str)
        + " "
        + raw_df["description"].fillna("").astype(str)
        + " "
        + raw_df["directions_list"].apply(lambda values: " ".join(values))
    ).str.split().str.len().astype(float)
    raw_df["ingredients_per_step"] = safe_divide(raw_df["ingredient_count_num"], raw_df["direction_step_count"])
    raw_df["words_per_step"] = safe_divide(raw_df["text_word_count"], raw_df["direction_step_count"])
    raw_df["active_time_share"] = safe_divide(raw_df["prep_minutes"], raw_df["total_minutes"])
    raw_df["protein_per_calorie"] = safe_divide(raw_df["protein"], raw_df["calories"])
    raw_df["fat_per_calorie"] = safe_divide(raw_df["fat"], raw_df["calories"])
    raw_df["carbs_per_calorie"] = safe_divide(raw_df["carbs"], raw_df["calories"])
    raw_df["calories_per_ingredient"] = safe_divide(raw_df["calories"], raw_df["ingredient_count_num"])
    raw_df["log_rating_count"] = np.log1p(raw_df["rating_count_num"].clip(lower=0))
    raw_df["log_review_count"] = np.log1p(raw_df["review_count_num"].fillna(0).clip(lower=0))

    combined_text_parts = (
        raw_df["title"].fillna("").astype(str)
        + " "
        + raw_df["description"].fillna("").astype(str)
        + " "
        + raw_df["directions_list"].apply(lambda values: " ".join(values))
    )
    raw_df["combined_text"] = combined_text_parts.str.replace(r"\s+", " ", regex=True).str.strip()

    long_df["ingredient_canonical"] = long_df["ingredient_canonical"].fillna("").astype(str).str.lower().apply(normalize_ingredient)
    long_df["ingredient_canonical"] = long_df["ingredient_canonical"].fillna("").astype(str)
    ingredient_lists = (
        long_df.groupby("recipe_id", as_index=False)["ingredient_canonical"]
        .agg(lambda values: sorted({value for value in values if value}))
        .rename(columns={"ingredient_canonical": "ingredient_canonical_list"})
    )

    group_features = add_ingredient_group_features(long_df)
    quantity_features = add_quantity_features(long_df)

    df = raw_df.merge(ingredient_lists, on="recipe_id", how="left")
    df = df.merge(group_features, on="recipe_id", how="left")
    df = df.merge(quantity_features, on="recipe_id", how="left")

    df["ingredient_canonical_list"] = df["ingredient_canonical_list"].apply(lambda value: value if isinstance(value, list) else [])

    for prep_feature, patterns in PREP_METHOD_PATTERNS.items():
        regex = "|".join(f"(?:{pattern})" for pattern in patterns)
        df[prep_feature] = df["combined_text"].fillna("").str.lower().str.count(regex).astype(float)

    for feature in QUANTITY_BASE_FEATURES + CURATED_QUANTITY_FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan
    for feature in QUANTITY_INTERACTION_FEATURES:
        df[feature] = 0.0

    df["inter_total_qty_x_ingredient_count"] = df["qty_total_normalized"].fillna(0) * df["ingredient_count_num"].fillna(0)
    df["inter_total_qty_x_prep_bake"] = df["qty_total_normalized"].fillna(0) * df["prep_bake"].fillna(0)
    df["inter_total_qty_x_prep_fry"] = df["qty_total_normalized"].fillna(0) * df["prep_fry"].fillna(0)
    df["inter_total_qty_x_prep_boil"] = df["qty_total_normalized"].fillna(0) * df["prep_boil"].fillna(0)
    df["inter_total_qty_x_prep_mix"] = df["qty_total_normalized"].fillna(0) * df["prep_mix"].fillna(0)
    df["inter_qty_sugar_x_prep_bake"] = df["qty_sugar"].fillna(0) * (df["prep_bake"].fillna(0) > 0).astype(float)
    df["inter_qty_oil_x_prep_fry"] = df["qty_oil"].fillna(0) * (df["prep_fry"].fillna(0) > 0).astype(float)
    df["inter_qty_cheese_x_prep_bake"] = df["qty_cheese"].fillna(0) * (df["prep_bake"].fillna(0) > 0).astype(float)

    return df, long_df, rating_info


def add_target_columns(df: pd.DataFrame, config: RunConfig, rating_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_df = df.copy()
    rating = target_df["base_rating"]
    counts = target_df["rating_count_num"].fillna(0)
    valid_mask = rating.notna()
    global_mean = float(rating[valid_mask].mean())

    category_counts = target_df.loc[valid_mask, "primary_category"].value_counts()
    category_means = target_df.loc[valid_mask].groupby("primary_category")["base_rating"].mean()
    category_supported = category_counts[category_counts >= config.category_min_support].index
    target_df["category_supported"] = target_df["primary_category"].isin(category_supported)
    target_df["category_prior_mean"] = target_df["primary_category"].map(category_means)
    target_df.loc[~target_df["category_supported"], "category_prior_mean"] = np.nan
    target_df["category_prior_mean"] = target_df["category_prior_mean"].fillna(global_mean)

    target_df["continuous_bayes_global_m25"] = ((counts / (counts + 25)) * rating) + ((25 / (counts + 25)) * global_mean)
    target_df["continuous_bayes_global_m50"] = ((counts / (counts + 50)) * rating) + ((50 / (counts + 50)) * global_mean)
    target_df["continuous_bayes_category_m25"] = ((counts / (counts + 25)) * rating) + ((25 / (counts + 25)) * target_df["category_prior_mean"])
    target_df["continuous_bayes_category_m50"] = ((counts / (counts + 50)) * rating) + ((50 / (counts + 50)) * target_df["category_prior_mean"])
    target_df["continuous_base_rating"] = target_df["base_rating"]
    target_df["continuous_category_residual"] = rating - target_df["category_prior_mean"]

    valid_rating = target_df.loc[valid_mask, "base_rating"]
    target_df["low_rating_bottom30_raw"] = (target_df["base_rating"] <= valid_rating.quantile(0.30)).astype(float)
    target_df["low_rating_bottom20_raw"] = (target_df["base_rating"] <= valid_rating.quantile(0.20)).astype(float)

    valid_bayes_global = target_df.loc[target_df["continuous_bayes_global_m25"].notna(), "continuous_bayes_global_m25"]
    target_df["low_rating_bottom30_bayes_global_m25"] = (
        target_df["continuous_bayes_global_m25"] <= valid_bayes_global.quantile(0.30)
    ).astype(float)

    valid_bayes_category = target_df.loc[target_df["continuous_bayes_category_m25"].notna(), "continuous_bayes_category_m25"]
    target_df["low_rating_bottom30_bayes_category_m25"] = (
        target_df["continuous_bayes_category_m25"] <= valid_bayes_category.quantile(0.30)
    ).astype(float)

    raw_rank = target_df.groupby("primary_category")["base_rating"].rank(pct=True, method="average")
    supported_mask = target_df["category_supported"] & target_df["base_rating"].notna()
    global_pct_rank = target_df["base_rating"].rank(pct=True, method="average")
    category_relative_rank = raw_rank.where(supported_mask, global_pct_rank)
    target_df["low_rating_bottom30_category_relative"] = (category_relative_rank <= 0.30).astype(float)

    summary_rows: List[Dict[str, Any]] = []
    target_notes = {
        "low_rating_bottom30_raw": "Global bottom 30 percent threshold using the best available base rating.",
        "low_rating_bottom20_raw": "More selective raw threshold using the global bottom 20 percent of the base rating.",
        "low_rating_bottom30_bayes_global_m25": "Bottom 30 percent after global Bayesian shrinkage with m=25.",
        "low_rating_bottom30_bayes_category_m25": "Bottom 30 percent after category-aware Bayesian shrinkage with m=25; sparse categories fall back to the global prior.",
        "low_rating_bottom30_category_relative": "Bottom 30 percent within category when category support >= 30, otherwise global percentile fallback.",
    }
    target_meta = {
        "low_rating_bottom30_raw": ("raw_percentile", False, False, None),
        "low_rating_bottom20_raw": ("raw_percentile", False, False, None),
        "low_rating_bottom30_bayes_global_m25": ("bayesian_global", True, False, 25),
        "low_rating_bottom30_bayes_category_m25": ("bayesian_category", True, True, 25),
        "low_rating_bottom30_category_relative": ("category_relative", False, True, None),
    }
    for target_name in config.classification_targets:
        for min_rating_count_filter in config.rating_count_filters:
            subset = target_df[target_df["base_rating"].notna()].copy()
            if min_rating_count_filter > 0:
                subset = subset[subset["rating_count_num"] >= min_rating_count_filter].copy()
            positive_share = float(subset[target_name].mean()) if len(subset) else np.nan
            target_type, uses_bayes, uses_category, m_value = target_meta[target_name]
            notes = target_notes[target_name]
            if target_name in ("low_rating_bottom30_bayes_category_m25", "low_rating_bottom30_category_relative"):
                fallback_n = int((~subset["category_supported"]).sum())
                notes = f"{notes} Fallback-to-global rows: {fallback_n}."
            summary_rows.append(
                {
                    "target_name": target_name,
                    "target_type": target_type,
                    "rating_source_used": rating_info["rating_source_used"],
                    "uses_bayesian_adjustment": uses_bayes,
                    "uses_category_adjustment": uses_category,
                    "m_value": m_value,
                    "min_rating_count_filter": min_rating_count_filter,
                    "n_obs": int(len(subset)),
                    "positive_class_share": positive_share,
                    "notes": notes,
                }
            )
    target_summary = pd.DataFrame(summary_rows)
    return target_df, target_summary


def get_classification_models(mode: str) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=0)),
            ]
        ),
        "LogisticRegressionBalanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)),
            ]
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=6,
            max_iter=110 if mode == "quick" else 220,
            min_samples_leaf=20,
            random_state=0,
        ),
    }
    if mode == "full":
        models["ExtraTreesClassifier"] = ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=0,
        )
        try:
            from xgboost import XGBClassifier

            models["XGBoostClassifier"] = XGBClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=0,
            )
        except Exception:
            pass
    return models


def get_regression_models(mode: str) -> Dict[str, Any]:
    return {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=120 if mode == "quick" else 250,
            min_samples_leaf=20,
            random_state=0,
        ),
    }


class DenseFeatureBuilder:
    def __init__(self, config: RunConfig, feature_set: str):
        self.config = config
        self.feature_set = feature_set
        spec = FEATURE_SET_SPECS[feature_set]
        self.include_reliability = spec["include_reliability"]
        self.include_quantity = spec["include_quantity"]
        self.include_interactions = spec["include_interactions"]
        self.numeric_features_: List[str] = []
        self.ingredient_labels_: List[str] = []
        self.category_encoder_: Optional[MultiLabelBinarizer] = None
        self.cuisine_encoder_: Optional[MultiLabelBinarizer] = None
        self.source_encoder_: Optional[MultiLabelBinarizer] = None
        self.ingredient_encoder_: Optional[MultiLabelBinarizer] = None
        self.text_vectorizer_: Optional[TfidfVectorizer] = None
        self.text_reducer_: Optional[TruncatedSVD] = None
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        self.feature_names_: List[str] = []

    def base_numeric_features(self, df: pd.DataFrame) -> List[str]:
        base = [
            "calories", "carbs", "fat", "protein",
            "log_calories", "log_carbs", "log_fat", "log_protein",
            "prep_minutes", "cook_minutes", "total_minutes",
            "yield_num", "ingredient_count_num", "direction_step_count",
            "title_length", "description_length", "text_word_count",
            "ingredients_per_step", "words_per_step", "active_time_share",
            "protein_per_calorie", "fat_per_calorie", "carbs_per_calorie",
            "calories_per_ingredient",
        ]
        base.extend([column for column in df.columns if column.startswith("inggrp_")])
        base.extend(list(PREP_METHOD_PATTERNS.keys()))
        if self.include_quantity:
            base.extend(QUANTITY_BASE_FEATURES)
            base.extend(CURATED_QUANTITY_FEATURES)
        if self.include_interactions:
            base.extend(QUANTITY_INTERACTION_FEATURES)
        if self.include_reliability:
            base.extend(["log_rating_count", "log_review_count"])
        return [feature for feature in base if feature in df.columns]

    def fit(self, train_df: pd.DataFrame) -> "DenseFeatureBuilder":
        self.numeric_features_ = self.base_numeric_features(train_df)
        self.numeric_imputer_.fit(train_df[self.numeric_features_])

        category_vocab = train_df["primary_category"].value_counts().head(self.config.top_categories).index.tolist()
        cuisine_vocab = train_df["primary_cuisine"].value_counts().head(self.config.top_cuisines).index.tolist()
        source_vocab = train_df["source_category_clean"].value_counts().head(self.config.top_categories).index.tolist()

        self.category_encoder_ = MultiLabelBinarizer(classes=category_vocab)
        self.cuisine_encoder_ = MultiLabelBinarizer(classes=cuisine_vocab)
        self.source_encoder_ = MultiLabelBinarizer(classes=source_vocab)
        self.category_encoder_.fit([[item] for item in category_vocab])
        self.cuisine_encoder_.fit([[item] for item in cuisine_vocab])
        self.source_encoder_.fit([[item] for item in source_vocab])

        ingredient_counter = (
            train_df["ingredient_canonical_list"]
            .explode()
            .dropna()
            .astype(str)
            .str.strip()
        )
        ingredient_counter = ingredient_counter[ingredient_counter != ""]
        self.ingredient_labels_ = ingredient_counter.value_counts().head(self.config.top_ingredients).index.tolist()
        self.ingredient_encoder_ = MultiLabelBinarizer(classes=self.ingredient_labels_)
        self.ingredient_encoder_.fit([self.ingredient_labels_])

        text_series = train_df["combined_text"].fillna("")
        self.text_vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.config.text_max_features,
            min_df=3,
            strip_accents="unicode",
        )
        text_matrix = self.text_vectorizer_.fit_transform(text_series)
        n_components = min(self.config.text_svd_components, max(1, text_matrix.shape[1] - 1)) if text_matrix.shape[1] > 1 else 1
        self.text_reducer_ = TruncatedSVD(n_components=n_components, random_state=0)
        self.text_reducer_.fit(text_matrix)

        self.feature_names_ = []
        self.feature_names_.extend(self.numeric_features_)
        self.feature_names_.extend([f"category::{item}" for item in self.category_encoder_.classes_])
        self.feature_names_.extend([f"cuisine::{item}" for item in self.cuisine_encoder_.classes_])
        self.feature_names_.extend([f"source::{item}" for item in self.source_encoder_.classes_])
        self.feature_names_.extend([f"ingredient::{item}" for item in self.ingredient_encoder_.classes_])
        self.feature_names_.extend([f"text_svd_{idx + 1}" for idx in range(self.text_reducer_.n_components)])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        numeric = self.numeric_imputer_.transform(df[self.numeric_features_])
        category = self.category_encoder_.transform(df["primary_category"].fillna("unknown").apply(lambda value: [value]))
        cuisine = self.cuisine_encoder_.transform(df["primary_cuisine"].fillna("unknown").apply(lambda value: [value]))
        source = self.source_encoder_.transform(df["source_category_clean"].fillna("unknown").apply(lambda value: [value]))
        ingredients = self.ingredient_encoder_.transform(df["ingredient_canonical_list"])
        text_matrix = self.text_vectorizer_.transform(df["combined_text"].fillna(""))
        text_dense = self.text_reducer_.transform(text_matrix)
        return np.hstack(
            [
                numeric.astype(float),
                category.astype(float),
                cuisine.astype(float),
                source.astype(float),
                ingredients.astype(float),
                text_dense.astype(float),
            ]
        )

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def feature_names(self) -> List[str]:
        return list(self.feature_names_)


def choose_threshold_from_train(y_true: np.ndarray, y_prob: np.ndarray, grid: Sequence[float]) -> float:
    best_threshold = 0.50
    best_score = -1.0
    for threshold in grid:
        pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def top_k_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, top_share: float) -> Dict[str, float]:
    n_obs = len(y_true)
    if n_obs == 0:
        return {"precision": np.nan, "recall": np.nan, "lift": np.nan}
    k = max(1, int(math.ceil(top_share * n_obs)))
    order = np.argsort(-y_prob)
    top_idx = order[:k]
    flagged = y_true[top_idx]
    precision = float(flagged.mean()) if len(flagged) else np.nan
    base_rate = float(np.mean(y_true)) if n_obs else np.nan
    recall = float(flagged.sum() / max(np.sum(y_true), 1))
    lift = float(precision / base_rate) if base_rate and np.isfinite(base_rate) else np.nan
    return {"precision": precision, "recall": recall, "lift": lift}


def evaluate_classifier(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold_grid: Sequence[float],
) -> Dict[str, Any]:
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    if hasattr(fitted_model, "predict_proba"):
        train_prob = fitted_model.predict_proba(X_train)[:, 1]
        test_prob = fitted_model.predict_proba(X_test)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        train_decision = fitted_model.decision_function(X_train)
        test_decision = fitted_model.decision_function(X_test)
        train_prob = 1 / (1 + np.exp(-train_decision))
        test_prob = 1 / (1 + np.exp(-test_decision))
    else:
        train_prob = fitted_model.predict(X_train).astype(float)
        test_prob = fitted_model.predict(X_test).astype(float)

    best_threshold = choose_threshold_from_train(y_train, train_prob, threshold_grid)
    default_pred = (test_prob >= 0.50).astype(int)
    tuned_pred = (test_prob >= best_threshold).astype(int)

    metrics: Dict[str, Any] = {
        "fitted_model": fitted_model,
        "y_prob_test": test_prob,
        "best_threshold": best_threshold,
        "default_f1": f1_score(y_test, default_pred, zero_division=0),
        "tuned_f1": f1_score(y_test, tuned_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_test, test_prob) if len(np.unique(y_test)) > 1 else np.nan,
        "brier_score": brier_score_loss(y_test, test_prob) if np.all((0 <= test_prob) & (test_prob <= 1)) else np.nan,
    }
    for share, label in ((0.05, "5"), (0.10, "10"), (0.20, "20")):
        top_metrics = top_k_binary_metrics(y_test, test_prob, share)
        metrics[f"precision_at_{label}"] = top_metrics["precision"]
        metrics[f"recall_at_{label}"] = top_metrics["recall"]
        metrics[f"lift_at_{label}"] = top_metrics["lift"]
    return metrics


def evaluate_regressor(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    pred = fitted_model.predict(X_test)
    pearson = pd.Series(pred).corr(pd.Series(y_test), method="pearson")
    spearman = pd.Series(pred).corr(pd.Series(y_test), method="spearman")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(fitted_model.score(X_test, y_test)),
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "pearson": float(pearson) if pd.notna(pearson) else np.nan,
    }


def split_with_stratification(
    df: pd.DataFrame,
    target_column: str,
    seed: int,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(idx, df[target_column].astype(int)))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def summarize_results(results: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    summary = (
        results.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        for col in summary.columns.to_flat_index()
    ]
    summary = summary.rename(columns={f"{metric_cols[0]}_count": "n_runs"})
    for metric in metric_cols[1:]:
        count_col = f"{metric}_count"
        if count_col in summary.columns:
            summary = summary.drop(columns=[count_col])
    return summary


def format_metric(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def best_summary_rows(summary_df: pd.DataFrame, by_cols: List[str], score_col: str) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    ordered = summary_df.sort_values(score_col, ascending=False)
    return ordered.groupby(by_cols, dropna=False, as_index=False).head(1).reset_index(drop=True)


def save_figures(
    summary_df: pd.DataFrame,
    content_compare_df: pd.DataFrame,
    quantity_compare_df: pd.DataFrame,
    best_all_vs_filter_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 5))
    pr_target = (
        summary_df.groupby("target")["pr_auc_mean"].max().sort_values(ascending=False)
        if not summary_df.empty
        else pd.Series(dtype=float)
    )
    pr_target.plot(kind="bar", color="#3b82f6")
    plt.ylabel("Best mean PR-AUC")
    plt.title("PR-AUC by target family")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fig_pr_auc_by_target_family.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    lift_target = (
        summary_df.groupby("target")["lift_at_10_mean"].max().sort_values(ascending=False)
        if not summary_df.empty
        else pd.Series(dtype=float)
    )
    lift_target.plot(kind="bar", color="#16a34a")
    plt.ylabel("Best mean lift@10")
    plt.title("Precision@10 / lift@10 by target family")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fig_lift10_by_target_family.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    if not best_all_vs_filter_df.empty:
        plot_df = best_all_vs_filter_df.copy()
        plot_df["filter_label"] = plot_df["min_rating_count_filter"].replace({0: "all recipes", 25: "rating_count >= 25"})
        grouped = plot_df.groupby("filter_label")["pr_auc_mean"].mean().reindex(["all recipes", "rating_count >= 25"])
        grouped.plot(kind="bar", color=["#64748b", "#ef4444"])
    plt.ylabel("Mean PR-AUC of best target/model")
    plt.title("All recipes vs rating_count >= 25")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fig_all_vs_rating_count25.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    if not content_compare_df.empty:
        tmp = content_compare_df.groupby("comparison")["pr_auc_delta"].mean().sort_values(ascending=False)
        tmp.plot(kind="bar", color="#f59e0b")
    plt.ylabel("Mean PR-AUC delta")
    plt.title("Content-only vs content-plus-reliability")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fig_content_vs_reliability.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    if not quantity_compare_df.empty:
        order = (
            quantity_compare_df.groupby("feature_set")["pr_auc_mean"].max().sort_values(ascending=False).index.tolist()
        )
        plot_df = quantity_compare_df.groupby("feature_set")["pr_auc_mean"].max().reindex(order)
        plot_df.plot(kind="bar", color="#8b5cf6")
    plt.ylabel("Best mean PR-AUC")
    plt.title("Quantity/interactions ablation")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "fig_quantity_interaction_ablation.png", dpi=160)
    plt.close()


def build_focused_comparisons(class_summary: pd.DataFrame, reg_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_by_target_filter = best_summary_rows(
        class_summary,
        by_cols=["target", "min_rating_count_filter"],
        score_col="pr_auc_mean",
    )
    all_vs_filter = best_by_target_filter[
        [
            "target", "feature_set", "model", "min_rating_count_filter", "n_obs_mean", "positive_class_share_mean",
            "pr_auc_mean", "pr_auc_std", "precision_at_10_mean", "precision_at_10_std",
            "lift_at_10_mean", "lift_at_10_std", "tuned_f1_mean", "tuned_f1_std",
        ]
    ].sort_values(["target", "min_rating_count_filter"])

    raw_vs_bayes = best_by_target_filter[
        best_by_target_filter["target"].isin(
            [
                "low_rating_bottom30_raw",
                "low_rating_bottom30_bayes_global_m25",
                "low_rating_bottom30_bayes_category_m25",
            ]
        )
    ].copy()

    reliability_pairs = [
        ("content_only_core", "content_plus_reliability"),
        ("content_plus_quantity", "content_quantity_plus_reliability"),
        ("content_quantity_interactions", "full_content_quantity_interactions_plus_reliability"),
    ]
    compare_rows: List[Dict[str, Any]] = []
    for left, right in reliability_pairs:
        left_df = class_summary[class_summary["feature_set"] == left]
        right_df = class_summary[class_summary["feature_set"] == right]
        merged = left_df.merge(
            right_df,
            on=["target", "model", "min_rating_count_filter"],
            suffixes=("_left", "_right"),
        )
        for _, row in merged.iterrows():
            compare_rows.append(
                {
                    "comparison": f"{left} -> {right}",
                    "target": row["target"],
                    "model": row["model"],
                    "min_rating_count_filter": row["min_rating_count_filter"],
                    "pr_auc_left": row["pr_auc_mean_left"],
                    "pr_auc_right": row["pr_auc_mean_right"],
                    "pr_auc_delta": row["pr_auc_mean_right"] - row["pr_auc_mean_left"],
                    "precision_at_10_delta": row["precision_at_10_mean_right"] - row["precision_at_10_mean_left"],
                    "lift_at_10_delta": row["lift_at_10_mean_right"] - row["lift_at_10_mean_left"],
                }
            )
    content_vs_reliability = pd.DataFrame(compare_rows).sort_values("pr_auc_delta", ascending=False)

    quantity_sets = [
        "content_only_core",
        "content_plus_quantity",
        "content_quantity_interactions",
        "content_plus_reliability",
        "full_content_quantity_interactions_plus_reliability",
    ]
    quantity_compare = best_summary_rows(
        class_summary[class_summary["feature_set"].isin(quantity_sets)],
        by_cols=["feature_set", "min_rating_count_filter"],
        score_col="pr_auc_mean",
    )

    use_case_rows: List[Dict[str, Any]] = []
    if not class_summary.empty:
        overall_candidates = class_summary.copy().sort_values(
            ["pr_auc_mean", "precision_at_10_mean", "lift_at_10_mean"],
            ascending=False,
        )
        overall = overall_candidates.iloc[0]
        use_case_rows.append(
            {
                "use_case": "pre_publication_content_only_screening",
                **select_use_case_row(
                    class_summary[class_summary["feature_set"] == "content_only_core"],
                    "Best content-only screen that excludes rating_count and review_count.",
                    "Still a ranking model; lower precision than reliability-aware versions is expected.",
                ),
            }
        )
        use_case_rows.append(
            {
                "use_case": "published_recipe_screening_with_reliability",
                **select_use_case_row(
                    class_summary[class_summary["feature_set"].isin(["content_plus_reliability", "content_quantity_plus_reliability", "full_content_quantity_interactions_plus_reliability"])],
                    "Best published-recipe screen allowed to use reliability features.",
                    "Reliability variables can improve screening but reduce pre-publication applicability.",
                ),
            }
        )
        use_case_rows.append(
            {
                "use_case": "reliable_rating_only_screening",
                **select_use_case_row(
                    class_summary[class_summary["min_rating_count_filter"] == 25],
                    "Best configuration on the reliable-rating subset.",
                    "Coverage is reduced because recipes with rating_count < 25 are excluded.",
                ),
            }
        )
        interpretable_pool = class_summary[
            class_summary["model"].isin(["LogisticRegression", "LogisticRegressionBalanced"])
            & ~class_summary["feature_set"].str.contains("full_content_quantity_interactions_plus_reliability")
        ]
        use_case_rows.append(
            {
                "use_case": "interpretable_culinary_association",
                **select_use_case_row(
                    interpretable_pool,
                    "Most interpretable association-oriented model among the logistic screens.",
                    "Associational only; coefficient patterns are not causal effects.",
                ),
            }
        )
        use_case_rows.append(
            {
                "use_case": "continuous_rating_prediction_secondary",
                **select_regression_use_case_row(
                    reg_summary,
                    "Best secondary regression benchmark; regression remains a support analysis, not the main recommendation.",
                    "Exact rating prediction is still weak when R^2 remains low.",
                ),
            }
        )
    best_use_cases = pd.DataFrame(use_case_rows)
    return all_vs_filter, raw_vs_bayes, content_vs_reliability, quantity_compare, best_use_cases


def select_use_case_row(pool: pd.DataFrame, why: str, limitation: str) -> Dict[str, Any]:
    if pool.empty:
        return {
            "selected_target": None,
            "selected_feature_set": None,
            "selected_model": None,
            "min_rating_count_filter": None,
            "PR_AUC": np.nan,
            "precision_at_10": np.nan,
            "lift_at_10": np.nan,
            "tuned_F1": np.nan,
            "why_selected": why,
            "limitations": limitation,
        }
    row = pool.sort_values(["pr_auc_mean", "precision_at_10_mean", "lift_at_10_mean"], ascending=False).iloc[0]
    return {
        "selected_target": row["target"],
        "selected_feature_set": row["feature_set"],
        "selected_model": row["model"],
        "min_rating_count_filter": row["min_rating_count_filter"],
        "PR_AUC": row["pr_auc_mean"],
        "precision_at_10": row["precision_at_10_mean"],
        "lift_at_10": row["lift_at_10_mean"],
        "tuned_F1": row["tuned_f1_mean"],
        "why_selected": why,
        "limitations": limitation,
    }


def select_regression_use_case_row(pool: pd.DataFrame, why: str, limitation: str) -> Dict[str, Any]:
    if pool.empty:
        return {
            "selected_target": None,
            "selected_feature_set": None,
            "selected_model": None,
            "min_rating_count_filter": None,
            "PR_AUC": np.nan,
            "precision_at_10": np.nan,
            "lift_at_10": np.nan,
            "tuned_F1": np.nan,
            "why_selected": why,
            "limitations": limitation,
        }
    row = pool.sort_values(["r2_mean", "spearman_mean"], ascending=False).iloc[0]
    return {
        "selected_target": row["target"],
        "selected_feature_set": row["feature_set"],
        "selected_model": row["model"],
        "min_rating_count_filter": row["min_rating_count_filter"],
        "PR_AUC": np.nan,
        "precision_at_10": np.nan,
        "lift_at_10": np.nan,
        "tuned_F1": np.nan,
        "why_selected": why,
        "limitations": limitation,
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    rows = []
    for idx, (observed, predicted) in enumerate(zip(prob_true, prob_pred), start=1):
        rows.append({"bin": idx, "mean_predicted_probability": predicted, "observed_positive_rate": observed})
    return pd.DataFrame(rows)


def threshold_table(y_true: np.ndarray, y_prob: np.ndarray, grid: Sequence[float]) -> pd.DataFrame:
    rows = []
    for threshold in grid:
        pred = (y_prob >= threshold).astype(int)
        flagged_share = float(pred.mean())
        rows.append(
            {
                "threshold": threshold,
                "flagged_share": flagged_share,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def save_top_model_artifacts(
    df: pd.DataFrame,
    class_summary: pd.DataFrame,
    config: RunConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top3 = class_summary.sort_values(["pr_auc_mean", "precision_at_10_mean"], ascending=False).head(3).copy()
    threshold_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    best_run_context: Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = None

    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        subset = df[df["base_rating"].notna()].copy()
        if int(row["min_rating_count_filter"]) > 0:
            subset = subset[subset["rating_count_num"] >= int(row["min_rating_count_filter"])].copy()
        train_df, test_df = split_with_stratification(subset, row["target"], config.seeds[0], config.test_size)
        builder = DenseFeatureBuilder(config, row["feature_set"])
        X_train = builder.fit_transform(train_df)
        X_test = builder.transform(test_df)
        y_train = train_df[row["target"]].astype(int).to_numpy()
        y_test = test_df[row["target"]].astype(int).to_numpy()
        model = get_classification_models(config.mode)[row["model"]]
        metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test, config.threshold_grid)
        tuning = threshold_table(y_test, metrics["y_prob_test"], config.threshold_grid)
        tuning["rank"] = rank
        tuning["target"] = row["target"]
        tuning["feature_set"] = row["feature_set"]
        tuning["model"] = row["model"]
        tuning["min_rating_count_filter"] = row["min_rating_count_filter"]
        threshold_rows.extend(tuning.to_dict("records"))

        calib = calibration_table(y_test, metrics["y_prob_test"])
        calib["rank"] = rank
        calib["target"] = row["target"]
        calib["feature_set"] = row["feature_set"]
        calib["model"] = row["model"]
        calib["min_rating_count_filter"] = row["min_rating_count_filter"]
        calibration_rows.extend(calib.to_dict("records"))

        if rank == 1:
            best_run_context = (y_test, metrics["y_prob_test"], {
                "target": row["target"],
                "feature_set": row["feature_set"],
                "model": row["model"],
            })

    threshold_df = pd.DataFrame(threshold_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    threshold_df.to_csv(REPORT_DIR / "threshold_tuning_top_models.csv", index=False)
    calibration_df.to_csv(REPORT_DIR / "calibration_top_models.csv", index=False)

    if best_run_context is not None:
        y_test, y_prob, meta = best_run_context
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        lift_rows = []
        order = np.argsort(-y_prob)
        y_sorted = y_test[order]
        base_rate = y_test.mean()
        for frac in np.linspace(0.05, 1.0, 20):
            k = max(1, int(math.ceil(frac * len(y_sorted))))
            precision_at_frac = y_sorted[:k].mean()
            lift_rows.append({"fraction_flagged": frac, "lift": precision_at_frac / base_rate if base_rate else np.nan})
        lift_df = pd.DataFrame(lift_rows)
        calib = calibration_table(y_test, y_prob)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="#2563eb")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Best model precision-recall curve")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fig_precision_recall_top_model.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#dc2626")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Best model ROC curve")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fig_roc_top_model.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(calib["mean_predicted_probability"], calib["observed_positive_rate"], marker="o", color="#16a34a")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed positive rate")
        plt.title("Best model calibration")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fig_calibration_top_model.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(lift_df["fraction_flagged"], lift_df["lift"], marker="o", color="#7c3aed")
        plt.xlabel("Top fraction flagged")
        plt.ylabel("Lift")
        plt.title("Best model lift curve")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fig_lift_curve_top_model.png", dpi=160)
        plt.close()

    return threshold_df, calibration_df


def subgroup_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    if len(y_true) < 25 or len(np.unique(y_true)) < 2:
        return {
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "precision_at_10": np.nan,
            "lift_at_10": np.nan,
            "note": "Too few observations or only one class in subgroup.",
        }
    top10 = top_k_binary_metrics(y_true, y_prob, 0.10)
    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision_at_10": top10["precision"],
        "lift_at_10": top10["lift"],
        "note": "",
    }


def evaluate_subgroups(df: pd.DataFrame, class_summary: pd.DataFrame, config: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen = class_summary.sort_values(["pr_auc_mean", "precision_at_10_mean"], ascending=False).head(3)
    subgroup_rows_count: List[Dict[str, Any]] = []
    subgroup_rows_category: List[Dict[str, Any]] = []
    subgroup_rows_cuisine: List[Dict[str, Any]] = []

    for _, row in chosen.iterrows():
        subset = df[df["base_rating"].notna()].copy()
        if int(row["min_rating_count_filter"]) > 0:
            subset = subset[subset["rating_count_num"] >= int(row["min_rating_count_filter"])].copy()
        train_df, test_df = split_with_stratification(subset, row["target"], config.seeds[0], config.test_size)
        builder = DenseFeatureBuilder(config, row["feature_set"])
        X_train = builder.fit_transform(train_df)
        X_test = builder.transform(test_df)
        y_train = train_df[row["target"]].astype(int).to_numpy()
        y_test = test_df[row["target"]].astype(int).to_numpy()
        model = get_classification_models(config.mode)[row["model"]]
        metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test, config.threshold_grid)
        prob = metrics["y_prob_test"]

        tmp = test_df.copy()
        tmp["y_true"] = y_test
        tmp["y_prob"] = prob
        tmp["rating_count_bucket"] = pd.cut(
            tmp["rating_count_num"].fillna(0),
            bins=[-np.inf, 4, 24, 99, np.inf],
            labels=["1-4", "5-24", "25-99", "100+"],
        )

        for bucket, frame in tmp.groupby("rating_count_bucket", dropna=False):
            result = subgroup_metrics(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())
            subgroup_rows_count.append(
                {
                    "target": row["target"],
                    "feature_set": row["feature_set"],
                    "model": row["model"],
                    "subgroup": str(bucket),
                    "n_obs": int(len(frame)),
                    "positive_class_share": float(frame["y_true"].mean()) if len(frame) else np.nan,
                    **result,
                }
            )

        category_counts = tmp["primary_category"].value_counts()
        keep_categories = category_counts[category_counts >= 50].index.tolist()
        for category, frame in tmp[tmp["primary_category"].isin(keep_categories)].groupby("primary_category"):
            result = subgroup_metrics(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())
            subgroup_rows_category.append(
                {
                    "target": row["target"],
                    "feature_set": row["feature_set"],
                    "model": row["model"],
                    "subgroup": category,
                    "n_obs": int(len(frame)),
                    "positive_class_share": float(frame["y_true"].mean()) if len(frame) else np.nan,
                    **result,
                }
            )

        cuisine_counts = tmp["primary_cuisine"].value_counts()
        keep_cuisines = cuisine_counts[cuisine_counts >= 50].index.tolist()
        for cuisine, frame in tmp[tmp["primary_cuisine"].isin(keep_cuisines)].groupby("primary_cuisine"):
            result = subgroup_metrics(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())
            subgroup_rows_cuisine.append(
                {
                    "target": row["target"],
                    "feature_set": row["feature_set"],
                    "model": row["model"],
                    "subgroup": cuisine,
                    "n_obs": int(len(frame)),
                    "positive_class_share": float(frame["y_true"].mean()) if len(frame) else np.nan,
                    **result,
                }
            )

    count_df = pd.DataFrame(subgroup_rows_count)
    category_df = pd.DataFrame(subgroup_rows_category)
    cuisine_df = pd.DataFrame(subgroup_rows_cuisine)
    count_df.to_csv(REPORT_DIR / "subgroup_rating_count_bucket.csv", index=False)
    category_df.to_csv(REPORT_DIR / "subgroup_category.csv", index=False)
    cuisine_df.to_csv(REPORT_DIR / "subgroup_cuisine.csv", index=False)
    return count_df, category_df, cuisine_df


def build_markdown_reports(
    class_summary: pd.DataFrame,
    reg_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    all_vs_filter: pd.DataFrame,
    raw_vs_bayes: pd.DataFrame,
    content_vs_reliability: pd.DataFrame,
    quantity_compare: pd.DataFrame,
    best_use_cases: pd.DataFrame,
    threshold_df: pd.DataFrame,
    rating_info: Dict[str, Any],
    config: RunConfig,
) -> None:
    best_overall = class_summary.sort_values(["pr_auc_mean", "precision_at_10_mean"], ascending=False).head(1)
    best_content_only = class_summary[class_summary["feature_set"] == "content_only_core"].sort_values(
        ["pr_auc_mean", "precision_at_10_mean"], ascending=False
    ).head(1)
    best_reliable = class_summary[class_summary["min_rating_count_filter"] == 25].sort_values(
        ["pr_auc_mean", "precision_at_10_mean"], ascending=False
    ).head(1)
    best_interpretable = class_summary[
        class_summary["model"].isin(["LogisticRegression", "LogisticRegressionBalanced"])
    ].sort_values(["pr_auc_mean", "precision_at_10_mean"], ascending=False).head(1)
    best_regression = reg_summary.sort_values(["r2_mean", "spearman_mean"], ascending=False).head(1)

    bayes_compare = raw_vs_bayes.pivot_table(
        index="min_rating_count_filter",
        columns="target",
        values="pr_auc_mean",
        aggfunc="first",
    )
    bayes_takeaway = "Bayesian targets did not clearly beat the raw bottom-30 target."
    if not bayes_compare.empty and "low_rating_bottom30_raw" in bayes_compare.columns:
        for candidate in ("low_rating_bottom30_bayes_global_m25", "low_rating_bottom30_bayes_category_m25"):
            if candidate in bayes_compare.columns:
                improvement = (bayes_compare[candidate] - bayes_compare["low_rating_bottom30_raw"]).dropna()
                if len(improvement) and improvement.max() > 0.005:
                    bayes_takeaway = "At least one Bayesian target matched or exceeded the raw bottom-30 target on PR-AUC."
                    break

    quantity_takeaway = "Quantity and interaction features should only be retained when they add measurable lift."
    if not quantity_compare.empty:
        best_quantity = quantity_compare.groupby("feature_set")["pr_auc_mean"].max()
        if "content_only_core" in best_quantity.index and "content_plus_quantity" in best_quantity.index:
            if best_quantity["content_plus_quantity"] > best_quantity["content_only_core"] + 0.005:
                quantity_takeaway = "Quantity features improved at least one screening setup enough to justify retention."

    threshold_best = threshold_df.sort_values("f1", ascending=False).head(1)
    threshold_text = "No threshold table was generated."
    if not threshold_best.empty:
        row = threshold_best.iloc[0]
        threshold_text = (
            f"The best observed held-out threshold among the top models was {row['threshold']:.2f}, "
            f"with precision {row['precision']:.3f}, recall {row['recall']:.3f}, and F1 {row['f1']:.3f}."
        )

    def row_text(frame: pd.DataFrame, default: str) -> str:
        if frame.empty:
            return default
        row = frame.iloc[0]
        return (
            f"`{row['target']}` with `{row['feature_set']}` and `{row['model']}` "
            f"(min_rating_count filter = {int(row['min_rating_count_filter'])}) "
            f"reached mean PR-AUC {row['pr_auc_mean']:.3f}, precision@10 {row['precision_at_10_mean']:.3f}, "
            f"lift@10 {row['lift_at_10_mean']:.3f}, and tuned F1 {row['tuned_f1_mean']:.3f}."
        )

    memo = f"""# Final model recommendation v3

## 1. Executive summary

This analysis reframes the project as a **lower-rating-risk ranking and screening problem**, not an exact star-rating prediction task. The strongest configurations are the ones that rank recipes by relative downside risk, especially when reliability information such as rating count is available or when the analysis is restricted to recipes with `rating_count >= 25`.

{row_text(best_overall, "No overall classification model summary was available.")}

## 2. What changed relative to v2

- Consolidated the work into a single `v3` runner with fixed seeds and repeatable output tables.
- Added target families that explicitly compare raw, Bayesian-global, Bayesian-category, and category-relative low-rating definitions.
- Added paired comparisons for all recipes versus the `rating_count >= 25` reliable subset.
- Added feature-family ablations separating content-only, reliability-aware, quantity-aware, and interaction-aware screens.
- Added threshold tuning, calibration, subgroup robustness, and report-ready summary tables.

## 3. Rating target construction

The base rating source used here was `{rating_info['rating_source_used']}`. The target-family summary is saved to `target_family_summary.csv`, including the raw bottom-30 and bottom-20 targets, the Bayesian global and category-adjusted targets, and the category-relative bottom-30 target with explicit sparse-category fallback behavior.

## 4. Why true rating reconstruction was or was not possible

{rating_info['notes']}

## 5. Final target comparison

{bayes_takeaway}

The target-family comparison tables prioritize PR-AUC, precision@10, and lift@10 because those directly align with ranking recipes for follow-up review rather than trying to predict exact star values.

## 6. All recipes vs reliable-rating subset

{row_text(best_reliable, "No reliable-subset model summary was available.")}

Filtering to `rating_count >= 25` is more statistically defensible because the observed rating is less noisy, but it reduces coverage. The paired comparison table is saved to `comparison_all_vs_rating_count25.csv`.

## 7. Content-only vs content-plus-reliability

{row_text(best_content_only, "No content-only model summary was available.")}

Reliability features such as `log_rating_count` and `log_review_count` are useful for post-publication screening, but they are not available for cold-start or pre-publication screening. The matched deltas are saved to `comparison_content_vs_reliability.csv`.

## 8. Quantity and interaction feature results

{quantity_takeaway}

The quantity/interactions ablation table is saved to `comparison_quantity_interactions.csv`.

## 9. Final recommended models by use case

The use-case recommendation table is saved to `best_model_by_use_case.csv`. It separates:

- pre-publication content-only screening
- published-recipe screening with reliability controls
- reliable-rating-only screening
- interpretable culinary association analysis
- secondary continuous-rating regression

## 10. Calibration and threshold recommendation

{threshold_text}

The threshold and calibration outputs are saved to `threshold_tuning_top_models.csv` and `calibration_top_models.csv`, with plots for the top model saved alongside the memo.

## 11. Subgroup robustness

Subgroup tables are provided for rating-count buckets, major categories, and cuisines where there was enough support. Sparse or single-class subgroups were skipped and explicitly flagged in the output notes.

## 12. Main limitations

- This is **not a causal model** of recipe quality; it is an associational screening model.
- Exact rating prediction remains secondary. If regression `R^2` remains low, exact star prediction should not be presented as a main success.
- The dataset does not provide per-star vote distributions, so reconstructed true ratings could not be recovered here.
- Category-aware targets depend on available category labels and use global fallback when category support is too small.

## 13. Final recommendation

Use the classification ranking framework as the main deliverable. Prefer either a reliability-filtered analysis (`rating_count >= 25`) or a Bayesian-adjusted target when the goal is a more defensible estimate of lower-rating risk. Keep quantity and interaction features only when they improve PR-AUC, precision@10, or lift@10, or when they add stable interpretability that a simpler content-only model does not provide.
"""

    best_setup = f"""# Best predictive setup v3

## Best overall classification setup

{row_text(best_overall, "No overall classification model summary was available.")}

## Best content-only setup

{row_text(best_content_only, "No content-only model summary was available.")}

## Best reliable-rating setup

{row_text(best_reliable, "No reliable-rating model summary was available.")}

## Best interpretable setup

{row_text(best_interpretable, "No interpretable classification model summary was available.")}

## Best regression setup

"""
    if best_regression.empty:
        best_setup += "No regression summary was available.\n"
    else:
        row = best_regression.iloc[0]
        best_setup += (
            f"`{row['target']}` with `{row['feature_set']}` and `{row['model']}` "
            f"(min_rating_count filter = {int(row['min_rating_count_filter'])}) "
            f"reached mean RMSE {row['rmse_mean']:.3f}, MAE {row['mae_mean']:.3f}, R^2 {row['r2_mean']:.3f}, "
            f"Spearman {row['spearman_mean']:.3f}, and Pearson {row['pearson_mean']:.3f}.\n"
        )

    best_setup += f"""

## Selected threshold guidance

{threshold_text}

## Practical interpretation of precision@10 and lift@10

- `precision@10` estimates the share of true lower-rating recipes among the top 10 percent of recipes flagged by the model.
- `lift@10` compares that precision against the base rate. A lift of 2.0 means the top-decile screen is finding lower-rating recipes at about twice the background rate.
- These are screening metrics, so they are most useful for ranking recipes for manual review or further QA, not for claiming exact rating prediction.
"""

    (REPORT_DIR / "final_model_recommendation_v3.md").write_text(memo)
    (REPORT_DIR / "best_predictive_setup_v3.md").write_text(best_setup)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final recipe-rating screening analysis v3.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Run the quick comparison suite.")
    mode_group.add_argument("--full", action="store_true", help="Run the fuller comparison suite.")
    args = parser.parse_args()

    mode = "full" if args.full else "quick"
    config = get_run_config(mode)
    ensure_report_dir()

    df, long_df, rating_info = build_master_dataset(config)
    df, target_summary = add_target_columns(df, config, rating_info)
    target_summary.to_csv(REPORT_DIR / "target_family_summary.csv", index=False)

    feature_summary_rows = []
    class_results: List[Dict[str, Any]] = []
    reg_results: List[Dict[str, Any]] = []

    class_models = get_classification_models(config.mode)
    reg_models = get_regression_models(config.mode)

    for min_rating_count_filter in config.rating_count_filters:
        print(f"[v3] running filter min_rating_count={min_rating_count_filter}", flush=True)
        subset = df[df["base_rating"].notna()].copy()
        if min_rating_count_filter > 0:
            subset = subset[subset["rating_count_num"] >= min_rating_count_filter].copy()
        if subset.empty:
            continue

        split_seed = config.seeds[0]
        anchor_target = "low_rating_bottom30_raw"
        train_anchor, test_anchor = split_with_stratification(subset, anchor_target, split_seed, config.test_size)

        for feature_set in FEATURE_SET_SPECS:
            print(f"[v3] feature_set={feature_set}", flush=True)
            builder = DenseFeatureBuilder(config, feature_set)
            X_train_anchor = builder.fit_transform(train_anchor)
            X_test_anchor = builder.transform(test_anchor)
            feature_summary_rows.append(
                {
                    "feature_set": feature_set,
                    "includes_content": True,
                    "includes_text": True,
                    "includes_quantity": FEATURE_SET_SPECS[feature_set]["include_quantity"],
                    "includes_quantity_interactions": FEATURE_SET_SPECS[feature_set]["include_interactions"],
                    "includes_preparation_interactions": FEATURE_SET_SPECS[feature_set]["include_interactions"],
                    "includes_rating_count": FEATURE_SET_SPECS[feature_set]["include_reliability"],
                    "n_features_train_average": int(X_train_anchor.shape[1]),
                    "notes": "Train-fitted dense feature builder with category/cuisine/source labels, ingredient indicators, nutrition/time/complexity features, and reduced text features.",
                }
            )

            for seed in config.seeds:
                train_df, test_df = split_with_stratification(subset, anchor_target, seed, config.test_size)
                builder = DenseFeatureBuilder(config, feature_set)
                X_train = builder.fit_transform(train_df)
                X_test = builder.transform(test_df)

                for target_name in config.classification_targets:
                    y_train = train_df[target_name].astype(int).to_numpy()
                    y_test = test_df[target_name].astype(int).to_numpy()
                    positive_share = float(y_test.mean()) if len(y_test) else np.nan
                    for model_name, model in class_models.items():
                        metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test, config.threshold_grid)
                        class_results.append(
                            {
                                "seed": seed,
                                "target": target_name,
                                "feature_set": feature_set,
                                "model": model_name,
                                "min_rating_count_filter": min_rating_count_filter,
                                "n_obs": int(len(test_df)),
                                "positive_class_share": positive_share,
                                "pr_auc": metrics["pr_auc"],
                                "roc_auc": metrics["roc_auc"],
                                "precision_at_5": metrics["precision_at_5"],
                                "precision_at_10": metrics["precision_at_10"],
                                "precision_at_20": metrics["precision_at_20"],
                                "recall_at_5": metrics["recall_at_5"],
                                "recall_at_10": metrics["recall_at_10"],
                                "recall_at_20": metrics["recall_at_20"],
                                "lift_at_5": metrics["lift_at_5"],
                                "lift_at_10": metrics["lift_at_10"],
                                "lift_at_20": metrics["lift_at_20"],
                                "default_f1": metrics["default_f1"],
                                "tuned_f1": metrics["tuned_f1"],
                                "best_threshold": metrics["best_threshold"],
                                "brier_score": metrics["brier_score"],
                            }
                        )

                for target_name in config.regression_targets:
                    y_train = train_df[target_name].to_numpy()
                    y_test = test_df[target_name].to_numpy()
                    train_mask = np.isfinite(y_train)
                    test_mask = np.isfinite(y_test)
                    if train_mask.sum() < 100 or test_mask.sum() < 50:
                        continue
                    X_train_reg = X_train[train_mask]
                    X_test_reg = X_test[test_mask]
                    y_train_reg = y_train[train_mask]
                    y_test_reg = y_test[test_mask]
                    for model_name, model in reg_models.items():
                        metrics = evaluate_regressor(model, X_train_reg, y_train_reg, X_test_reg, y_test_reg)
                        reg_results.append(
                            {
                                "seed": seed,
                                "target": target_name,
                                "feature_set": feature_set,
                                "model": model_name,
                                "min_rating_count_filter": min_rating_count_filter,
                                "n_obs": int(len(y_test_reg)),
                                **metrics,
                            }
                        )

    feature_summary = (
        pd.DataFrame(feature_summary_rows)
        .drop_duplicates(subset=["feature_set"])
        .sort_values("feature_set")
        .reset_index(drop=True)
    )
    feature_summary.to_csv(REPORT_DIR / "feature_family_summary.csv", index=False)

    class_results_df = pd.DataFrame(class_results)
    reg_results_df = pd.DataFrame(reg_results)
    class_results_df.to_csv(REPORT_DIR / "model_results_classification.csv", index=False)
    reg_results_df.to_csv(REPORT_DIR / "model_results_regression.csv", index=False)

    class_metric_cols = [
        "n_obs", "positive_class_share", "pr_auc", "roc_auc",
        "precision_at_5", "precision_at_10", "precision_at_20",
        "recall_at_5", "recall_at_10", "recall_at_20",
        "lift_at_5", "lift_at_10", "lift_at_20",
        "default_f1", "tuned_f1", "best_threshold", "brier_score",
    ]
    reg_metric_cols = ["n_obs", "rmse", "mae", "r2", "spearman", "pearson"]

    class_summary = summarize_results(
        class_results_df,
        group_cols=["target", "feature_set", "model", "min_rating_count_filter"],
        metric_cols=class_metric_cols,
    )
    reg_summary = summarize_results(
        reg_results_df,
        group_cols=["target", "feature_set", "model", "min_rating_count_filter"],
        metric_cols=reg_metric_cols,
    )
    class_summary.to_csv(REPORT_DIR / "model_results_classification_summary.csv", index=False)
    reg_summary.to_csv(REPORT_DIR / "model_results_regression_summary.csv", index=False)

    all_vs_filter, raw_vs_bayes, content_vs_reliability, quantity_compare, best_use_cases = build_focused_comparisons(
        class_summary,
        reg_summary,
    )
    all_vs_filter.to_csv(REPORT_DIR / "comparison_all_vs_rating_count25.csv", index=False)
    raw_vs_bayes.to_csv(REPORT_DIR / "comparison_raw_vs_bayesian.csv", index=False)
    content_vs_reliability.to_csv(REPORT_DIR / "comparison_content_vs_reliability.csv", index=False)
    quantity_compare.to_csv(REPORT_DIR / "comparison_quantity_interactions.csv", index=False)
    best_use_cases.to_csv(REPORT_DIR / "best_model_by_use_case.csv", index=False)

    threshold_df, calibration_df = save_top_model_artifacts(df, class_summary, config)
    evaluate_subgroups(df, class_summary, config)
    save_figures(class_summary, content_vs_reliability, quantity_compare, all_vs_filter)
    build_markdown_reports(
        class_summary,
        reg_summary,
        target_summary,
        feature_summary,
        all_vs_filter,
        raw_vs_bayes,
        content_vs_reliability,
        quantity_compare,
        best_use_cases,
        threshold_df,
        rating_info,
        config,
    )

    print(f"Run mode: {config.mode}")
    print(f"Report directory: {REPORT_DIR}")
    print(f"Classification results rows: {len(class_results_df)}")
    print(f"Regression results rows: {len(reg_results_df)}")


if __name__ == "__main__":
    main()
