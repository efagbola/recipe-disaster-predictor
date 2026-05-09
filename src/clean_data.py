import ast
import re
import time
import pandas as pd


INPUT_FILE = "allrecipes_all.csv"

LONG_OUT = "recipes_ingredients_long.csv"
RECIPES_OUT = "recipes_clean.csv"
INGREDIENT_SUMMARY_OUT = "ingredient_summary.csv"


def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x if x else None


def extract_number(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def parse_ingredient_list(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    try:
        vals = ast.literal_eval(x)
        if isinstance(vals, list):
            return [str(v).strip() for v in vals if str(v).strip()]
    except Exception:
        pass

    return []


def strip_quantity_prefix(ingredient):
    s = clean_text(ingredient)
    if not s:
        return None

    s = s.replace("(", " ").replace(")", " ")
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    qty_pattern = r"""
        ^
        (?:
            \d+\s+\d+/\d+ |
            \d+/\d+ |
            \d+(?:\.\d+)?
        )
        (?:\s*-\s*
            (?:
                \d+\s+\d+/\d+ |
                \d+/\d+ |
                \d+(?:\.\d+)?
            )
        )?
        \s*
    """
    s = re.sub(qty_pattern, "", s, flags=re.VERBOSE)

    units = [
        "teaspoon", "teaspoons", "tsp",
        "tablespoon", "tablespoons", "tbsp",
        "cup", "cups",
        "ounce", "ounces", "oz",
        "pound", "pounds", "lb", "lbs",
        "gram", "grams", "g", "kg",
        "milliliter", "milliliters", "ml",
        "liter", "liters", "l",
        "clove", "cloves",
        "can", "cans",
        "package", "packages", "packet", "packets",
        "slice", "slices",
        "pinch", "pinches",
        "dash", "dashes",
        "bunch", "bunches",
        "stick", "sticks",
        "quart", "quarts",
        "pint", "pints",
        "inch", "inches",
    ]

    units_pattern = r"^(?:" + "|".join(re.escape(u) for u in units) + r")\b\s*"
    s = re.sub(units_pattern, "", s)

    prep_words = [
        "fresh", "softened", "melted", "chopped", "diced", "minced",
        "crushed", "ground", "sliced", "beaten", "shredded", "grated",
        "divided", "drained", "rinsed", "optional", "to taste",
        "for serving", "for dipping", "if desired",
    ]

    for w in prep_words:
        s = re.sub(rf"\b{re.escape(w)}\b", "", s)

    s = re.sub(r"\bany color\b", "", s)
    s = re.sub(r"\breal\b", "", s)
    s = re.sub(r"\bfinely\b", "", s)
    s = re.sub(r"\bcoarsely\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else None


def normalize_ingredient(s):
    s = clean_text(s)
    if not s:
        return None

    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    replacements = {
        "bell pepper": "pepper",
        "bell peppers": "pepper",
        "cream cheese": "cream cheese",
        "olive oil": "olive oil",
        "garlic powder": "garlic powder",
        "garlic cloves": "garlic",
        "cloves garlic": "garlic",
        "eggs": "egg",
        "egg whites": "egg white",
        "egg yolks": "egg yolk",
        "yams": "yam",
        "tortillas": "tortilla",
        "string cheese": "cheese",
        "marinara sauce": "marinara",
        "onions": "onion",
        "tomatoes": "tomato",
        "potatoes": "potato",
        "breadcrumbs": "breadcrumb",
        "scallions": "green onion",
        "spring onions": "green onion",
        "coriander leaves": "cilantro",
    }

    if s in replacements:
        s = replacements[s]

    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]

    return s


def is_bad_ingredient(s):
    if not s:
        return True

    junk_exact = {
        "package",
        "packet",
        "container",
    }

    return s in junk_exact


def main():
    t0 = time.time()

    print("Reading CSV...")
    df = pd.read_csv(INPUT_FILE).reset_index(drop=False).rename(columns={"index": "recipe_id"})
    print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} cols | {time.time() - t0:.1f}s")

    print("Parsing ingredient lists...")
    df["ingredient_list"] = df["ingredients"].apply(parse_ingredient_list)
    print(f"Ingredient lists parsed | {time.time() - t0:.1f}s")

    print("Converting numeric columns...")
    numeric_map = {
        "rating_value": "rating_value_num",
        "rating_count": "rating_count_num",
        "nutrition_calories": "nutrition_calories_num",
        "nutrition_carbs": "nutrition_carbs_num",
        "nutrition_fat": "nutrition_fat_num",
        "nutrition_protein": "nutrition_protein_num",
    }

    for src, dst in numeric_map.items():
        if src in df.columns:
            df[dst] = df[src].apply(extract_number)
    print(f"Numeric conversion done | {time.time() - t0:.1f}s")

    print("Building long rows...")
    long_rows = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        recipe_id = row["recipe_id"]
        title = row.get("title")
        rating_value = row.get("rating_value")
        rating_value_num = row.get("rating_value_num")
        rating_count = row.get("rating_count")
        rating_count_num = row.get("rating_count_num")
        calories = row.get("nutrition_calories")
        calories_num = row.get("nutrition_calories_num")
        carbs = row.get("nutrition_carbs")
        carbs_num = row.get("nutrition_carbs_num")
        fat = row.get("nutrition_fat")
        fat_num = row.get("nutrition_fat_num")
        protein = row.get("nutrition_protein")
        protein_num = row.get("nutrition_protein_num")
        url = row.get("url")

        for i, raw_ing in enumerate(row["ingredient_list"], start=1):
            ingredient_raw = clean_text(raw_ing)
            ingredient_clean = strip_quantity_prefix(raw_ing)
            ingredient_canonical = normalize_ingredient(ingredient_clean)

            if ingredient_canonical is None or is_bad_ingredient(ingredient_canonical):
                continue

            long_rows.append({
                "recipe_id": recipe_id,
                "title": title,
                "ingredient_index": i,
                "ingredient_raw": ingredient_raw,
                "ingredient_clean": ingredient_clean,
                "ingredient_canonical": ingredient_canonical,
                "rating_value": rating_value,
                "rating_value_num": rating_value_num,
                "rating_count": rating_count,
                "rating_count_num": rating_count_num,
                "nutrition_calories": calories,
                "nutrition_calories_num": calories_num,
                "nutrition_carbs": carbs,
                "nutrition_carbs_num": carbs_num,
                "nutrition_fat": fat,
                "nutrition_fat_num": fat_num,
                "nutrition_protein": protein,
                "nutrition_protein_num": protein_num,
                "url": url,
            })

        if idx % 1000 == 0:
            print(f"Processed {idx}/{total} recipes | {time.time() - t0:.1f}s")

    print(f"Finished building long rows | {time.time() - t0:.1f}s")

    print("Creating long dataframe...")
    long_df = pd.DataFrame(long_rows)
    print(f"Long dataframe created: {long_df.shape} | {time.time() - t0:.1f}s")

    print("Dropping duplicate recipe-ingredient pairs...")
    long_df = long_df.drop_duplicates(subset=["recipe_id", "ingredient_canonical"]).copy()
    print(f"After dedupe: {long_df.shape} | {time.time() - t0:.1f}s")

    print(f"Saving {LONG_OUT} ...")
    long_df.to_csv(LONG_OUT, index=False)
    print(f"Saved {LONG_OUT} | {time.time() - t0:.1f}s")

    print("Preparing recipe-level clean file...")
    recipe_info_cols = [
        c for c in [
            "recipe_id",
            "title",
            "rating_value",
            "rating_value_num",
            "rating_count",
            "rating_count_num",
            "nutrition_calories",
            "nutrition_calories_num",
            "nutrition_carbs",
            "nutrition_carbs_num",
            "nutrition_fat",
            "nutrition_fat_num",
            "nutrition_protein",
            "nutrition_protein_num",
            "url",
        ] if c in df.columns
    ]

    recipe_info = df[recipe_info_cols].drop_duplicates(subset=["recipe_id"]).copy()

    ingredient_counts = (
        long_df.groupby("recipe_id")["ingredient_canonical"]
        .nunique()
        .reset_index(name="ingredient_count")
    )

    recipes_clean = recipe_info.merge(ingredient_counts, on="recipe_id", how="left")
    recipes_clean["ingredient_count"] = recipes_clean["ingredient_count"].fillna(0).astype(int)

    print(f"Saving {RECIPES_OUT} ...")
    recipes_clean.to_csv(RECIPES_OUT, index=False)
    print(f"Saved {RECIPES_OUT} | {time.time() - t0:.1f}s")

    print("Building ingredient summary...")
    ingredient_summary = (
        long_df.groupby("ingredient_canonical")
        .agg(
            recipe_count=("recipe_id", "nunique"),
            avg_rating=("rating_value_num", "mean"),
            median_rating=("rating_value_num", "median"),
            avg_rating_count=("rating_count_num", "mean"),
            avg_calories=("nutrition_calories_num", "mean"),
            avg_carbs=("nutrition_carbs_num", "mean"),
            avg_fat=("nutrition_fat_num", "mean"),
            avg_protein=("nutrition_protein_num", "mean"),
        )
        .reset_index()
        .sort_values(["recipe_count", "avg_rating"], ascending=[False, False])
    )

    print(f"Saving {INGREDIENT_SUMMARY_OUT} ...")
    ingredient_summary.to_csv(INGREDIENT_SUMMARY_OUT, index=False)
    print(f"Saved {INGREDIENT_SUMMARY_OUT} | {time.time() - t0:.1f}s")

    print("\nDone.")
    print("Long shape:", long_df.shape)
    print("Recipes shape:", recipes_clean.shape)
    print("Ingredient summary shape:", ingredient_summary.shape)
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()