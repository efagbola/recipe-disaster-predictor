import ast
import re
import pandas as pd


INPUT_FILE = "allrecipes_all.csv"
LONG_OUT = "recipes_ingredients_long.csv"
WIDE_OUT = "recipes_ingredients_wide.csv"


def clean_text(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


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
    """
    Turns:
    '1 tablespoon olive oil' -> 'olive oil'
    '1/2 teaspoon garlic powder' -> 'garlic powder'
    '8 ounces cream cheese, softened' -> 'cream cheese softened'
    """
    s = clean_text(ingredient)
    if not s:
        return None

    s = s.replace("(", " ").replace(")", " ")
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    qty_pattern = r"""
        ^
        (?:
            \d+\s+\d+/\d+ |      # 1 1/2
            \d+/\d+ |            # 1/2
            \d+(?:\.\d+)?        # 1 or 1.5
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


def canonicalize_ingredient(s):
    if not s:
        return None

    replacements = {
        "bell pepper": "pepper",
        "bell peppers": "pepper",
        "cream cheese": "cream cheese",
        "olive oil": "olive oil",
        "garlic powder": "garlic powder",
        "garlic cloves": "garlic",
        "cloves garlic": "garlic",
        "eggs": "egg",
        "yams": "yam",
        "tortillas": "tortilla",
        "string cheese": "cheese",
        "marinara sauce": "marinara",
    }

    s = s.strip()

    if s in replacements:
        return replacements[s]

    # simple singularization
    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("s") and not s.endswith("ss") and len(s) > 3:
        s = s[:-1]

    return s


def main():
    df = pd.read_csv(INPUT_FILE).reset_index(drop=False).rename(columns={"index": "recipe_id"})

    df["ingredient_list"] = df["ingredients"].apply(parse_ingredient_list)

    long_rows = []

    for _, row in df.iterrows():
        recipe_id = row["recipe_id"]
        title = row.get("title")
        rating_value = row.get("rating_value")
        rating_count = row.get("rating_count")
        calories = row.get("nutrition_calories")
        carbs = row.get("nutrition_carbs")
        fat = row.get("nutrition_fat")
        protein = row.get("nutrition_protein")
        url = row.get("url")

        for i, raw_ing in enumerate(row["ingredient_list"], start=1):
            cleaned = strip_quantity_prefix(raw_ing)
            canonical = canonicalize_ingredient(cleaned)

            long_rows.append({
                "recipe_id": recipe_id,
                "title": title,
                "ingredient_index": i,
                "ingredient_raw": raw_ing,
                "ingredient_clean": cleaned,
                "ingredient_canonical": canonical,
                "rating_value": rating_value,
                "rating_count": rating_count,
                "nutrition_calories": calories,
                "nutrition_carbs": carbs,
                "nutrition_fat": fat,
                "nutrition_protein": protein,
                "url": url,
            })

    long_df = pd.DataFrame(long_rows)

    # remove blank ingredients
    long_df = long_df[long_df["ingredient_canonical"].notna()].copy()

    # save long format
    long_df.to_csv(LONG_OUT, index=False)

    # make wide binary matrix
    wide = (
        long_df.assign(value=1)
        .drop_duplicates(subset=["recipe_id", "ingredient_canonical"])
        .pivot(index="recipe_id", columns="ingredient_canonical", values="value")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    recipe_info_cols = [
        "recipe_id", "title", "rating_value", "rating_count",
        "nutrition_calories", "nutrition_carbs", "nutrition_fat",
        "nutrition_protein", "url"
    ]
    recipe_info = df[recipe_info_cols].drop_duplicates(subset=["recipe_id"])

    wide_df = recipe_info.merge(wide, on="recipe_id", how="left").fillna(0)

    wide_df.to_csv(WIDE_OUT, index=False)

    print(f"Saved long file: {LONG_OUT}")
    print(f"Saved wide file: {WIDE_OUT}")
    print(f"Long shape: {long_df.shape}")
    print(f"Wide shape: {wide_df.shape}")


if __name__ == "__main__":
    main()