import time
from itertools import combinations

import pandas as pd


# -----------------------------
# Input files
# -----------------------------
LONG_INPUT = "recipes_ingredients_long.csv"
RECIPES_INPUT = "recipes_clean.csv"

# -----------------------------
# Output files
# -----------------------------
PAIR_SUMMARY_OUT = "ingredient_pair_summary.csv"
RISKY_PAIRS_OUT = "risky_ingredient_pairs.csv"
TRIPLE_SUMMARY_OUT = "ingredient_triple_summary.csv"
RISKY_TRIPLES_OUT = "risky_ingredient_triples.csv"

# -----------------------------
# Settings
# -----------------------------
MIN_RATING_COUNT = 10
LOW_RATING_THRESHOLD = 4.4

MIN_PAIR_COUNT = 20
MIN_TRIPLE_COUNT = 10

MAKE_TRIPLES = True


def load_data():
    print("Loading cleaned data...")

    long_df = pd.read_csv(LONG_INPUT)
    recipes_df = pd.read_csv(RECIPES_INPUT)

    print(f"Long ingredient data shape: {long_df.shape}")
    print(f"Recipe-level data shape: {recipes_df.shape}")

    return long_df, recipes_df


def prepare_long_data(long_df):
    print("\nPreparing long ingredient data...")

    needed_cols = [
        "recipe_id",
        "ingredient_canonical",
        "rating_value_num",
        "rating_count_num",
    ]

    missing_cols = [col for col in needed_cols if col not in long_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns from long data: {missing_cols}")

    df = long_df[needed_cols].copy()

    df = df.dropna(subset=[
        "recipe_id",
        "ingredient_canonical",
        "rating_value_num",
        "rating_count_num",
    ])

    df["rating_value_num"] = pd.to_numeric(df["rating_value_num"], errors="coerce")
    df["rating_count_num"] = pd.to_numeric(df["rating_count_num"], errors="coerce")

    df = df.dropna(subset=["rating_value_num", "rating_count_num"])

    print(f"Rows before rating-count filter: {df.shape[0]}")

    df = df[df["rating_count_num"] >= MIN_RATING_COUNT].copy()

    print(f"Rows after rating-count filter: {df.shape[0]}")

    df["is_lower_rated"] = (
        df["rating_value_num"] < LOW_RATING_THRESHOLD
    ).astype(int)

    df = df.drop_duplicates(subset=["recipe_id", "ingredient_canonical"])

    print(f"Rows after recipe-ingredient dedupe: {df.shape[0]}")
    print(f"Unique recipes retained: {df['recipe_id'].nunique()}")
    print(f"Unique ingredients retained: {df['ingredient_canonical'].nunique()}")

    return df


def build_recipe_ingredient_lists(long_df):
    print("\nBuilding recipe ingredient lists...")

    recipe_ingredients = (
        long_df.groupby("recipe_id")["ingredient_canonical"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    recipe_ingredients["ingredient_count"] = recipe_ingredients[
        "ingredient_canonical"
    ].apply(len)

    print(f"Recipes with ingredient lists: {recipe_ingredients.shape[0]}")
    print(
        "Average ingredients per recipe:",
        round(recipe_ingredients["ingredient_count"].mean(), 2),
    )

    return recipe_ingredients


def build_recipe_rating_table(long_df):
    print("\nBuilding recipe rating table...")

    recipe_ratings = (
        long_df[
            [
                "recipe_id",
                "rating_value_num",
                "rating_count_num",
                "is_lower_rated",
            ]
        ]
        .drop_duplicates(subset=["recipe_id"])
        .copy()
    )

    print(f"Recipe rating table shape: {recipe_ratings.shape}")

    return recipe_ratings


def generate_combinations(recipe_ingredients, combo_size=2):
    print(f"\nGenerating ingredient combinations of size {combo_size}...")

    combo_rows = []

    for idx, row in recipe_ingredients.iterrows():
        recipe_id = row["recipe_id"]
        ingredients = row["ingredient_canonical"]

        if len(ingredients) < combo_size:
            continue

        for combo in combinations(ingredients, combo_size):
            combo_row = {
                "recipe_id": recipe_id,
                "ingredient_combo": " + ".join(combo),
            }

            for i, ingredient in enumerate(combo, start=1):
                combo_row[f"ingredient_{i}"] = ingredient

            combo_rows.append(combo_row)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} recipes...")

    combo_df = pd.DataFrame(combo_rows)

    print(f"Generated combination rows: {combo_df.shape}")

    return combo_df


def summarize_combinations(combo_df, recipe_ratings, min_count):
    print("\nMerging combinations with recipe ratings...")

    combo_df = combo_df.merge(recipe_ratings, on="recipe_id", how="left")

    print("Summarizing combinations...")

    combo_summary = (
        combo_df.groupby("ingredient_combo")
        .agg(
            recipe_count=("recipe_id", "nunique"),
            avg_rating=("rating_value_num", "mean"),
            median_rating=("rating_value_num", "median"),
            avg_rating_count=("rating_count_num", "mean"),
            lower_rated_count=("is_lower_rated", "sum"),
            lower_rated_share=("is_lower_rated", "mean"),
        )
        .reset_index()
    )

    combo_summary = combo_summary[
        combo_summary["recipe_count"] >= min_count
    ].copy()

    combo_summary["lower_rated_share"] = combo_summary[
        "lower_rated_share"
    ].round(4)

    combo_summary["avg_rating"] = combo_summary["avg_rating"].round(4)
    combo_summary["median_rating"] = combo_summary["median_rating"].round(4)
    combo_summary["avg_rating_count"] = combo_summary[
        "avg_rating_count"
    ].round(2)

    combo_summary = combo_summary.sort_values(
        ["lower_rated_share", "avg_rating", "recipe_count"],
        ascending=[False, True, False],
    )

    print(f"Filtered summary shape: {combo_summary.shape}")

    return combo_summary


def add_baseline_comparisons(combo_summary, combo_df, recipe_ratings):
    print("\nAdding comparison against overall recipe baseline...")

    overall_avg_rating = recipe_ratings["rating_value_num"].mean()
    overall_lower_rated_share = recipe_ratings["is_lower_rated"].mean()

    combo_summary["overall_avg_rating"] = round(overall_avg_rating, 4)
    combo_summary["overall_lower_rated_share"] = round(
        overall_lower_rated_share, 4
    )

    combo_summary["rating_difference_from_overall"] = (
        combo_summary["avg_rating"] - overall_avg_rating
    ).round(4)

    combo_summary["lower_rated_share_difference_from_overall"] = (
        combo_summary["lower_rated_share"] - overall_lower_rated_share
    ).round(4)

    return combo_summary


def save_outputs(summary_df, summary_path, risky_path, top_n=50):
    print(f"\nSaving full summary to {summary_path}...")
    summary_df.to_csv(summary_path, index=False)

    print(f"Saving top risky combinations to {risky_path}...")
    summary_df.head(top_n).to_csv(risky_path, index=False)

    print("Saved.")


def print_top_results(summary_df, label, top_n=20):
    print(f"\nTop {top_n} risky {label}:")
    print(
        summary_df[
            [
                "ingredient_combo",
                "recipe_count",
                "avg_rating",
                "median_rating",
                "lower_rated_count",
                "lower_rated_share",
                "rating_difference_from_overall",
                "lower_rated_share_difference_from_overall",
            ]
        ]
        .head(top_n)
        .to_string(index=False)
    )


def main():
    start_time = time.time()

    print("Starting ingredient combination analysis...")
    print(f"Minimum rating count: {MIN_RATING_COUNT}")
    print(f"Lower-rated threshold: rating < {LOW_RATING_THRESHOLD}")
    print(f"Minimum pair count: {MIN_PAIR_COUNT}")
    print(f"Minimum triple count: {MIN_TRIPLE_COUNT}")
    print(f"Make triples: {MAKE_TRIPLES}")

    long_df, recipes_df = load_data()

    long_df = prepare_long_data(long_df)

    recipe_ingredients = build_recipe_ingredient_lists(long_df)

    recipe_ratings = build_recipe_rating_table(long_df)

    # -----------------------------
    # Ingredient pair analysis
    # -----------------------------
    pair_df = generate_combinations(
        recipe_ingredients=recipe_ingredients,
        combo_size=2,
    )

    pair_summary = summarize_combinations(
        combo_df=pair_df,
        recipe_ratings=recipe_ratings,
        min_count=MIN_PAIR_COUNT,
    )

    pair_summary = add_baseline_comparisons(
        combo_summary=pair_summary,
        combo_df=pair_df,
        recipe_ratings=recipe_ratings,
    )

    save_outputs(
        summary_df=pair_summary,
        summary_path=PAIR_SUMMARY_OUT,
        risky_path=RISKY_PAIRS_OUT,
        top_n=50,
    )

    print_top_results(pair_summary, label="ingredient pairs", top_n=20)

    # -----------------------------
    # Ingredient triple analysis
    # -----------------------------
    if MAKE_TRIPLES:
        triple_df = generate_combinations(
            recipe_ingredients=recipe_ingredients,
            combo_size=3,
        )

        triple_summary = summarize_combinations(
            combo_df=triple_df,
            recipe_ratings=recipe_ratings,
            min_count=MIN_TRIPLE_COUNT,
        )

        triple_summary = add_baseline_comparisons(
            combo_summary=triple_summary,
            combo_df=triple_df,
            recipe_ratings=recipe_ratings,
        )

        save_outputs(
            summary_df=triple_summary,
            summary_path=TRIPLE_SUMMARY_OUT,
            risky_path=RISKY_TRIPLES_OUT,
            top_n=50,
        )

        print_top_results(triple_summary, label="ingredient triples", top_n=20)

    runtime = time.time() - start_time

    print("\nDone.")
    print(f"Total runtime: {runtime:.1f} seconds")


if __name__ == "__main__":
    main()