# Recipe Ratings Project

This project looks at whether recipe ingredients and nutrition information can help predict lower-rated recipes. It also identifies individual ingredients and ingredient combinations that are associated with lower ratings.

`scrape_all.py` collects recipe data from Allrecipes, including titles, ingredients, ratings, rating counts, nutrition values, and URLs.

`clean_data.py` cleans the scraped recipe data. It standardizes ingredient names, removes quantities and units, converts rating and nutrition columns to numeric values, and creates cleaned datasets for analysis.

`descriptive_stats.ipynb` explores the cleaned data. It includes summaries of ratings, nutrition values, ingredient counts, common ingredients, and basic ingredient rating patterns.

`classification_models.ipynb` builds and compares machine learning models to predict whether a recipe is lower-rated. It compares the champion Logistic Regression model with challenger models and evaluates them using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

`ingredient_combinations.py` analyzes ingredient pairs and triples. It finds combinations that appear often enough to study and checks which ones are associated with lower ratings.

`ingredient_interpretation.ipynb` explains the ingredient findings. It uses the chosen Logistic Regression model to identify individual ingredients associated with lower-rated recipes, and it also summarizes risky ingredient pairs and triples.

The overall workflow is to scrape the data, clean it, explore the dataset, build classification models, and then interpret which ingredients or ingredient combinations are linked to lower-rated recipes.

## Final deliverable

The deliverable analysis layer is `src/final_model_recommendation_v3.py`. It consolidates the project around a lower-rating-risk screening workflow and writes the final report bundle to `report/final_model_v3/`.

The script is reproducible on another machine as long as the two source CSV files are available:

- `allrecipes_all.csv`
- `recipes_ingredients_long.csv`

It looks for those files in common project-relative locations such as the repo root, `data/`, `inputs/`, the parent folder, and the parent folder's `data/` or `inputs/` directories. You can also pass explicit paths.

### Run commands

Quick iteration:

```bash
python3 src/final_model_recommendation_v3.py --quick --recipes-file /path/to/allrecipes_all.csv --ingredients-file /path/to/recipes_ingredients_long.csv
```

Full deliverable run:

```bash
python3 src/final_model_recommendation_v3.py --full --recipes-file /path/to/allrecipes_all.csv --ingredients-file /path/to/recipes_ingredients_long.csv
```

If the files are placed in one of the default search locations, the explicit path flags are optional.

### Final outputs kept in git

This repo's deliver-ready outputs are the latest files under `report/final_model_v3/`, including:

- the final markdown memo
- the best-setup summary
- comparison tables
- full classification and regression summaries
- calibration, threshold, subgroup, and figure outputs
- `run_manifest_v3.json` documenting the last run inputs and seed settings
