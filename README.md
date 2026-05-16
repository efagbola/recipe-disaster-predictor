# Recipe Rating Prediction Project

This project investigates whether recipe ingredients, nutrition information, cooking details, and recipe metadata can help predict recipe ratings and identify recipes that may be at risk of receiving lower user ratings.

The project combines machine learning notebooks, supporting scripts, cleaned output datasets, final report outputs, and a Streamlit demo app.

## Project Goal

The overall goal is to understand the relationship between recipe features and user ratings.

The project has two connected aims:

1. Predict the rating or lower-rating risk of a recipe using machine learning.
2. Identify ingredients, ingredient combinations, and recipe features associated with lower ratings.

This means the project includes both rating prediction and lower-rating-risk analysis.

## Repository Structure

```text
recipe-disaster-predictor/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── scrape_all.py
│   ├── clean_data.py
│   ├── descriptive_stats.ipynb
│   ├── applied_ml_recipe_rating_feature_components.ipynb
│   ├── classification_models.ipynb
│   ├── ingredient_interpretation.ipynb
│   ├── ingredient_combinations.py
│   └── final_model_recommendation_v3.py
│
├── outputs/
│   ├── allrecipes_all.csv.zip
│   ├── recipes_clean.csv
│   ├── recipes_ingredients_long.csv.zip
│   ├── ingredient_summary.csv
│   ├── risky_ingredient_pairs.csv
│   └── risky_ingredient_triples.csv
│
├── report/
│   └── final_model_v3/
│
└── streamlit_app/
    ├── streamlit_app.py
    ├── requirements.txt
    └── streamlit_artifacts/
```

## Main Analysis Notebooks

The main analysis work is in three key notebooks.

### `src/applied_ml_recipe_rating_feature_components.ipynb`

This notebook focuses on predicting recipe ratings using different feature groups. It uses ingredients, nutrition values, recipe complexity, popularity signals, and metadata to predict the rating a recipe is likely to receive.

This notebook supports the rating prediction side of the project and helps show which feature groups are useful for modelling recipe ratings.

### `src/classification_models.ipynb`

This notebook focuses on predicting whether a recipe is likely to be lower-rated. It treats the task as a classification problem and compares different models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

This notebook is important because lower-rated recipes are the main risk group the project wants to identify.

### `src/ingredient_interpretation.ipynb`

This notebook explains which ingredients and ingredient combinations are associated with lower ratings. It uses model interpretation and ingredient summary outputs to investigate individual ingredients, risky ingredient pairs, and risky ingredient triples.

This notebook is important because it helps move the project beyond prediction and into explanation.

## Supporting Files in `src/`

### `src/scrape_all.py`

This script collects recipe data from Allrecipes, including recipe titles, ingredients, ratings, rating counts, nutrition values, and recipe URLs.

### `src/clean_data.py`

This script cleans the scraped recipe data. It standardizes ingredient names, converts rating and nutrition columns to numeric values, and creates cleaned recipe-level and ingredient-level datasets.

### `src/descriptive_stats.ipynb`

This notebook explores the cleaned dataset. It summarizes ratings, nutrition values, ingredient counts, common ingredients, and basic ingredient-rating patterns.

### `src/ingredient_combinations.py`

This script analyzes ingredient pairs and triples. It identifies combinations that appear often enough to study and checks which combinations are associated with lower ratings.

### `src/final_model_recommendation_v3.py`

This script runs the final modelling workflow and creates report-ready outputs. It compares model setups, evaluates classification and regression results, and saves summary tables and figures to the report folder.

## Outputs Folder

The `outputs/` folder contains cleaned datasets and ingredient summary files used by the notebooks and scripts.

Important files include:

- `allrecipes_all.csv.zip`: zipped raw scraped recipe dataset
- `recipes_clean.csv`: cleaned recipe-level dataset
- `recipes_ingredients_long.csv.zip`: ingredient-level dataset with one row per recipe ingredient
- `ingredient_summary.csv`: summary of ingredient frequency and rating patterns
- `risky_ingredient_pairs.csv`: ingredient pairs associated with lower ratings
- `risky_ingredient_triples.csv`: ingredient triples associated with lower ratings

Some large files are zipped so they can be stored more easily in GitHub.

## Report Folder

The `report/` folder contains the final pdf required for submission.

Inside `report/final_model_v3/` there are report-ready outputs from the final modelling script. This includes model comparison tables, figures, summaries, and final recommendation files.


## Streamlit Demo App

A Streamlit demo app is included as a side application.

The app allows users to enter or select recipe ingredients and optional nutrition values. It then predicts a likely recipe rating and uses SHAP values to show which ingredients are pushing the prediction higher or lower.

The app is included to demonstrate how the machine learning model could be used in a practical setting. For example, a recipe creator or food platform could test ingredient combinations and see how the predicted rating changes.

Deployed app:

https://recipe-predictor-ml.streamlit.app/

## Running the Streamlit App Locally

To run the Streamlit app yourself:

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Running the Final Modelling Script

The main final modelling script can be run from the project root.

Quick run:

```bash
python3 src/final_model_recommendation_v3.py --quick
```

Full run:

```bash
python3 src/final_model_recommendation_v3.py --full
```

If the input files are not found automatically, pass the paths manually:

```bash
python3 src/final_model_recommendation_v3.py --full \
  --recipes-file outputs/allrecipes_all.csv \
  --ingredients-file outputs/recipes_ingredients_long.csv
```

## Summary

This project uses machine learning to study recipe ratings from both a predictive and explanatory perspective.

The feature components notebook predicts recipe ratings, the classification notebook identifies lower-rated recipes, and the ingredient interpretation notebook explains which ingredients and combinations are linked to lower ratings.

The Streamlit app provides a simple interactive demo of the model in action.
