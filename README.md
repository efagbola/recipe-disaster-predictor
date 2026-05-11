# Recipe Ratings Project

This project looks at whether recipe ingredients and nutrition information can help predict lower-rated recipes. It also identifies individual ingredients and ingredient combinations that are associated with lower ratings.

`scrape_all.py` collects recipe data from Allrecipes, including titles, ingredients, ratings, rating counts, nutrition values, and URLs.

`clean_data.py` cleans the scraped recipe data. It standardizes ingredient names, removes quantities and units, converts rating and nutrition columns to numeric values, and creates cleaned datasets for analysis.

`descriptive_stats.ipynb` explores the cleaned data. It includes summaries of ratings, nutrition values, ingredient counts, common ingredients, and basic ingredient rating patterns.

`classification_models.ipynb` builds and compares machine learning models to predict whether a recipe is lower-rated. It compares the champion Logistic Regression model with challenger models and evaluates them using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

`ingredient_combinations.py` analyzes ingredient pairs and triples. It finds combinations that appear often enough to study and checks which ones are associated with lower ratings.

`ingredient_interpretation.ipynb` explains the ingredient findings. It uses the chosen Logistic Regression model to identify individual ingredients associated with lower-rated recipes, and it also summarizes risky ingredient pairs and triples.

The overall workflow is to scrape the data, clean it, explore the dataset, build classification models, and then interpret which ingredients or ingredient combinations are linked to lower-rated recipes.
