# Recipe Ingredient Ratings Project

This project uses Allrecipes data to study whether ingredients and nutrition information can help predict lower-rated recipes. The main goal is to understand which ingredients and ingredient combinations are associated with worse ratings.

`scrape_all.py` is the scraping script. It collects recipe information from Allrecipes, including titles, ingredients, ratings, rating counts, nutrition values, and recipe URLs.

`clean_data.py` is the data cleaning script. It takes the raw scraped data and cleans the ingredient lists, removes quantities and units, standardizes ingredient names, and converts ratings and nutrition values into numeric columns.

`descriptive_stats.ipynb` is the exploratory analysis notebook. It looks at the cleaned data, summarizes ratings and nutrition values, shows common ingredients, and explores which ingredients are linked to lower average ratings.

`ingredient_combinations.py` analyzes ingredient pairs and triples. It finds combinations of ingredients that appear often enough to study and checks whether those combinations are associated with lower ratings.

`classification_models.ipynb` is the machine learning notebook. It builds models to predict whether a recipe is lower-rated using ingredients, nutrition variables, ingredient count, and rating count.

The project workflow is to scrape the data, clean it, explore the descriptive statistics, analyze ingredient combinations, and then build classification models. The final goal is to predict recipe quality from ingredients and identify risky ingredients or ingredient combinations.
