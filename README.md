# Recipe Ingredient Ratings Project

This project uses Allrecipes data to study whether ingredients and nutrition information can help predict lower-rated recipes. The main goal is to understand which ingredients are associated with worse ratings and whether recipe quality can be predicted from ingredient composition.

`scrape_all.py` is the scraping script. It collects recipe information from Allrecipes, including titles, ingredients, ratings, rating counts, nutrition values, and recipe URLs.

`clean_data.py` is the data cleaning script. It takes the raw scraped data, cleans the ingredient lists, removes quantities and units, standardizes ingredient names, and converts ratings and nutrition values into numeric columns.

`descriptive_stats.ipynb` is the exploratory analysis notebook. It summarizes the cleaned data, looks at ratings and nutrition values, identifies common ingredients, and explores which individual ingredients are linked to lower average ratings.

`classification_models.ipynb` is the machine learning notebook. It builds baseline classification models to predict whether a recipe is lower-rated using ingredients, nutrition variables, ingredient count, and rating count.

`ingredient_combinations.py` is the ingredient interaction script. It was added after the baseline models to analyze ingredient pairs and triples and see which combinations are associated with lower ratings.

The project workflow is to scrape the data, clean it, explore the descriptive statistics, build baseline classification models, and then analyze ingredient combinations for deeper interpretation. The final goal is to predict recipe quality from ingredients and identify ingredients or ingredient combinations that may be linked to lower ratings.
