# Best predictive setup v3

## Best overall classification setup

`low_rating_bottom30_bayes_category_m25` with `content_quantity_plus_reliability` and `XGBoostClassifier` (min_rating_count filter = 25) reached mean PR-AUC 0.590, precision@10 0.695, lift@10 2.384, and tuned F1 0.527.

## Best content-only setup

`low_rating_bottom30_bayes_global_m25` with `content_only_core` and `ExtraTreesClassifier` (min_rating_count filter = 25) reached mean PR-AUC 0.566, precision@10 0.676, lift@10 2.156, and tuned F1 0.327.

## Best reliable-rating setup

`low_rating_bottom30_bayes_category_m25` with `content_quantity_plus_reliability` and `XGBoostClassifier` (min_rating_count filter = 25) reached mean PR-AUC 0.590, precision@10 0.695, lift@10 2.384, and tuned F1 0.527.

## Best interpretable setup

`low_rating_bottom30_bayes_category_m25` with `content_plus_reliability` and `LogisticRegression` (min_rating_count filter = 25) reached mean PR-AUC 0.572, precision@10 0.682, lift@10 2.342, and tuned F1 0.562.

## Best regression setup

`continuous_bayes_global_m25` with `content_quantity_plus_reliability` and `HistGradientBoostingRegressor` (min_rating_count filter = 25) reached mean RMSE 0.159, MAE 0.119, R^2 0.282, Spearman 0.550, and Pearson 0.532.


## Selected threshold guidance

The best observed held-out threshold among the top models was 0.25, with precision 0.467, recall 0.723, and F1 0.568.

## Practical interpretation of precision@10 and lift@10

- `precision@10` estimates the share of true lower-rating recipes among the top 10 percent of recipes flagged by the model.
- `lift@10` compares that precision against the base rate. A lift of 2.0 means the top-decile screen is finding lower-rating recipes at about twice the background rate.
- These are screening metrics, so they are most useful for ranking recipes for manual review or further QA, not for claiming exact rating prediction.
