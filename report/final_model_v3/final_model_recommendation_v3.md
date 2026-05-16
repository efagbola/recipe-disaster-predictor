# Final model recommendation v3

## 1. Executive summary

This analysis reframes the project as a **lower-rating-risk ranking and screening problem**, not an exact star-rating prediction task. The strongest configurations are the ones that rank recipes by relative downside risk, especially when reliability information such as rating count is available or when the analysis is restricted to recipes with `rating_count >= 25`.

`low_rating_bottom30_bayes_category_m25` with `content_quantity_plus_reliability` and `XGBoostClassifier` (min_rating_count filter = 25) reached mean PR-AUC 0.590, precision@10 0.695, lift@10 2.384, and tuned F1 0.527.

## 2. What changed relative to v2

- Consolidated the work into a single `v3` runner with fixed seeds and repeatable output tables.
- Added target families that explicitly compare raw, Bayesian-global, Bayesian-category, and category-relative low-rating definitions.
- Added paired comparisons for all recipes versus the `rating_count >= 25` reliable subset.
- Added feature-family ablations separating content-only, reliability-aware, quantity-aware, and interaction-aware screens.
- Added threshold tuning, calibration, subgroup robustness, and report-ready summary tables.

## 3. Rating target construction

The base rating source used here was `rating_value`. The target-family summary is saved to `target_family_summary.csv`, including the raw bottom-30 and bottom-20 targets, the Bayesian global and category-adjusted targets, and the category-relative bottom-30 target with explicit sparse-category fallback behavior.

## 4. Why true rating reconstruction was or was not possible

No individual star-count distribution columns were detected in the raw data; exact true-rating reconstruction was not possible.

## 5. Final target comparison

At least one Bayesian target matched or exceeded the raw bottom-30 target on PR-AUC.

The target-family comparison tables prioritize PR-AUC, precision@10, and lift@10 because those directly align with ranking recipes for follow-up review rather than trying to predict exact star values.

## 6. All recipes vs reliable-rating subset

`low_rating_bottom30_bayes_category_m25` with `content_quantity_plus_reliability` and `XGBoostClassifier` (min_rating_count filter = 25) reached mean PR-AUC 0.590, precision@10 0.695, lift@10 2.384, and tuned F1 0.527.

Filtering to `rating_count >= 25` is more statistically defensible because the observed rating is less noisy, but it reduces coverage. The paired comparison table is saved to `comparison_all_vs_rating_count25.csv`.

## 7. Content-only vs content-plus-reliability

`low_rating_bottom30_bayes_global_m25` with `content_only_core` and `ExtraTreesClassifier` (min_rating_count filter = 25) reached mean PR-AUC 0.566, precision@10 0.676, lift@10 2.156, and tuned F1 0.327.

Reliability features such as `log_rating_count` and `log_review_count` are useful for post-publication screening, but they are not available for cold-start or pre-publication screening. The matched deltas are saved to `comparison_content_vs_reliability.csv`.

## 8. Quantity and interaction feature results

Quantity features improved at least one screening setup enough to justify retention.

The quantity/interactions ablation table is saved to `comparison_quantity_interactions.csv`.

## 9. Final recommended models by use case

The use-case recommendation table is saved to `best_model_by_use_case.csv`. It separates:

- pre-publication content-only screening
- published-recipe screening with reliability controls
- reliable-rating-only screening
- interpretable culinary association analysis
- secondary continuous-rating regression

## 10. Calibration and threshold recommendation

The best observed held-out threshold among the top models was 0.25, with precision 0.467, recall 0.723, and F1 0.568.

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
