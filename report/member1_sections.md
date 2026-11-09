# Member 1 – Report sections (2, 5, 7)

All numbers are copied from `notebooks/01_tabular.ipynb` (seed 42).

## Section 2. Data merge and exploratory data analysis

**Data.** `customer_social_profiles.csv` has 155 rows and 5 columns (customer ID, platform, engagement score, purchase interest score, review sentiment). `customer_transactions.csv` has 150 rows and 6 columns (legacy customer ID, transaction ID, amount, date, product category, rating).

**Cleaning decisions.**
- The two files name the customer ID differently and store it differently (`'A178'` text vs `178` integer). We removed the `A` prefix, converted to integer, and named the column `customer_id` in both.
- 5 exact duplicate rows in the profiles file were dropped (155 → 150 rows).
- 10 transactions had a missing `customer_rating`. We kept them as missing and fill them with the training-set median inside the model pipeline, so no test information is used.
- A customer can have several profile rows (different platforms) and several transactions. Profile rows were summarised into one row per customer; every transaction was kept as one row.

**Join.** Inner join on `customer_id`, many transactions to one profile summary. We chose an inner join because a left join would keep customers without any social data and we would have to invent those values. Merge checks: 150 transactions before; 117 rows after; 61 customers in both files; 14 transaction customers without a profile (33 transactions dropped); no duplicated keys; no nulls introduced by the join (8 missing ratings come from the original data).

**EDA (Notebook Plots 1–4).**
1. Target distribution: five fairly balanced categories (Sports 23.9 %, Electronics 23.1 %, Clothing 18.8 %, Groceries 17.1 %, Books 17.1 %). A majority-class guess is right ~24 % of the time. Purchase amounts range from 62 to 495 with no strong peak.
2. Boxplots: no outliers by the 1.5×IQR rule for purchase amount, engagement or purchase interest, so no rows were removed. Median amount per category is between 247 and 312 with heavy overlap.
3. Correlation heatmap: only the sentiment shares correlate noticeably (−0.60, −0.53, −0.36; they sum to 1). All other pairs have |r| ≤ 0.23.
4. Social signals by category: mean purchase interest is 2.7–3.5 across categories with ~1 point spread; every platform buys from all five categories.

Meaning for modelling: the features carry only weak information about the category bought, so high accuracy should not be expected.

## Section 5. Product recommendation model

**Purpose.** After the face check passes, predict which product category the identified customer is likely to buy.

**Target.** `product_category` (Books, Clothing, Electronics, Groceries, Sports).

**Features (13).** Social profile summary per customer: number of profiles, average engagement, average purchase interest, shares of positive/neutral/negative reviews, main platform. Purchase history built only from the customer's earlier transactions: number of earlier purchases, total and average earlier spend, average earlier rating, days since last purchase (recency), most frequent earlier category. The current transaction's own amount, rating and date are not used (target leakage). Categorical columns are one-hot encoded; missing numbers are median-imputed; Logistic Regression uses scaled inputs. All of this sits inside a scikit-learn pipeline so it is fitted on training data only.

**Training.** Stratified, customer-grouped split (StratifiedGroupKFold, no customer in both train and test): 93 training rows, 24 test rows. Models: Logistic Regression (C = 0.1), Random Forest (300 trees, max depth 4, min leaf 3) and a majority-class baseline. We also ran 5-fold grouped cross-validation because 24 test rows is very noisy.

**Results (Table 5.1).**

| Model | Test accuracy | Test F1 macro | Test log loss | CV accuracy | CV F1 macro | CV log loss |
|---|---|---|---|---|---|---|
| Baseline (majority class) | 0.250 | 0.080 | 1.593 | 0.222 | 0.073 | 1.602 |
| Logistic Regression | 0.333 | 0.332 | 1.677 | 0.281 | 0.247 | 1.687 |
| Random Forest | 0.250 | 0.213 | 1.660 | 0.256 | 0.198 | 1.626 |

Test weighted F1: Logistic Regression 0.337, Random Forest 0.239, baseline 0.100. We selected Logistic Regression (best CV macro F1) and saved it in `models/product_model.pkl`, re-fitted on all 117 rows.

**Interpretation.** Both models are only slightly above the baseline in accuracy, and their log loss is worse than the baseline's (random guessing across 5 classes gives 1.609). The accuracy gap is about the size of the fold-to-fold standard deviation (0.05–0.06), so we cannot claim the models truly beat chance. On the test split, Groceries (F1 0.75) and Electronics (0.55) were predicted best and Books was never correct, but with 4–6 test rows per class this is noisy.

**Limitations.** Only 117 rows from 61 customers; weak relationship between the features and the target (see EDA); 14 customers dropped by the inner join; many customers have no purchase history, so their history features are imputed.

## Section 7. Results summary, limitations and conclusion (Member 1 part)

*Member 2 and Member 3 results for the face and voice models must be added from their notebooks.*

| Model | Accuracy | F1 (macro) | Log loss | Evaluation |
|---|---|---|---|---|
| Face model | (Member 2) | (Member 2) | (Member 2) | |
| Voice model | (Member 3) | (Member 3) | (Member 3) | |
| Product model (Logistic Regression) | 0.333 (CV 0.281) | 0.332 (CV 0.247) | 1.677 (CV 1.687) | grouped test split / 5-fold grouped CV |

**Limitations of the product model.** Small dataset, weak signal, and a log loss no better than the baseline. Its output should be treated as a demonstration of the pipeline, not a reliable recommendation. Better results would need more customers, more transactions per customer and richer product or browsing features.

**Conclusion.** We merged the two customer files on a cleaned ID, validated the join, explored the data, engineered leakage-free features from both sources and trained and compared two models against a baseline. The product model runs end-to-end through `predict_product(customer_id)` and is ready for the face → product → voice flow, but its predictive power is limited by the data.
