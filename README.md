# Financial Health Prediction (FinAccess 2024, Kenya)

A machine learning project that predicts how a Kenyan household's financial situation has changed (**Improved**, **Stayed the same** or **Worsened**) from demographic, income and financial-inclusion data in the FinAccess 2024 data sprint.

## Why this project

Financial inclusion surveys collect a lot of information about who saves, who borrows and who uses mobile money, but it is not obvious which of those factors are linked to a household doing better or worse. This project works through the full data science process on that data: cleaning, statistical testing, modelling and saving a model that can be reused.

## Dataset

- **Source:** FinAccess 2024 data sprint (`finaccess2024_datasprint.xlsx`)
- **Size:** 20,871 respondents and 28 columns (20,850 rows remain after cleaning)
- **Target:** `financial_status` (3 classes: Improved, Stayed the same, Worsened)
- **Coverage:** 47 counties, urban and rural respondents

| Group | Example columns |
|---|---|
| Demographics | `county`, `location_type`, `Sex`, `Age` (6 bands), `household_size`, `education_level`, `marital_status`, `has_disability` |
| Income | `monthly_income` |
| Savings and credit | `Savings_formal`, `Savings_informal`, `Loan_formal`, `Loan_informal`, `defaulted` |
| Mobile money and banking | `mobile_money_access`, `mobile_ownership_1`, `barriers_mobile_money`, `barriers_bank`, `formal_service_use` |
| Financial resilience | `experienced_shock`, `accessto_13k_1month`, `not_difficult`, `fl_score`, `prodsum1` |

## Workflow

### 1. Data checks
- Checked duplicates (5 found), shape, missing values, data types, and summary statistics for categorical and numerical columns.
- Reviewed every categorical column's unique values and drew boxplots for the numerical columns (`household_size`, `monthly_income`, `prodsum1`).

### 2. Data cleaning
- Dropped the 5 duplicate rows.
- **Education level:** merged the long survey labels into a short set (None, Primary Incomplete, Primary, Secondary Incomplete, Secondary, Technical Training, University Incomplete, University) and removed "Don't know" and "Refused to answer" responses.
- **Marital status:** removed the "Don't know" and "Refused to answer" responses (together under 0.05% of rows).
- **Mobile money barriers:** replaced the placeholder `0` with `None`.
- **Missing values:** `barriers_bank` was the only column with nulls (27.5%), filled with the most frequent category (`Affordability`).

### 3. Exploratory analysis
- Visualised the distribution of financial status overall and against age, education level, disability status and formal savings use.
- Compared monthly income across formal savings users and non-users.

### 4. Hypothesis testing
- **Categorical features vs target:** chi-square test of independence for each categorical column. All of them showed a statistically significant association with `financial_status` at the 5% level.
- **Numerical features vs target:** Kruskal-Wallis test (used because the numeric variables are skewed and the group variances differ). `household_size`, `monthly_income` and `prodsum1` all differed significantly across the three groups.
- **Numerical vs numerical:** Spearman correlation. `monthly_income` and `prodsum1` have a moderate positive relationship (0.43). `household_size` is weakly negatively related to both (-0.12 and -0.16).

### 5. Preprocessing
- Stratified 70/30 train-test split (14,595 train, 6,255 test).
- `Age` ordinal-encoded in its natural order (16-17 up to Above 55).
- Other categorical columns one-hot encoded.
- Target encoded with `LabelEncoder` (0 = Improved, 1 = Stayed the same, 2 = Worsened).
- Numerical columns scaled with `RobustScaler`, which copes better with the income outliers than standard scaling.
- After encoding the model has 99 input features.

### 6. Modelling
Trained and compared Logistic Regression, K-Nearest Neighbours, Random Forest, Gradient Boosting and AdaBoost (AdaBoost was trained but not evaluated in the notebook).

## Results (test set, 6,255 respondents)

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression** | **0.57** | **0.46** |
| Gradient Boosting | 0.57 | 0.44 |
| Random Forest (200 trees, depth 10) | 0.56 | 0.39 |
| K-Nearest Neighbours (k=5) | 0.46 | 0.41 |

Logistic Regression was the best overall and was saved as the final model (`financial_health_best_model.pkl`, stored together with its label encoder).

**What the results tell us:** the target is hard to predict from this data. Accuracy of about 57% is modest for a three-class problem, and all models struggle most with the **Improved** and **Stayed the same** classes, with F1 scores around 0.3 compared with about 0.7 for **Worsened**. The classes are imbalanced, so accuracy alone flatters the models. Macro F1 is the fairer measure here.

## Key takeaways

- Every categorical variable in the dataset is statistically associated with financial status, but being statistically significant does not mean being strongly predictive. With about 20,000 rows even small differences come out as significant.
- Income and the number of financial products used (`prodsum1`) move together, which fits the idea that households with more income use more financial services.
- Simple models matched or beat the ensembles, which suggests the limit is the information in the features, not the model's complexity.

## Limitations and next steps

- **Class imbalance:** try `class_weight="balanced"`, SMOTE, or tune the decision threshold to improve recall on the minority classes.
- **Hyperparameter tuning and cross-validation:** the models were run with mostly default settings on a single split.
- **Pipeline:** move the encoding and scaling into a scikit-learn `Pipeline` / `ColumnTransformer`. This fits the preprocessing on training data only and removes the need to align train and test columns by hand.
- **Cleaning tidy-ups:** "Technical training" and "Technical Training" are still separate education categories, and a stray numeric value (`95`) remains in `education_level`. Both should be merged or removed.
- **Feature work:** consider grouping the 47 counties into regions to cut the number of dummy columns.
- **Deployment:** wrap the saved model in a Streamlit app so a user can enter details and see the predicted outcome.

## Tech stack

Python, pandas, NumPy, SciPy (chi-square, Kruskal-Wallis), scikit-learn, seaborn, matplotlib, joblib

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy scipy scikit-learn seaborn matplotlib openpyxl joblib jupyter
jupyter notebook Financess_ML-Project.ipynb
```

Update the file paths at the top of the notebook to point to where you keep the data.

## Author

Ephantus Macharia, aspiring data analyst, Nairobi, Kenya
[GitHub](https://github.com/machariaephantus96-coder)
