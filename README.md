# IPL-Data-Analysis
Data analytics project on IPL 2008–2022 dataset. Covers EDA, outlier detection (IQR &amp; Z-score), logistic regression for match outcome prediction, and chi-square hypothesis testing. Built with pandas, numpy, matplotlib, seaborn, scikit-learn, and scipy.

# 🏏 Decoding the IPL: Can Data Predict the Next Champion?

**Analytics Induction Programme Project | IPL Dataset 2008–2022**

---

## 📌 Project Overview

This project applies a complete data analytics pipeline to the **IPL Complete Dataset (2008–2022)** to answer one core question:

> *Given the venue, competing teams, and toss decision — can a machine learning model predict whether the team batting first will win?*

---

## 📁 Repository Structure

```
ipl-analytics/
│
├── IPL_Analytics_Project.ipynb   # Main Jupyter Notebook (all code + outputs)
├── analysis_report.pdf           # Formal written report (D6)
│
├── plots/                        # All 8 saved visualisations (D2)
│   ├── team_wins.png
│   ├── toss_pie.png
│   ├── matches_per_season.png
│   ├── venue_heatmap.png
│   ├── score_distribution.png
│   ├── outlier_boxplot.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── README.md
```

> **Note:** The dataset files (`Dataset1.csv` and `IPL_Ball_by_Ball_2008_2022.csv`) are not included in this repository due to file size. Download them from [Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) and place them in the root folder before running the notebook.

---

## 📊 Dataset

| File | Description | Size |
|------|-------------|------|
| `Dataset1.csv` | One row per match — venue, teams, toss, result | 950 rows × 20 columns |
| `IPL_Ball_by_Ball_2008_2022.csv` | One row per ball bowled | 225,954 rows × 17 columns |

**Source:** [IPL Complete Dataset (2008–2022) on Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

---

## 🔧 Libraries Used

```python
pandas        # Data loading and manipulation
numpy         # Numerical computation
matplotlib    # Plotting
seaborn       # Statistical visualisation
scikit-learn  # Machine learning model and evaluation
scipy         # Statistical hypothesis testing
```

---

## 📋 Project Pipeline

### Section 1 — Load and Inspect Data
- Loaded both CSV files using `pd.read_csv()`
- Inspected shape, data types, null counts, and descriptive statistics

### Section 2 — Data Cleaning
- Removed 4 no-result matches (`WonBy = 'NoResults'`)
- Normalised season labels (e.g. `2007/08` → `2008`)
- Standardised venue names to remove duplicates
- Engineered the target variable **`bat_first_won`** (1 = batting-first team won, 0 = lost)

### Section 3 — Statistical Visualisation
Created 5 different plot types:

| Plot | Type | File |
|------|------|------|
| Top 10 Team Wins | Bar chart | `team_wins.png` |
| Toss Decision Split | Pie chart | `toss_pie.png` |
| Matches per Season | Line chart | `matches_per_season.png` |
| Team Wins by Venue | Heatmap | `venue_heatmap.png` |
| Score Distribution | Scatter plot | `score_distribution.png` |

### Section 4 — Outlier Detection
Detected extreme innings scores using two methods:

| Method | Rule | Outliers Found |
|--------|------|----------------|
| IQR Fence | Outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR] = [77.4, 236.4] | **32 innings** |
| Z-Score | \|z\| > 3 | **13 innings** |

All Z-score outliers were also caught by IQR. The IQR method is more sensitive.

### Section 5 — Feature Engineering
- Encoded `TossDecision` using `LabelEncoder`
- Created binary feature `team1_won_toss`
- One-hot encoded `Venue`, `Team1`, `Team2` using `pd.get_dummies()`
- Final feature matrix: **946 rows × 73 columns**

### Section 6 — Model Training and Evaluation
- **Model:** Logistic Regression (`max_iter=1000`, `random_state=42`)
- **Split:** Stratified 80/20 train-test split (`random_state=42`)
- **Train size:** 756 samples | **Test size:** 190 samples

#### D3 — Metric Summary Table

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| Accuracy | 0.6376 | 0.5368 |
| Precision | 0.6387 | 0.4789 |
| Recall | 0.4471 | 0.4000 |
| F1-Score | 0.5260 | 0.4359 |
| ROC-AUC | 0.6811 | 0.5393 |

### Section 7 — Chi-Square Hypothesis Test

#### D4 — Statistical Test Result

- **H0:** Toss decision and match outcome are independent
- **H1:** They are associated
- **Chi-Square Statistic:** 0.0291
- **Degrees of Freedom:** 1
- **p-value:** 0.8647

**Result:** Since p = 0.8647 > 0.05, we **fail to reject H0** — toss decision is NOT significantly associated with match outcome.

---

## 📈 Key Findings

- Teams batting first win only **44.9%** of matches — the chasing team has a slight advantage
- **Mumbai Indians** lead all-time wins across 15 seasons
- About **63.2%** of captains choose to field after winning the toss
- The logistic regression model achieves only **~54% test accuracy** — confirming that pre-match information (teams, venue, toss) has limited predictive power
- The chi-square test confirms: **winning the toss and choosing to bat or field does not significantly affect the match result**

---

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/your-username/ipl-analytics.git
cd ipl-analytics
```

2. Download the dataset from Kaggle and place both CSV files in the root folder

3. Install required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

4. Open the notebook
```bash
jupyter notebook IPL_Analytics_Project.ipynb
```

5. Run all cells from top to bottom — make sure a `plots/` folder exists in the same directory

---

## 📦 Deliverables

| ID | Deliverable | Description |
|----|-------------|-------------|
| D1 | `IPL_Analytics_Project.ipynb` | Complete Jupyter Notebook |
| D2 | `plots/*.png` | 8 saved visualisation files |
| D3 | Metric Summary Table | Inside notebook Section 6 |
| D4 | Chi-Square Result | Inside notebook Section 7 |
| D5 | Written Reflection | Inside notebook Section 8 |
| D6 | `analysis_report.pdf` | Formal written report |

---

## 👤 Author

**Om Dehariya**
First Year Studnt at IIT INDORE
---

## 📄 License

This project is for educational purposes as part of a college club induction programme.
