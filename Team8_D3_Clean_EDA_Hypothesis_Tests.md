# Team 8 – People and HR Analytics
## D3: Clean Data, EDA and Hypothesis Tests
BANA Team Project, PGDM BA 2026 · Team 8: K Krishnakanth Maheedhar, Irene Roby, Nirshitha Vijaya Kumar, Shirley Dutta, Shashanki Shikha, Shradha Shruti

*Platform: Databricks. Structure follows the class reference notebook "Business Analytics – Zero to Hero".*

# 1. Problem Statement

**Option B – What keeps employees satisfied and paid well?**

Which work and career factors are *associated with* higher job satisfaction and higher monthly income, and do employees with different satisfaction levels differ in pay?

**Decision supported:** how HR managers and senior leaders plan pay and improve employee experience.

**Questions this notebook answers (from D2):**
1. Which work and career factors go with higher job satisfaction?
2. How do experience, job level, role and other factors relate to monthly income?
3. Do employees with different satisfaction levels differ in pay?

**Data:** IBM HR Analytics Employee Attrition and Performance (Kaggle, Pavan Subhash). Simulated data, 1,470 rows × 35 columns. It has **no developer role**, so the unit of analysis is *employees*, not developers. Pay currency, country and year are not stated, so pay is a **relative** measure.

**Language rule:** we say "associated with", never "causes". The data is a single snapshot.

# 2. Exploratory Data Analysis

## 2.1 Loading a dataset

Upload the Kaggle CSV (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) to a Unity Catalog **Volume** (Catalog → your catalog → schema → *Create* → *Volume* → *Upload to this volume*), then set `DATA_PATH` below to that file's path. The raw file is never modified; all work happens on a copy.


```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# If a library is missing on your cluster, run in its own cell:  %pip install seaborn scipy scikit-learn

SEED = 42                      # fixed seed for reproducibility
np.random.seed(SEED)
ALPHA = 0.05                   # significance level used for every test

# >>> EDIT THIS PATH to your Volume location <<<
DATA_PATH = os.environ.get(
    "HR_CSV",
    "/Volumes/hr_analytics/ibm_hr/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
)
CLEAN_PATH = os.environ.get(
    "HR_CLEAN_OUT",
    "/Volumes/hr_analytics/ibm_hr/raw/hr_clean.csv",
)

pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid", palette="deep")
```


```python
df_raw = pd.read_csv(DATA_PATH)   # original, kept untouched
df = df_raw.copy()                # working copy
```

## 2.2 Inspecting the dataset


```python
df.shape
```


```python
df.head()
```


```python
df.info()
```


```python
df.describe()
```

### 2.2.1 Checking for missing values and duplicates


```python
df.isna().sum()
```


```python
print("Total missing cells :", int(df.isna().sum().sum()))
print("Duplicate rows      :", int(df.duplicated().sum()))
```

### 2.2.2 Slicing Data into groups


```python
df.loc[0:10, ["Department", "JobSatisfaction", "MonthlyIncome"]]   # .loc selects by label
```


```python
df.iloc[0:10, [0, 18]]   # .iloc selects by index position
```


```python
for idx, col in enumerate(df.columns):
    print(f"{idx}: {col}")
```

How many employees work overtime?


```python
overtime_yes = df[df.OverTime == "Yes"]
overtime_yes.shape
```

#### 2.2.2.1 Reshaping into conditional tables

D2 asked for a second view on **Research & Development**, the closest technical group, so we create it here. Small groups (for example Human Resources) are shown with their sizes and not over-interpreted.


```python
rd_group    = df.loc[df["Department"] == "Research & Development", ["Age", "MonthlyIncome", "JobSatisfaction", "OverTime"]]
sales_group = df.loc[df["Department"] == "Sales",                   ["Age", "MonthlyIncome", "JobSatisfaction", "OverTime"]]
hr_group    = df.loc[df["Department"] == "Human Resources",         ["Age", "MonthlyIncome", "JobSatisfaction", "OverTime"]]
print("R&D:", rd_group.shape, "| Sales:", sales_group.shape, "| HR:", hr_group.shape)
```

**Check the split of satisfaction in each group**


```python
df.groupby("Department")["JobSatisfaction"].value_counts(normalize=True).mul(100).round(1).unstack()
```

**Observing central tendency across groups**


```python
df.groupby("JobSatisfaction")["MonthlyIncome"].agg(["mean", "median", "std", "count"]).round(1)
```


```python
(df.groupby("Department")["MonthlyIncome"]
   .agg(["mean", "median", "min", "max", "count"])
   .rename(columns={"mean": "Avg_Monthly_Income", "median": "Median_Monthly_Income",
                    "min": "Min_Monthly_Income", "max": "Max_Monthly_Income", "count": "Count_Employees"})
   .round(1))
```

## 2.3 Data cleaning and the cleaning log

Every step below writes one row to `cleaning_log`: *what we checked, what we found, what we did, and why*. This follows the Data Quality Plan in D2 (Section 4). Where the data differs from what the documentation says, the difference is recorded here.


```python
cleaning_log = []

def log_step(step, check, finding, action, reason, d):
    cleaning_log.append({
        "Step": len(cleaning_log) + 1,
        "Check": check,
        "Finding": finding,
        "Action": action,
        "Reason": reason,
        "Rows_after": d.shape[0],
        "Cols_after": d.shape[1],
    })

df = df_raw.copy()
log_step(0, "Load raw file", f"{df.shape[0]} rows x {df.shape[1]} columns",
         "Work on a copy; raw file untouched", "Protect the original data", df)
```

**Step: compare against the documentation** (D2 says 1,470 rows and 35 columns)


```python
expected = (1470, 35)
match = df.shape == expected
log_step(1, "Shape vs documentation (1470 x 35)",
         "Matches" if match else f"DIFFERENT: found {df.shape}, expected {expected}",
         "None" if match else "Difference recorded; continue with the file as loaded",
         "D2 asked us to confirm counts on load", df)
```

**Step: missing values and duplicate rows**


```python
n_missing = int(df.isna().sum().sum())
log_step(2, "Missing values", f"{n_missing} missing cells in total",
         "No imputation needed" if n_missing == 0 else "Rows/cells with missing values listed below; decide per column",
         "Missing data can bias group comparisons", df)

n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
log_step(3, "Duplicate rows", f"{n_dup} duplicate rows",
         "Dropped" if n_dup else "None to drop",
         "Duplicates would double-count employees", df)
```

**Step: remove constant and identifier fields** (D2 Section 3.1: EmployeeCount, Over18, StandardHours are constants; EmployeeNumber is a row ID, which is also the only direct identifier under the DPDP Act checklist)


```python
planned_drop = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
constant_found = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
print("Constant columns found by the data :", constant_found)

to_drop = [c for c in planned_drop if c in df.columns]
df = df.drop(columns=to_drop)
log_step(4, "Constant / identifier fields",
         f"Constants in data: {constant_found}; plus ID EmployeeNumber",
         f"Dropped {to_drop}", "No information (constants) or privacy risk / no meaning (ID); data minimisation", df)
```

**Step: impossible or inconsistent values.** Each rule is a logical test (for example, you cannot have spent more years at this company than your whole career). Rows that fail are **flagged and kept**, not deleted, so the decision stays visible.


```python
rules = {
    "YearsAtCompany > TotalWorkingYears":        df.YearsAtCompany > df.TotalWorkingYears,
    "YearsInCurrentRole > YearsAtCompany":       df.YearsInCurrentRole > df.YearsAtCompany,
    "YearsWithCurrManager > YearsAtCompany":     df.YearsWithCurrManager > df.YearsAtCompany,
    "YearsSinceLastPromotion > YearsAtCompany":  df.YearsSinceLastPromotion > df.YearsAtCompany,
    "Started work before age 14 (Age - TotalWorkingYears < 14)": (df.Age - df.TotalWorkingYears) < 14,
    "MonthlyIncome <= 0":                        df.MonthlyIncome <= 0,
    "Any 1-4 rating outside 1..4": ~df[["JobSatisfaction","EnvironmentSatisfaction","JobInvolvement",
                                         "RelationshipSatisfaction","WorkLifeBalance","PerformanceRating"]]
                                     .isin([1,2,3,4]).all(axis=1),
}
rule_counts = pd.Series({k: int(v.sum()) for k, v in rules.items()}, name="Rows_failing")
rule_counts
```


```python
df["Flag_Inconsistent"] = np.column_stack(list(rules.values())).any(axis=1)
n_flag = int(df["Flag_Inconsistent"].sum())
log_step(5, "Impossible / inconsistent values", f"{n_flag} rows fail at least one rule (see rule_counts)",
         "Flagged in column Flag_Inconsistent; rows kept",
         "Few flagged rows should not drive results; we report them rather than silently delete", df)
```

**Step: types and coding.** Category fields become labelled categories. The 1-4 scales become **ordered categories** with readable labels (we keep the numeric code too, for correlations).


```python
cat_cols = ["Attrition", "BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"]
for c in cat_cols:
    df[c] = df[c].astype("category")

four = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
ordinal_maps = {
    "Education":                {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction":  four,
    "JobInvolvement":           four,
    "JobSatisfaction":          four,
    "RelationshipSatisfaction": four,
    "PerformanceRating":        {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "WorkLifeBalance":          {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}
for c, m in ordinal_maps.items():
    order = [m[k] for k in sorted(m)]
    df[c + "_Label"] = pd.Categorical(df[c].map(m), categories=order, ordered=True)

log_step(6, "Types and coding",
         f"{len(cat_cols)} category fields; {len(ordinal_maps)} ordinal fields given labelled ordered versions (*_Label)",
         "Converted; numeric codes kept alongside", "Ordinal scales are ranks, not exact measurements", df)
df[[c + "_Label" for c in ordinal_maps]].head()
```

**Step: skew and outliers in MonthlyIncome.** Pay is usually right-skewed (a few very high earners pull the average up). We measure the skew and use the IQR rule to count extreme values.


```python
inc = df["MonthlyIncome"]
skew_raw = inc.skew()
q1, q3 = inc.quantile([0.25, 0.75])
iqr = q3 - q1
low_f, high_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["Flag_IncomeOutlier"] = (inc < low_f) | (inc > high_f)
df["LogMonthlyIncome"] = np.log(inc.where(inc > 0))
skew_log = df["LogMonthlyIncome"].skew()
n_out = int(df["Flag_IncomeOutlier"].sum())

print(f"Skewness of MonthlyIncome     : {skew_raw:.2f}")
print(f"Skewness of log(MonthlyIncome): {skew_log:.2f}")
print(f"IQR fence: {low_f:,.0f} to {high_f:,.0f}  ->  {n_out} high/low outliers")

log_step(7, "Skew and outliers in MonthlyIncome",
         f"Skew {skew_raw:.2f} (log: {skew_log:.2f}); {n_out} IQR outliers",
         "Rows kept and flagged (Flag_IncomeOutlier); added LogMonthlyIncome",
         "High earners are real senior staff in this data, not errors; the log scale tames skew for tests and charts", df)
```

**Step: pay-rate fields.** D2 asked us to check how DailyRate, HourlyRate and MonthlyRate relate to MonthlyIncome before keeping or dropping them.


```python
rate_cols = ["DailyRate", "HourlyRate", "MonthlyRate"]
rate_corr = df[rate_cols + ["MonthlyIncome"]].corr()["MonthlyIncome"].drop("MonthlyIncome")
print(rate_corr.round(3))

weak = [c for c in rate_cols if abs(rate_corr[c]) < 0.10]
MODEL_EXCLUDE = weak
log_step(8, "DailyRate / HourlyRate / MonthlyRate vs MonthlyIncome",
         "Correlations: " + ", ".join(f"{c}={rate_corr[c]:.2f}" for c in rate_cols),
         f"Marked for exclusion from modelling (|r|<0.10): {weak}" if weak else "Kept for review",
         "Fields that do not track income add noise and can confuse readers; kept in the table, excluded later if flagged", df)
```

**Step: analysis groups.** `JobSat_Group` (High = 3-4, Low = 1-2) is the target for the t-test, chi-square and the later Decision Tree. `AgeBand` is for the fairness check (cut-offs from the D2 plan; confirmed against the age spread below).


```python
df["JobSat_Group"] = pd.Categorical(np.where(df["JobSatisfaction"] >= 3, "High", "Low"),
                                    categories=["Low", "High"], ordered=True)
df["AgeBand"] = pd.cut(df["Age"], bins=[0, 29, 44, 200], labels=["Under 30", "30-44", "45 and over"])

print("Age range:", df.Age.min(), "to", df.Age.max())
balance = df["JobSat_Group"].value_counts().to_frame("Employees")
balance["Percent"] = (balance["Employees"] / len(df) * 100).round(1)
print(balance)
print()
print(df["AgeBand"].value_counts().sort_index())

minority = balance["Percent"].min()
log_step(9, "Class balance of JobSat_Group",
         f"Smaller class is {minority:.1f}% of rows",
         "Recorded; for the Decision Tree (D4) we will report recall for the Low group, not accuracy alone",
         "If one class is rare, accuracy alone hides poor performance", df)
```

### Cleaning log


```python
log_df = pd.DataFrame(cleaning_log)
pd.set_option("display.max_colwidth", 120)
log_df
```


```python
df_clean = df.copy()
print("Raw   :", df_raw.shape)
print("Clean :", df_clean.shape)

# Save cleaned table next to the raw file (needed again in D4 for the models)
try:
    df_clean.to_csv(CLEAN_PATH, index=False)
    log_df.to_csv(CLEAN_PATH.replace(".csv", "_cleaning_log.csv"), index=False)
    print("Saved:", CLEAN_PATH)
except Exception as e:
    print("Could not save (check Volume path / permissions):", e)
```

## 2.4 Visualizing distributions and relationships for business reporting – Univariate and Multivariate

Nine charts follow. Each has a plain-English "what to look for" note. Group sizes are shown wherever groups could be small. Patterns shown are **associations** in simulated data, not causes.

**Chart 1 – How satisfied are employees overall?** (Univariate)


```python
fig, ax = plt.subplots(figsize=(8, 5))
order = list(df["JobSatisfaction_Label"].cat.categories)
sns.countplot(data=df, x="JobSatisfaction_Label", order=order, ax=ax, color="#4C72B0")
total = len(df)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.0f}\n({p.get_height()/total*100:.0f}%)",
                (p.get_x() + p.get_width()/2, p.get_height()), ha="center", va="bottom", fontsize=9)
ax.set_title("Chart 1: Job satisfaction – number of employees at each level")
ax.set_xlabel("Job satisfaction (ordered 1 Low to 4 Very High)")
ax.set_ylabel("Employees")
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
plt.tight_layout(); plt.show()
```

**Chart 2 – Shape of monthly income, raw and log scale** (Univariate). *Look for:* a long right tail on the left panel; the log panel should look more symmetric. This is why tests use the log scale.


```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df["MonthlyIncome"], kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Monthly income (as recorded)")
axes[0].set_xlabel("Monthly income (currency not stated)")
sns.histplot(df["LogMonthlyIncome"], kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Monthly income (log scale)")
axes[1].set_xlabel("log(Monthly income)")
fig.suptitle("Chart 2: Distribution of monthly income", y=1.02)
plt.tight_layout(); plt.show()
```

**Chart 3 – Income by job level** (Multivariate). *Look for:* whether income steps up with level, and how much levels overlap.


```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="JobLevel", y="MonthlyIncome", ax=ax, color="#8172B2")
counts = df["JobLevel"].value_counts().sort_index()
ax.set_xticks(range(len(counts)))
ax.set_xticklabels([f"Level {i}\n(n={n})" for i, n in counts.items()])
ax.set_title("Chart 3: Monthly income by job level")
ax.set_xlabel("Job level"); ax.set_ylabel("Monthly income")
plt.tight_layout(); plt.show()
```

**Chart 4 – Income for High vs Low satisfaction** (Multivariate; this is the picture behind the t-test). *Look for:* whether the two boxes sit at a similar height.


```python
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="JobSat_Group", y="MonthlyIncome", order=["Low", "High"], ax=ax,
            palette=["#C44E52", "#55A868"], hue="JobSat_Group", legend=False)
n = df["JobSat_Group"].value_counts()
ax.set_xticks([0, 1])
ax.set_xticklabels([f"Low (1-2)\nn={n['Low']}", f"High (3-4)\nn={n['High']}"])
ax.set_title("Chart 4: Monthly income by satisfaction group")
ax.set_xlabel("Satisfaction group"); ax.set_ylabel("Monthly income")
plt.tight_layout(); plt.show()
```

**Chart 5 – Share of employees with High satisfaction, by overtime and by work-life balance** (Multivariate; behind the chi-square test). *Look for:* bars that differ clearly from the overall share (dashed line).


```python
overall = (df["JobSat_Group"] == "High").mean() * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ot = df.groupby("OverTime", observed=True)["JobSat_Group"].apply(lambda s: (s == "High").mean() * 100)
ot_n = df["OverTime"].value_counts()
axes[0].bar([f"{k}\n(n={ot_n[k]})" for k in ot.index], ot.values, color="#4C72B0")
axes[0].set_title("By overtime"); axes[0].set_ylabel("% with High satisfaction")

wl = df.groupby("WorkLifeBalance_Label", observed=True)["JobSat_Group"].apply(lambda s: (s == "High").mean() * 100)
wl_n = df["WorkLifeBalance_Label"].value_counts()
axes[1].bar([f"{k}\n(n={wl_n[k]})" for k in wl.index], wl.values, color="#55A868")
axes[1].set_title("By work-life balance")

for ax, ser in zip(axes, [ot, wl]):
    ax.axhline(overall, ls="--", color="grey")
    for i, v in enumerate(ser.values):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
fig.suptitle(f"Chart 5: Share with High satisfaction (dashed line = overall {overall:.0f}%)", y=1.02)
plt.tight_layout(); plt.show()
```

**Chart 6 – Experience vs income, coloured by job level** (Multivariate). *Look for:* the upward trend, and whether job level explains the spread at the same experience.


```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="TotalWorkingYears", y="MonthlyIncome", hue="JobLevel", palette="viridis", alpha=0.7, ax=ax)
ax.set_title("Chart 6: Total working years vs monthly income")
ax.set_xlabel("Total working years"); ax.set_ylabel("Monthly income")
plt.tight_layout(); plt.show()
```

**Chart 7 – Correlation heatmap** (Multivariate). *Look for:* strong pairs (dark colours), such as income with job level and experience. Strong correlation between predictors is a warning for the later models.


```python
num = df.select_dtypes(include="number").drop(columns=["Flag_Inconsistent", "Flag_IncomeOutlier"], errors="ignore")
correlation_matrix = num.corr()

plt.figure(figsize=(14, 11))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True,
            linewidths=0.5, annot_kws={"size": 7})
plt.title("Chart 7: Correlation heatmap of numeric fields")
plt.tight_layout(); plt.show()
```


```python
# Top correlations with the two targets, in plain table form
targets = ["MonthlyIncome", "JobSatisfaction"]
for t in targets:
    print(f"\nStrongest correlations with {t}:")
    print(correlation_matrix[t].drop(labels=[t, "LogMonthlyIncome"], errors="ignore")
          .sort_values(key=abs, ascending=False).head(6).round(3))
```

**Chart 8 – Income by department** (Multivariate; the HR group is small, so read it carefully).


```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="Department", y="MonthlyIncome", ax=ax, color="#CCB974")
dn = df["Department"].value_counts()
labels = [t.get_text() for t in ax.get_xticklabels()]
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f"{l}\n(n={dn[l]})" for l in labels])
ax.set_title("Chart 8: Monthly income by department")
ax.set_xlabel(""); ax.set_ylabel("Monthly income")
plt.tight_layout(); plt.show()
```

**Chart 9 – Fairness view: median income by gender within each job level** (Multivariate; see also section 2.7). *Look for:* whether the two bars differ at the *same* level, which is a fairer comparison than averages across everyone.


```python
g = (df.groupby(["JobLevel", "Gender"], observed=True)["MonthlyIncome"]
       .agg(median="median", n="count").reset_index())
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=g, x="JobLevel", y="median", hue="Gender", ax=ax)
for c, (_, grp) in zip(ax.containers, g.groupby("Gender", observed=True)):
    ax.bar_label(c, labels=[f"n={n}" for n in grp.sort_values("JobLevel")["n"]], fontsize=8, padding=2)
ax.set_title("Chart 9: Median monthly income by job level and gender (n shown on bars)")
ax.set_xlabel("Job level"); ax.set_ylabel("Median monthly income")
plt.tight_layout(); plt.show()
```

## 2.5 Transforming the data through standard scaling for machine learning

K-Means (planned for D4) measures distances, so a field in thousands (income) would swamp a field in single digits (years). **Standard scaling** puts every numeric field on the same footing (mean 0, standard deviation 1). We create a scaled *copy*; the statistical tests in 2.6 use the original units (or the log scale) so results stay readable.


```python
from sklearn.preprocessing import StandardScaler

scale_cols = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
              "YearsSinceLastPromotion", "YearsWithCurrManager", "JobLevel", "JobSatisfaction",
              "NumCompaniesWorked", "DistanceFromHome", "PercentSalaryHike", "TrainingTimesLastYear"]
scale_cols = [c for c in scale_cols if c not in MODEL_EXCLUDE]

scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[scale_cols] = scaler.fit_transform(df_clean[scale_cols])

# Check: every scaled column should have mean ~0 and std ~1
df_scaled[scale_cols].agg(["mean", "std"]).round(2)
```

## 2.6 Statistical tests, explained in plain English

All tests use a **5% significance level** (α = 0.05). In plain terms: we only call a difference "statistically significant" if, *were there really no difference in the wider population*, a gap this large would show up by chance less than 5 times in 100.

**Two cautions for every test:**
- A small p-value means the pattern is unlikely to be luck. It does **not** mean it is large, important or caused by the factor. We therefore also report an **effect size**.
- The data is simulated, so results show the *method*, not facts about a real company.


```python
results = []   # collects one row per test for the summary table at the end

def strength_d(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"

def strength_v(v):
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"
```

### Test 1 – t-test: do High- and Low-satisfaction employees earn different pay?

**Question (Q3 in D2).** Is average pay different between employees who rate satisfaction High (3-4) and Low (1-2)?

**What a t-test does, in plain English.** It compares the *averages* of two groups and asks whether the gap is bigger than what random variation would produce. We use **Welch's version**, which does not assume the two groups have equal spread, and we run it on **log income**, because raw income is skewed. A gap on the log scale roughly equals a percentage difference in pay.

- **H0 (null):** average pay is the same in the High and Low groups.
- **H1 (alternative):** average pay differs between the groups.
- **How to read it:** p below 0.05 → reject H0 (the gap is unlikely to be chance). p at or above 0.05 → we cannot rule out chance. *Not* the same as "proved equal".


```python
hi = df_clean.loc[df_clean.JobSat_Group == "High", "LogMonthlyIncome"].dropna()
lo = df_clean.loc[df_clean.JobSat_Group == "Low",  "LogMonthlyIncome"].dropna()

t_stat, p_t = stats.ttest_ind(hi, lo, equal_var=False)          # Welch's t-test

# Effect size: Cohen's d (gap measured in standard deviations)
pooled_sd = np.sqrt(((len(hi)-1)*hi.var() + (len(lo)-1)*lo.var()) / (len(hi)+len(lo)-2))
d = (hi.mean() - lo.mean()) / pooled_sd

# 95% confidence interval for the gap on the log scale, then as % difference in pay
se = np.sqrt(hi.var()/len(hi) + lo.var()/len(lo))
dfree = se**4 / ((hi.var()/len(hi))**2/(len(hi)-1) + (lo.var()/len(lo))**2/(len(lo)-1))
tcrit = stats.t.ppf(0.975, dfree)
diff = hi.mean() - lo.mean()
ci_pct = (np.exp(diff - tcrit*se) - 1) * 100, (np.exp(diff + tcrit*se) - 1) * 100
pct_gap = (np.exp(diff) - 1) * 100

# Robustness check that does not assume a bell-shaped distribution
u_stat, p_u = stats.mannwhitneyu(
    df_clean.loc[df_clean.JobSat_Group == "High", "MonthlyIncome"],
    df_clean.loc[df_clean.JobSat_Group == "Low",  "MonthlyIncome"], alternative="two-sided")

raw_tbl = df_clean.groupby("JobSat_Group", observed=True)["MonthlyIncome"].agg(["count", "mean", "median"]).round(0)
print(raw_tbl, "\n")
print(f"Welch t = {t_stat:.3f}, p = {p_t:.4f}")
print(f"Cohen's d = {d:.3f} ({strength_d(d)}); typical pay gap High vs Low = {pct_gap:+.1f}% (95% CI {ci_pct[0]:+.1f}% to {ci_pct[1]:+.1f}%)")
print(f"Robustness – Mann-Whitney U p = {p_u:.4f}")

sig = p_t < ALPHA
print("\nPLAIN ENGLISH")
print(f"- Typical pay of High-satisfaction employees is {abs(pct_gap):.1f}% {'higher' if pct_gap >= 0 else 'lower'} than Low-satisfaction employees "
      f"(95% range {ci_pct[0]:+.1f}% to {ci_pct[1]:+.1f}%).")
print(f"- The difference is {'statistically significant' if sig else 'NOT statistically significant'} at 5% (p = {p_t:.3f}), "
      f"so it is {'unlikely' if sig else 'quite possibly'} due to chance.")
print(f"- Its size is {strength_d(d)}, so pay {'differs a little' if abs(d)<0.5 and sig else 'differs meaningfully' if sig else 'does not clearly differ'} between the groups.")
print(f"- The rank-based check agrees: {'yes' if (p_u < ALPHA) == sig else 'NO – treat the result with caution'}.")
print("- This shows association only; it does not say pay causes satisfaction or the reverse.")

results.append({"Test": "Welch t-test: log income, High vs Low satisfaction", "Statistic": round(t_stat, 3),
                "p_value": round(p_t, 4), "Effect_size": f"d = {d:.2f} ({strength_d(d)})",
                "Significant_at_5pct": sig})
```

### Test 2 – chi-square: is overtime linked to satisfaction?

**Question (Q1 in D2).** Are High and Low satisfaction spread differently between employees who do and do not work overtime?

**What a chi-square test does, in plain English.** Both fields are *categories*, so we cannot compare averages. Instead we count how many people fall in each box of a 2×2 table (overtime × satisfaction) and compare those counts with what we would *expect* if overtime and satisfaction were unrelated. A big gap between actual and expected counts gives a small p-value.

- **H0:** overtime and satisfaction group are independent (no association).
- **H1:** they are associated.
- **Effect size:** Cramér's V, from 0 (no link) to 1 (perfect link).
- **Check:** every expected count should be at least 5 for the test to be reliable.


```python
ct = pd.crosstab(df_clean["OverTime"], df_clean["JobSat_Group"])
chi2, p_c, dof, expected = stats.chi2_contingency(ct)
n_obs = ct.values.sum()
v = np.sqrt(chi2 / (n_obs * (min(ct.shape) - 1)))
min_exp = expected.min()

print("Observed counts:\n", ct, "\n")
print("Share with High satisfaction (%):")
print((ct["High"] / ct.sum(axis=1) * 100).round(1), "\n")
print(f"Chi-square = {chi2:.3f}, df = {dof}, p = {p_c:.4f}")
print(f"Cramér's V = {v:.3f} ({strength_v(v)}); smallest expected count = {min_exp:.1f} ({'OK' if min_exp >= 5 else 'TOO SMALL – interpret with care'})")

sig = p_c < ALPHA
hi_share = (ct["High"] / ct.sum(axis=1) * 100)
print("\nPLAIN ENGLISH")
print(f"- {hi_share.get('Yes', float('nan')):.0f}% of overtime workers have High satisfaction, against {hi_share.get('No', float('nan')):.0f}% of those who do not work overtime.")
print(f"- The difference is {'statistically significant' if sig else 'NOT statistically significant'} at 5% (p = {p_c:.3f}).")
print(f"- The strength of the link is {strength_v(v)} (Cramér's V = {v:.2f}).")
print("- Overtime IS associated with satisfaction in this data; that does not show overtime causes the difference." if sig else
      "- We found no reliable link between overtime and satisfaction in this data (absence of evidence is not proof of no link).")

results.append({"Test": "Chi-square: OverTime x Satisfaction group", "Statistic": round(chi2, 3),
                "p_value": round(p_c, 4), "Effect_size": f"V = {v:.2f} ({strength_v(v)})",
                "Significant_at_5pct": sig})
```

### Test 3 – chi-square: is work-life balance linked to satisfaction?

Same method as Test 2, but with a 4-level field (Bad / Good / Better / Best) against the High/Low satisfaction group. It answers Q1 for a second workplace factor.

- **H0:** work-life balance rating and satisfaction group are independent.
- **H1:** they are associated.


```python
ct2 = pd.crosstab(df_clean["WorkLifeBalance_Label"], df_clean["JobSat_Group"])
chi2b, p_c2, dof2, exp2 = stats.chi2_contingency(ct2)
v2 = np.sqrt(chi2b / (ct2.values.sum() * (min(ct2.shape) - 1)))

print(ct2, "\n")
print((ct2["High"] / ct2.sum(axis=1) * 100).round(1).rename("% High satisfaction"), "\n")
print(f"Chi-square = {chi2b:.3f}, df = {dof2}, p = {p_c2:.4f}, Cramér's V = {v2:.3f} ({strength_v(v2)})")
print(f"Smallest expected count = {exp2.min():.1f} ({'OK' if exp2.min() >= 5 else 'TOO SMALL – interpret with care'})")

sig = p_c2 < ALPHA
print("\nPLAIN ENGLISH")
print(f"- Work-life balance rating and satisfaction are {'statistically linked' if sig else 'NOT clearly linked'} at 5% (p = {p_c2:.3f}), and the link is {strength_v(v2)}.")
print("- Association only; it does not show that improving balance would raise satisfaction." if sig else
      "- No reliable link found in this data.")

results.append({"Test": "Chi-square: WorkLifeBalance x Satisfaction group", "Statistic": round(chi2b, 3),
                "p_value": round(p_c2, 4), "Effect_size": f"V = {v2:.2f} ({strength_v(v2)})",
                "Significant_at_5pct": sig})
```

### Test 4 – t-test: do men and women earn different pay?

**Question (fairness check from D2 Section 7).** Is average pay different by gender?

This uses the same Welch t-test as Test 1. A raw gap can come from *who holds which job* (for example, more men in senior roles) rather than from unequal pay for the same job, so Section 2.7 repeats the comparison **within job level**.

- **H0:** average pay is the same for men and women.
- **H1:** average pay differs.


```python
m = df_clean.loc[df_clean.Gender == "Male",   "LogMonthlyIncome"].dropna()
f = df_clean.loc[df_clean.Gender == "Female", "LogMonthlyIncome"].dropna()
t4, p4 = stats.ttest_ind(m, f, equal_var=False)
sd4 = np.sqrt(((len(m)-1)*m.var() + (len(f)-1)*f.var()) / (len(m)+len(f)-2))
d4 = (m.mean() - f.mean()) / sd4
gap4 = (np.exp(m.mean() - f.mean()) - 1) * 100

print(df_clean.groupby("Gender", observed=True)["MonthlyIncome"].agg(["count", "mean", "median"]).round(0), "\n")
print(f"Welch t = {t4:.3f}, p = {p4:.4f}, Cohen's d = {d4:.3f} ({strength_d(d4)})")

sig = p4 < ALPHA
print("\nPLAIN ENGLISH")
print(f"- Men's typical pay is {gap4:+.1f}% relative to women's across all employees.")
print(f"- This gap is {'statistically significant' if sig else 'NOT statistically significant'} at 5% (p = {p4:.3f}) and {strength_d(d4)} in size.")
print("- Not yet a verdict on fairness: see 2.7, which compares within the same job level.")

results.append({"Test": "Welch t-test: log income, Male vs Female", "Statistic": round(t4, 3),
                "p_value": round(p4, 4), "Effect_size": f"d = {d4:.2f} ({strength_d(d4)})",
                "Significant_at_5pct": sig})
```

## 2.7 Fairness check (from D2 Section 7.2)

D2 promised to (a) compare income and High-satisfaction share by gender and age band, (b) report group sizes, and (c) say whether any gap remains after controlling for job level and experience. Groups with fewer than 30 people are marked **small** and should not support strong claims.


```python
def group_table(d, by):
    t = d.groupby(by, observed=True).agg(
        n=("MonthlyIncome", "size"),
        mean_income=("MonthlyIncome", "mean"),
        median_income=("MonthlyIncome", "median"),
        pct_high_satisfaction=("JobSat_Group", lambda s: (s == "High").mean() * 100),
    ).round(1)
    t["note"] = np.where(t["n"] < 30, "small group", "")
    return t

print("By gender");     display_g = group_table(df_clean, "Gender");   print(display_g, "\n")
print("By age band");   display_a = group_table(df_clean, "AgeBand");  print(display_a, "\n")
if df_clean["MaritalStatus"].value_counts().min() >= 30:
    print("By marital status"); print(group_table(df_clean, "MaritalStatus"))
```

**Does a gender gap remain at the same job level?** For each level we compare median income of men and women and test it with a Mann-Whitney test (suitable for small and skewed groups). With five levels we run five tests, so we also apply a **Bonferroni correction** (the 5% threshold is divided by the number of tests) to avoid false alarms from multiple testing.


```python
rows = []
for lvl, sub in df_clean.groupby("JobLevel"):
    a = sub.loc[sub.Gender == "Male",   "MonthlyIncome"]
    b = sub.loc[sub.Gender == "Female", "MonthlyIncome"]
    if len(a) >= 5 and len(b) >= 5:
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    else:
        p = np.nan
    rows.append({"JobLevel": lvl, "n_male": len(a), "n_female": len(b),
                 "median_male": a.median(), "median_female": b.median(),
                 "gap_pct_male_vs_female": (a.median() / b.median() - 1) * 100 if len(b) else np.nan,
                 "p_value": p})
within = pd.DataFrame(rows).round(3)
k = within["p_value"].notna().sum()
bonf = ALPHA / max(k, 1)
within["significant_after_bonferroni"] = within["p_value"] < bonf
print(f"Bonferroni threshold = {bonf:.4f} ({k} tests)")
within
```


```python
n_sig = int(within["significant_after_bonferroni"].sum())
print("PLAIN ENGLISH")
print(f"- Overall gender gap (Test 4): {gap4:+.1f}%, p = {p4:.3f}.")
print(f"- Within the same job level, {n_sig} of {k} levels {'shows' if n_sig == 1 else 'show'} a gender gap that survives the stricter threshold.")
print("- If the overall gap is large but disappears within levels, it is mostly about *who holds which level*, not unequal pay within a level. "
      "If gaps persist within levels, flag them for HR review with the group sizes above.")
```

# 3. Summary of results, limitations and next steps


```python
summary = pd.DataFrame(results)
summary
```

**How to report this (fill in from the tables above once the notebook has run on the real file):**

1. *Data quality* – state from the cleaning log how many rows/columns were removed, how many rows were flagged, and whether the shape matched the documentation.
2. *Satisfaction* – which factors Tests 2 and 3 and Charts 5 and 7 associate with High satisfaction, and how strong the links are.
3. *Pay* – what Charts 3, 6 and 7 show about job level and experience, and what Test 1 says about income by satisfaction.
4. *Fairness* – whether the gap in Test 4 remains within job level (section 2.7), with group sizes.

**Limitations (carried over from D2):**
- Simulated data: results describe the method, not a real company or country.
- No developer role: findings apply to *employees* in three departments (Sales, R&D, HR).
- Single snapshot: association, not cause.
- Pay currency, country and year are unstated, so pay is relative only.
- Satisfaction is a self-reported 1-4 scale; people may use scales differently.
- Several tests were run; p-values near 0.05 should be treated as suggestive.

**Next (D4, due 13 November 2026):** Decision Tree on `JobSat_Group` (train/test split with `SEED`, report recall for the Low group, and accuracy/precision/recall by gender and age band) and K-Means on `df_scaled` (small number of clusters, profiled using the original-unit table `df_clean`). The cleaned table is saved to `CLEAN_PATH` for reuse.
