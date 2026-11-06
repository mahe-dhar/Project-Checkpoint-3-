# D3: Clean Data, EDA and Hypothesis Tests

Team 8 – People and HR Analytics · BANA Team Project, PGDM BA 2026

Members: K Krishnakanth Maheedhar, Irene Roby, Nirshitha Vijaya Kumar, Shirley Dutta, Shashanki Shikha, Shradha Shruti

Due: 6 November 2026 · Deliverable: notebook with cleaning log, 5 or more charts and 2 or more statistical tests explained in plain English · Companion file: Team8_D3_Clean_EDA_Hypothesis_Tests.ipynb

# 1. Purpose and scope

This document explains the D3 notebook: what it does, how it is organised, how to run it in Databricks and how to read each result. It builds on the D1 problem statement and the D2 data plan and ethics check.

Question (Option B): which work and career factors are associated with job satisfaction and with monthly income, and do employees with different satisfaction levels differ in pay?

Dataset: IBM HR Analytics Employee Attrition and Performance (Kaggle, Pavan Subhash), 1,470 rows and 35 columns. The source describes it as simulated.

## 1.1 Two points the team should agree on before submission

Employees, not developers. D1 refers to developers, annual compensation and the Stack Overflow survey. D2 correctly notes that the IBM file has no developer role and records monthly income. The notebook follows D2 (unit of analysis: employees; pay measure: MonthlyIncome). The D1 wording should be aligned in the final report.

Faculty approval. D2 Section 5 asks the faculty to confirm the dataset may be used for the Option B question. No approval was recorded in the D2 file, so it should be confirmed before the final report.

# 2. Notebook structure

The notebook follows the numbering of the class reference notebook (Business Analytics – Zero to Hero): Problem Statement, then Exploratory Data Analysis with loading, inspecting, missing values, slicing, visualising, scaling and statistical tests. Cleaning, the fairness check and the summary are added because D2 and the D3 brief require them.

# 3. Running the notebook in Databricks

Download the CSV from Kaggle (free account): WA_Fn-UseC_-HR-Employee-Attrition.csv. Check the licence on the Kaggle page and record it, as D2 requires.

In Databricks, open Catalog and choose or create a catalog and schema. Select Create, then Volume, and name it (for example raw). Choose Upload to this volume and add the CSV.

In Workspace, choose Import and upload Team8_D3_Clean_EDA_Hypothesis_Tests.ipynb. Attach it to a cluster or serverless compute.

In section 2.1 edit DATA_PATH (and CLEAN_PATH) to your Volume path, in the form /Volumes/<catalog>/<schema>/<volume>/<file>.csv. Right-click the file in Catalog and choose Copy path to get it exactly.

Run all cells. If a library is missing, run %pip install seaborn scipy scikit-learn in its own cell and rerun.

Share the notebook with faculty and teammates (Share button). Export a HTML copy (File, Export) as a record of the executed results.

Note: the notebook uses only pandas, numpy, matplotlib, seaborn, scipy and scikit-learn, all included in standard Databricks runtimes. The random seed is fixed at 42.

# 4. Cleaning log

Every cleaning step writes one row to the cleaning_log table (step, check, finding, action, reason, rows and columns remaining). The log is displayed in the notebook and saved beside the cleaned data. The rules below are what the notebook applies; the findings come from the notebook output when it runs on the real file.

# 5. Charts

The notebook has nine charts (the brief asks for five or more). Group sizes appear on charts where groups may be small, and titles avoid causal wording.

# 6. Statistical tests in plain English

All tests use a 5% significance level. In plain terms: a result is called “statistically significant” only if, were there truly no difference in the wider population, a gap this large would appear by chance fewer than 5 times in 100. A small p-value shows that a pattern is unlikely to be luck. It does not show that the pattern is large, important or caused by the factor, so each test also reports an effect size. Each test’s “Plain English” output in the notebook is generated from its own result, so the wording changes if the result changes.

How to read a result. If p is below 0.05, say the difference or link is “statistically significant” and then state its size using the effect size. If p is 0.05 or above, say “we found no reliable difference”, not “there is no difference”. Always add that the finding is an association in a single snapshot of simulated data.

# 7. Fairness check

As promised in D2 Section 7.2, the notebook reports mean and median income and the share with High satisfaction by gender and age band (and marital status where every group has at least 30 people), always with group sizes and a “small group” note below 30. It then tests the gender gap within each job level and says whether it remains after the Bonferroni correction. Sensitive fields (age, gender, marital status) are used only for this check and are not intended as model inputs.

# 8. Ethics, privacy and limitations

Simulated data: no real individuals are described; the DPDP Act principles are still followed as good practice (purpose limitation, data minimisation, no re-identification, group-level reporting only).

EmployeeNumber, the only identifier, is dropped in the cleaning step and no row-level lists are shown.

No developer role: results apply to employees in Sales, Research & Development and Human Resources. An R&D-only view is created in section 2.2.2 as the closest technical group.

Single snapshot: association only. The notebook and report use “associated with”, never “causes”.

Pay currency, country and year are not stated, so pay is relative only.

Self-reported 1–4 scales may be used differently by different people.

Several tests are run; results with p near 0.05 should be treated as suggestive. The within-level fairness tests use a Bonferroni correction.

# 9. Results to insert after running on the real data

The notebook was tested end to end for errors, but the Kaggle file could not be downloaded in the environment where this document was prepared, so no real findings are stated here. After running section 2 on the real file, copy the values from the notebook into this table and into the final report.

# 10. Next steps

Confirm faculty approval and the Kaggle licence (D2 Section 5 and 2).

D4 (13 November 2026): Decision Tree on JobSat_Group with a train/test split and the fixed seed, reporting recall for the Low group and accuracy, precision and recall by gender and age band; K-Means on the scaled table with a small number of clusters, profiled in original units.

D5 (25 November 2026): final report, reproducible notebook, recommendations and ethics section.

# 11. References

Pavan Subhash. IBM HR Analytics Employee Attrition and Performance [dataset]. Kaggle. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Government of India (2023). The Digital Personal Data Protection Act, 2023.

Mehrabi, N., et al. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6).

Chapman, P., et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.

Provost, F., & Fawcett, T. (2013). Data Science for Business. O’Reilly Media.

Class reference notebook: Business Analytics – Zero to Hero (DoctorNDJonathan/bana-class-material, GitHub).

| Section | What it does | Link to reference notebook / D2 |

| --- | --- | --- |

| 1 | Problem statement, decision supported, data and language rule | Reference section 1 |

| 2.1 | Load the CSV from a Databricks Volume into a working copy | Reference 2.1 Loading a dataset |

| 2.2 | Shape, head, info, describe; missing values and duplicates (2.2.1); slicing with .loc/.iloc and conditional tables by department and overtime (2.2.2) | Reference 2.2, 2.2.1, 2.2.2 |

| 2.3 | Cleaning steps, each recorded in a cleaning log; saves the cleaned table | D2 Section 4 Data Quality Plan |

| 2.4 | Nine charts, each with a “what to look for” note | Reference 2.2.3 Visualizing distributions |

| 2.5 | Standard scaling of numeric fields (scaled copy for K-Means) | Reference 2.4 Standard scaling |

| 2.6 | Four hypothesis tests, each with plain-English hypotheses and a generated plain-English reading | Reference 2.5 Statistical tests; D2 Section 1.2 |

| 2.7 | Fairness check by gender and age band, and within job level | D2 Sections 7.1 and 7.2 |

| 3 | Summary table of tests, how to report, limitations, next steps (D4) | D2 Sections 8 and 9 |

| # | Check | Rule / decision | Finding (copy from notebook log) |

| --- | --- | --- | --- |

| 1 | Shape vs documentation | Compare with 1,470 × 35; record any difference |  |

| 2 | Missing values | Count missing cells; no imputation unless found |  |

| 3 | Duplicate rows | Drop exact duplicates |  |

| 4 | Constant and ID fields | Drop EmployeeCount, Over18, StandardHours (constants) and EmployeeNumber (identifier) | 4 columns dropped (35 → 31) if as documented |

| 5 | Impossible values | Flag, do not delete: years at company, in role, with manager or since promotion above logical limits; started work before 14; income ≤ 0; ratings outside 1–4 |  |

| 6 | Types and coding | Category fields to labelled categories; 1–4 scales to ordered labelled categories, numeric code kept |  |

| 7 | Skew and outliers in MonthlyIncome | Measure skew; IQR outliers flagged and kept; add log income for tests |  |

| 8 | Pay-rate fields | Correlate Daily, Hourly and MonthlyRate with MonthlyIncome; mark fields with |r| < 0.10 as excluded from modelling |  |

| 9 | Analysis groups and balance | JobSat_Group: High = 3–4, Low = 1–2; AgeBand: under 30, 30–44, 45 and over; check class balance |  |

| Chart | What it shows | What the reader should look for |

| --- | --- | --- |

| 1 | Count of employees at each satisfaction level | Whether most people are satisfied; how rare Low is (class balance) |

| 2 | Monthly income, raw and log scale | Right-skew in raw income; more symmetric on log scale (why tests use log income) |

| 3 | Income by job level (box plot) | Whether pay steps up with level and how much levels overlap |

| 4 | Income for High vs Low satisfaction (box plot) | Whether the boxes sit at similar heights (the picture behind Test 1) |

| 5 | % High satisfaction by overtime and by work-life balance | Bars far from the overall share (the picture behind Tests 2 and 3) |

| 6 | Total working years vs income, coloured by job level | Upward trend and how much level explains the spread |

| 7 | Correlation heatmap of numeric fields | Strong pairs; overlap among predictors that matters for the models in D4 |

| 8 | Income by department with group sizes | Differences between departments; caution with the small HR group |

| 9 | Median income by job level and gender with group sizes | Whether a gender gap appears at the same level (fairness view) |

| Test | Question it answers | How it works (plain English) | Hypotheses and effect size |

| --- | --- | --- | --- |

| Test 1: Welch t-test | Do High- and Low-satisfaction employees have different average pay? (D2 question 3) | Compares two group averages and asks whether the gap is bigger than random variation would give. Welch’s version does not assume equal spread. Run on log income because raw income is skewed; a log gap reads as a % difference. Mann-Whitney is run as a cross-check. | H0: same average pay. H1: different. Effect size: Cohen’s d (gap in standard deviations), plus a % pay gap with 95% range. |

| Test 2: Chi-square, overtime × satisfaction | Is the share with High satisfaction different for those who do and do not work overtime? (D2 question 1) | Both fields are categories, so it counts people in each cell of a table and compares with the counts expected if the two were unrelated. | H0: independent. H1: associated. Effect size: Cramér’s V (0 = no link, 1 = perfect). Check expected counts ≥ 5. |

| Test 3: Chi-square, work-life balance × satisfaction | Is work-life balance rating linked to High or Low satisfaction? (D2 question 1) | Same method as Test 2 with a four-level field. | As Test 2. |

| Test 4: Welch t-test, gender × income | Is average pay different between men and women? (D2 fairness check) | Same method as Test 1. A raw gap may reflect who holds which job, so section 2.7 repeats the comparison within job level using Mann-Whitney tests with a Bonferroni correction (5% divided by the number of levels tested). | H0: same average pay. H1: different. Effect size: Cohen’s d. |

| Effect size | Negligible | Small to medium | Large |

| --- | --- | --- | --- |

| Cohen’s d (t-tests) | below 0.2 | 0.2 to under 0.8 | 0.8 or more |

| Cramér’s V (chi-square) | below 0.1 | 0.1 to under 0.5 | 0.5 or more |

| Test | Statistic | p-value | Effect size | Plain-English reading |

| --- | --- | --- | --- | --- |

| Test 1: Welch t-test, income by satisfaction group |  |  |  |  |

| Test 2: Chi-square, overtime × satisfaction |  |  |  |  |

| Test 3: Chi-square, work-life balance × satisfaction |  |  |  |  |

| Test 4: Welch t-test, income by gender |  |  |  |  |

| Fairness: gender gap within job levels (levels significant after correction) |  |  |  |  |
