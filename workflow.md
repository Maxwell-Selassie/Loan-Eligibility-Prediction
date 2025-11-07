This is a loan eligibility prediction model to predict an individual's eligibility for a loan.
Workflow:
- Exploratory data analysis
(Dataset overview, numeric and categorical values, outliers, missing values, duplicates, visualizations)
- Preprocessing
(Handling outliers, handling missing values, business sanity checks, scaling, encoding)
- Feature engineering
(Interation features,ratio features, polynomial features, log features, time-bound features)
- Model training
(Feature selection, mlflow integration, model training and hyperparameter tuning)
- Model evaluation
(metrics, error analysis, cross validation)
- Model serving
(fastApi, streamlit)
- Automation
(docker, apache-airflow)


# Complete Guide to Statistical Hypothesis Testing
## From First Principles to Practical Application

---

## Part 1: Foundation Concepts

### 1.1 What is a Confidence Interval?

**Plain English**: A confidence interval gives you a range where you believe the true population value lies, based on your sample data.

**Example**: 
- You sample 100 loan applicants and find their average income is $50,000
- A 95% confidence interval might be [$47,500, $52,500]
- **Interpretation**: "I'm 95% confident the true average income of ALL applicants (not just my sample) is between $47,500 and $52,500"

**The Math Behind It**:

```
CI = Sample Mean ± (Critical Value × Standard Error)

Where:
- Sample Mean (x̄) = sum of all values / n
- Standard Error (SE) = s / √n
  - s = sample standard deviation
  - n = sample size
- Critical Value = from t-distribution table (usually 1.96 for 95% CI with large samples)
```

**Example Calculation**:
```
Sample: [45k, 52k, 48k, 55k, 50k] (5 applicants)

Step 1: Calculate mean
x̄ = (45 + 52 + 48 + 55 + 50) / 5 = 50k

Step 2: Calculate standard deviation
Deviations: [-5, 2, -2, 5, 0]
Squared: [25, 4, 4, 25, 0]
Variance = 58/4 = 14.5
s = √14.5 = 3.81k

Step 3: Calculate SE
SE = 3.81 / √5 = 1.70k

Step 4: Find critical value (t-table, df=4, 95% CI)
t = 2.776

Step 5: Calculate CI
CI = 50 ± (2.776 × 1.70)
CI = 50 ± 4.72
CI = [45.28k, 54.72k]
```

**Key Insight**: 
- Larger sample size → smaller SE → narrower CI → more precise estimate
- 95% means: if we repeated this sampling 100 times, 95 CIs would contain the true mean

---

## Part 2: Hypothesis Testing Framework

### 2.1 The Core Logic

**The Setup**:
1. **Null Hypothesis (H₀)**: The "boring" assumption (no difference, no effect)
2. **Alternative Hypothesis (H₁)**: What you're trying to prove
3. **Test Statistic**: A number that measures how far your data is from H₀
4. **P-value**: Probability of seeing your data (or more extreme) IF H₀ is true

**Decision Rule**:
- If p-value < 0.05 → Reject H₀ → "Statistically significant"
- If p-value ≥ 0.05 → Fail to reject H₀ → "Not statistically significant"

**Analogy**: Criminal trial
- H₀ = "Defendant is innocent"
- H₁ = "Defendant is guilty"
- Evidence = Your data
- P-value = How likely this evidence would exist if they were innocent
- If p < 0.05 → Evidence too unlikely under innocence → Reject H₀ (convict)

---

## Part 3: T-Test (Comparing Two Means)

### 3.1 When to Use It
**Question**: "Is the average income of approved applicants different from rejected applicants?"

### 3.2 The Hypotheses
```
H₀: μ_approved = μ_rejected (no difference in means)
H₁: μ_approved ≠ μ_rejected (there IS a difference)
```

### 3.3 The Formula

**Independent T-Test**:
```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)

Where:
- x̄₁, x̄₂ = means of group 1 and group 2
- s₁², s₂² = variances of group 1 and group 2
- n₁, n₂ = sample sizes
```

### 3.4 Worked Example

**Scenario**: Income of approved vs rejected applicants

```
Approved Group (n=5): [60k, 65k, 58k, 62k, 61k]
Rejected Group (n=5): [45k, 48k, 50k, 47k, 46k]

Step 1: Calculate means
x̄₁ = (60 + 65 + 58 + 62 + 61) / 5 = 61.2k
x̄₂ = (45 + 48 + 50 + 47 + 46) / 5 = 47.2k
Difference = 14k

Step 2: Calculate variances
s₁² = 6.7 (variance of approved group)
s₂² = 3.7 (variance of rejected group)

Step 3: Calculate t-statistic
SE = √(6.7/5 + 3.7/5) = √2.08 = 1.44
t = (61.2 - 47.2) / 1.44 = 9.72

Step 4: Find p-value
With df = 8, t = 9.72 → p-value < 0.001
```

### 3.5 Interpretation

**What the numbers mean**:
- **t = 9.72**: The means are 9.72 standard errors apart (HUGE!)
- **p < 0.001**: If there were truly no difference, we'd see this extreme data less than 0.1% of the time
- **Conclusion**: Reject H₀. There IS a significant difference in income between approved and rejected applicants

**Business Meaning**: Income is a strong predictor of loan approval. Approved applicants earn ~$14k more on average.

### 3.6 Assumptions & Alternatives

**T-Test Assumptions**:
1. Data is approximately normally distributed
2. Independent observations

**If assumptions violated → Use Mann-Whitney U Test** (non-parametric alternative)
- Compares medians instead of means
- Ranks the data instead of using raw values
- More robust to outliers and non-normal distributions

---

## Part 4: Chi-Square Test (Categorical Associations)

### 4.1 When to Use It
**Question**: "Is gender associated with loan approval status?"

### 4.2 The Hypotheses
```
H₀: Gender and loan approval are independent (no association)
H₁: Gender and loan approval are associated
```

### 4.3 The Logic

**Contingency Table** (Observed counts):
```
                Approved    Rejected    Total
Male               120         80        200
Female             100        100        200
Total              220        180        400
```

**Expected Counts** (if H₀ is true):
```
Expected = (Row Total × Column Total) / Grand Total

Male-Approved = (200 × 220) / 400 = 110
Male-Rejected = (200 × 180) / 400 = 90
Female-Approved = (200 × 220) / 400 = 110
Female-Rejected = (200 × 180) / 400 = 90
```

### 4.4 The Formula

```
χ² = Σ [(Observed - Expected)² / Expected]

For each cell:
Male-Approved: (120 - 110)² / 110 = 0.91
Male-Rejected: (80 - 90)² / 90 = 1.11
Female-Approved: (100 - 110)² / 110 = 0.91
Female-Rejected: (100 - 90)² / 90 = 1.11

χ² = 0.91 + 1.11 + 0.91 + 1.11 = 4.04
```

### 4.5 Worked Example

```
Step 1: Calculate χ²
χ² = 4.04

Step 2: Find degrees of freedom
df = (rows - 1) × (columns - 1) = (2-1) × (2-1) = 1

Step 3: Find p-value
With df=1, χ²=4.04 → p-value = 0.044
```

### 4.6 Interpretation

**What the numbers mean**:
- **χ² = 4.04**: Measures how far observed counts are from expected
- **p = 0.044 < 0.05**: Statistically significant
- **Conclusion**: Gender and loan approval ARE associated

**Business Meaning**: Males have higher approval rate (60%) than females (50%). This could indicate bias and needs investigation.

### 4.7 Effect Size: Cramér's V

```
V = √[χ² / (n × min(rows-1, cols-1))]
V = √[4.04 / (400 × 1)] = 0.10
```

**Interpretation**:
- V = 0.10 → Small effect size
- Statistically significant but weak association
- **Key Learning**: Statistical significance ≠ practical importance

---

## Part 5: ANOVA (Comparing Multiple Groups)

### 5.1 When to Use It
**Question**: "Does average income differ across education levels (High School, Bachelor's, Master's)?"

### 5.2 The Hypotheses
```
H₀: μ_HS = μ_Bach = μ_Mast (all means are equal)
H₁: At least one mean is different
```

### 5.3 The Logic

**ANOVA partitions variance**:
```
Total Variance = Between-Group Variance + Within-Group Variance

If groups truly different → Between-Group Variance is large
If groups same → Between-Group Variance is small
```

### 5.4 The Formula

```
F = (Between-Group Variance) / (Within-Group Variance)

Between-Group Variance (MSB):
MSB = Σ[n_i × (x̄_i - x̄_grand)²] / (k - 1)

Within-Group Variance (MSW):
MSW = Σ Σ(x_ij - x̄_i)² / (N - k)

Where:
- k = number of groups
- n_i = size of group i
- N = total sample size
- x̄_grand = overall mean
```

### 5.5 Worked Example

**Data**:
```
High School (n=4): [35k, 38k, 36k, 37k] → Mean = 36.5k
Bachelor's (n=4): [50k, 52k, 48k, 54k] → Mean = 51k
Master's (n=4): [65k, 68k, 67k, 66k] → Mean = 66.5k

Grand Mean = (36.5 + 51 + 66.5) / 3 = 51.33k
```

**Step 1: Calculate Between-Group Variance**
```
SS_between = 4×(36.5-51.33)² + 4×(51-51.33)² + 4×(66.5-51.33)²
SS_between = 4×219.7 + 4×0.11 + 4×230.1 = 1799.3
MS_between = 1799.3 / 2 = 899.65
```

**Step 2: Calculate Within-Group Variance**
```
SS_within = [(35-36.5)² + (38-36.5)² + ...] (for all groups)
SS_within ≈ 150 (calculated)
MS_within = 150 / 9 = 16.67
```

**Step 3: Calculate F-statistic**
```
F = 899.65 / 16.67 = 53.96
```

**Step 4: Find p-value**
```
With df1=2, df2=9, F=53.96 → p-value < 0.001
```

### 5.6 Interpretation

**What the numbers mean**:
- **F = 53.96**: Between-group variance is 54 times larger than within-group variance
- **p < 0.001**: Extremely unlikely if all groups had same mean
- **Conclusion**: Reject H₀. Income DOES differ significantly across education levels

**Business Meaning**: Education is a strong predictor. Higher education → higher income.

### 5.7 Post-Hoc Tests

**ANOVA only tells you "at least one is different"**. To find WHICH pairs differ, use:
- **Tukey's HSD**: Compares all pairs with adjustment for multiple comparisons
- **Bonferroni**: More conservative, divides α by number of comparisons

---

## Part 6: Proportion Test

### 6.1 When to Use It
**Question**: "Is the approval rate for self-employed (45%) significantly different from salaried (60%)?"

### 6.2 The Hypotheses
```
H₀: p_self = p_salaried (approval rates are equal)
H₁: p_self ≠ p_salaried (approval rates differ)
```

### 6.3 The Formula

**Two-Proportion Z-Test**:
```
z = (p̂₁ - p̂₂) / √[p̂_pooled × (1 - p̂_pooled) × (1/n₁ + 1/n₂)]

Where:
p̂₁, p̂₂ = sample proportions
p̂_pooled = (x₁ + x₂) / (n₁ + n₂)
```

### 6.4 Worked Example

```
Self-Employed: 45 approved out of 100 → p̂₁ = 0.45
Salaried: 120 approved out of 200 → p̂₂ = 0.60

Step 1: Calculate pooled proportion
p̂_pooled = (45 + 120) / (100 + 200) = 165/300 = 0.55

Step 2: Calculate standard error
SE = √[0.55 × 0.45 × (1/100 + 1/200)]
SE = √[0.2475 × 0.015] = √0.003712 = 0.061

Step 3: Calculate z-statistic
z = (0.45 - 0.60) / 0.061 = -0.15 / 0.061 = -2.46

Step 4: Find p-value
z = -2.46 → p-value = 0.014 (two-tailed)
```

### 6.5 Interpretation

**What the numbers mean**:
- **z = -2.46**: The proportions are 2.46 standard errors apart
- **p = 0.014 < 0.05**: Statistically significant
- **Conclusion**: Self-employed have significantly lower approval rate

**Business Meaning**: Employment type matters. Self-employed perceived as higher risk (15% point difference).

---

## Part 7: Effect Size (The Missing Piece)

### 7.1 Why Effect Size Matters

**Problem**: With large samples, even tiny differences become "statistically significant"

**Example**:
- Sample 1 million loans
- Approved income: $50,001
- Rejected income: $50,000
- Difference: $1
- P-value: < 0.05 (significant!)
- **But who cares about $1 difference?**

### 7.2 Cohen's d (For T-Tests)

```
d = (x̄₁ - x̄₂) / s_pooled

Interpretation:
d = 0.2 → Small effect
d = 0.5 → Medium effect
d = 0.8 → Large effect
```

**Example**:
```
Approved income: 61.2k, SD = 2.6k
Rejected income: 47.2k, SD = 1.9k

s_pooled = √[(2.6² + 1.9²) / 2] = 2.27

d = (61.2 - 47.2) / 2.27 = 6.17 → HUGE effect!
```

### 7.3 Cramér's V (For Chi-Square)

```
V = √[χ² / (n × df)]

Interpretation:
V = 0.1 → Small
V = 0.3 → Medium
V = 0.5 → Large
```

---

## Part 8: Common Mistakes & How to Avoid Them

### 8.1 Mistake: Confusing Statistical vs Practical Significance

**Bad**: "p < 0.05, so this is important!"
**Good**: "p < 0.05 AND d = 0.8, so this is both statistically significant and practically meaningful"

### 8.2 Mistake: Multiple Testing Problem

**Problem**: Run 20 tests, expect 1 false positive by chance (0.05 × 20 = 1)

**Solution**: Bonferroni Correction
```
Adjusted α = 0.05 / number_of_tests
If running 10 tests → α = 0.005
```

### 8.3 Mistake: Ignoring Assumptions

**T-Test requires**:
- Normality (check with Shapiro-Wilk test)
- Independence
- If violated → Use Mann-Whitney U

**Chi-Square requires**:
- Expected counts ≥ 5
- If violated → Use Fisher's Exact Test

### 8.4 Mistake: P-Hacking

**Bad**: 
1. Run test → p = 0.07
2. Remove "outliers" → p = 0.048
3. Claim success!

**Good**: Pre-register hypotheses, stick to analysis plan

---

## Part 9: Interpretation Cheat Sheet

### P-Value Interpretation

| P-Value | Interpretation | Action |
|---------|----------------|--------|
| < 0.001 | Very strong evidence against H₀ | Reject H₀ |
| 0.001 - 0.01 | Strong evidence against H₀ | Reject H₀ |
| 0.01 - 0.05 | Moderate evidence against H₀ | Reject H₀ (borderline) |
| 0.05 - 0.10 | Weak evidence against H₀ | Fail to reject H₀ |
| > 0.10 | Little/no evidence against H₀ | Fail to reject H₀ |

### Test Selection Guide

| Question | Test |
|----------|------|
| Compare 2 means (normal data) | Independent T-Test |
| Compare 2 means (non-normal) | Mann-Whitney U |
| Compare 3+ means (normal data) | ANOVA |
| Compare 3+ means (non-normal) | Kruskal-Wallis |
| Compare 2 proportions | Two-Proportion Z-Test |
| Compare 2 categorical variables | Chi-Square Test |
| Paired measurements | Paired T-Test |

---

## Part 10: Real-World Application to Your Loan Data

### Step-by-Step Analysis Plan

**1. Confidence Intervals**
```
For each numeric feature (income, loan_amount, etc.):
- Calculate 95% CI
- Report: "Average income is $50k [95% CI: $48k, $52k]"
- Business use: Set minimum income thresholds
```

**2. T-Tests (or Mann-Whitney)**
```
For each numeric feature:
H₀: Mean_approved = Mean_rejected
- Calculate t-statistic and p-value
- Calculate Cohen's d
- Report: "Approved applicants have significantly higher income 
          (t=9.72, p<0.001, d=2.1 - large effect)"
- Business use: Identify which features discriminate most
```

**3. Chi-Square Tests**
```
For each categorical feature vs loan_status:
H₀: Feature independent of approval
- Calculate χ² and p-value
- Calculate Cramér's V
- Report: "Gender is associated with approval (χ²=4.04, p=0.044, V=0.10)"
- Business use: Check for potential bias
```

**4. ANOVA**
```
For income across education levels:
H₀: All education levels have same mean income
- Calculate F-statistic
- If significant → Run Tukey post-hoc
- Report: "Income differs by education (F=53.96, p<0.001)"
- Business use: Weight education in scoring model
```

**5. Proportion Tests**
```
For approval rates by employment type:
H₀: All employment types have same approval rate
- Calculate z-statistic
- Report: "Self-employed have lower approval rate (45% vs 60%, z=-2.46, p=0.014)"
- Business use: Risk-adjust by employment type
```

### What You'll Learn

From these tests, you'll discover:
1. **Which features matter most** (large effect sizes)
2. **Which relationships are real vs noise** (p-values)
3. **Potential fairness issues** (demographic disparities)
4. **Feature engineering ideas** (interaction effects)
5. **Model validation targets** (expected performance bounds)

---

## Summary

**The Big Picture**:
- **Confidence Intervals**: Quantify uncertainty in your estimates
- **Hypothesis Tests**: Determine if relationships are real or chance
- **Effect Sizes**: Measure practical importance beyond statistics
- **Together**: They give you statistical rigor + business insight

**For Your ML Project**:
These tests help you understand your data deeply BEFORE building models. Features with strong statistical relationships will likely be strong predictors. You're not just throwing features at a model—you're making informed decisions backed by statistical evidence.
