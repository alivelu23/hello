To analyze the relationship between NLI entailment scores, NLI evidence, hallucinations, and truthfulness, follow this step-by-step guide with Python code and explanations:

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings("ignore")
```

### 2. Load Data & Preprocess
```python
# Load your DataFrame (replace with your data path)
df = pd.read_csv("your_data.csv")

# Handle NaN values in 'truthfulness' (drop or treat as a category)
# Option 1: Drop rows with NaN in 'truthfulness'
df = df.dropna(subset=['truthfulness'])

# Option 2: Keep NaN as a separate category (uncomment below)
# df['truthfulness'] = df['truthfulness'].fillna('NaN')

# Ensure categorical variables are properly encoded
df['truthfulness'] = pd.Categorical(
    df['truthfulness'], 
    categories=['Fully True', 'Partially True', 'Not True'], 
    ordered=True
)
```

---

### 3. Relationship Between Entailment Scores and Truthfulness
#### Hypothesis Testing
```python
# Kruskal-Wallis test (non-parametric ANOVA for ordinal data)
groups = [
    df[df['truthfulness'] == 'Fully True']['entailment_score'],
    df[df['truthfulness'] == 'Partially True']['entailment_score'],
    df[df['truthfulness'] == 'Not True']['entailment_score']
]
kw_stat, kw_p = kruskal(*groups)
print(f"Kruskal-Wallis p-value: {kw_p:.4f}")

# Post-hoc pairwise Mann-Whitney U tests with Bonferroni correction
truth_categories = df['truthfulness'].cat.categories
posthoc_results = []
for i in range(len(truth_categories)):
    for j in range(i+1, len(truth_categories)):
        cat1 = truth_categories[i]
        cat2 = truth_categories[j]
        group1 = df[df['truthfulness'] == cat1]['entailment_score']
        group2 = df[df['truthfulness'] == cat2]['entailment_score']
        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
        posthoc_results.append((f"{cat1} vs {cat2}", p))

# Adjust p-values for multiple comparisons
adjusted_ps = [min(p * len(posthoc_results), 1.0) for _, p in posthoc_results]
for (comparison, p), adj_p in zip(posthoc_results, adjusted_ps):
    print(f"{comparison}: raw p = {p:.4f}, adjusted p = {adj_p:.4f}")
```

#### Visualization
```python
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='truthfulness', 
    y='entailment_score', 
    data=df, 
    order=['Fully True', 'Partially True', 'Not True']
)
plt.title("Entailment Scores by Truthfulness Category")
plt.xlabel("Truthfulness")
plt.ylabel("Entailment Score")
plt.show()
```

---

### 4. Relationship Between NLI Evidence and Truthfulness
```python
# Contingency table and Chi-Square test
contingency_table = pd.crosstab(df['NLI_evidence'], df['truthfulness'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square p-value: {p:.4f}")

# Visualization: Stacked bar plot
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title("NLI Evidence vs Truthfulness")
plt.xlabel("Truthfulness")
plt.ylabel("NLI Evidence")
plt.show()
```

---

### 5. Relationship Between Entailment Scores and Hallucinations
```python
# Mann-Whitney U test (hallucination Yes vs No)
group_yes = df[df['hallucination'] == 'Yes']['entailment_score']
group_no = df[df['hallucination'] == 'No']['entailment_score']
mw_stat, mw_p = mannwhitneyu(group_yes, group_no)
print(f"Mann-Whitney U p-value: {mw_p:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
sns.violinplot(x='hallucination', y='entailment_score', data=df)
plt.title("Entailment Scores by Hallucination Status")
plt.xlabel("Hallucination")
plt.ylabel("Entailment Score")
plt.show()
```

---

### 6. Relationship Between NLI Evidence and Hallucinations
```python
# Contingency table and Chi-Square test
contingency_table = pd.crosstab(df['NLI_evidence'], df['hallucination'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square p-value: {p:.4f}")

# Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Reds')
plt.title("NLI Evidence vs Hallucinations")
plt.xlabel("Hallucination")
plt.ylabel("NLI Evidence")
plt.show()
```

---

### Key Interpretations
1. **Entailment Scores & Truthfulness**:
   - High scores should correlate with "Fully True", medium with "Partially True", and low with "Not True".
   - Significant Kruskal-Wallis results indicate differences between groups. Use post-hoc tests to identify which pairs differ.

2. **NLI Evidence (True/False)**:
   - A significant Chi-Square test suggests evidence status is associated with truthfulness/hallucinations.

3. **Hallucinations**:
   - Lower entailment scores and `NLI_evidence = False` are expected for hallucinated summaries.

---

### Documentation Notes
- **Statistical Tests**:
  - Kruskal-Wallis/Mann-Whitney U: Used for non-normal distributions.
  - Chi-Square: Tests independence between categorical variables.
- **Visualizations**:
  - Boxplots show distribution spread.
  - Heatmaps display frequency relationships.
- **Assumptions**:
  - Handle missing data appropriately (e.g., `dropna`).
  - Bonferroni correction reduces false positives in multiple comparisons.

Let me know if you need further refinements!
To find correlations between your variables, we need to handle mixed data types appropriately. Here's how to approach it and which columns to consider:

---

### **Key Columns to Correlate**
1. **`entailment_score`** (continuous: 0-1)
2. **`truthfulness`** (ordinal: "Fully True" > "Partially True" > "Not True")
3. **`hallucination`** (binary: "Yes"/"No")
4. **`NLI_evidence`** (binary: True/False)

---

### **Correlation Strategies by Data Type**
| Pairing                          | Data Types                     | Correlation Method               |
|----------------------------------|--------------------------------|-----------------------------------|
| Entailment Score vs Truthfulness | Continuous vs Ordinal          | Spearman's Rank Correlation      |
| Entailment Score vs Hallucination| Continuous vs Binary           | Point-Biserial Correlation       |
| NLI Evidence vs Truthfulness     | Binary vs Ordinal              | Rank Biserial Correlation        |
| NLI Evidence vs Hallucination    | Binary vs Binary               | Phi Coefficient (Cramér's V)     |

---

### **Python Code Implementation**
#### 1. Preprocess Data
```python
# Convert ordinal truthfulness to numerical (3=most true, 1=least true)
truthfulness_map = {
    'Fully True': 3,
    'Partially True': 2,
    'Not True': 1
}
df['truthfulness_encoded'] = df['truthfulness'].map(truthfulness_map)

# Convert binary variables to 0/1
df['hallucination_encoded'] = df['hallucination'].map({'Yes': 1, 'No': 0})
df['NLI_evidence_encoded'] = df['NLI_evidence'].astype(int)  # True=1, False=0
```

#### 2. Compute Pairwise Correlations
```python
from scipy.stats import spearmanr, pointbiserialr, rank_biserialr, chi2_contingency

# ------------------------------------------
# 1. Entailment Score vs Truthfulness (Ordinal)
# ------------------------------------------
corr_spearman, p_spearman = spearmanr(
    df['entailment_score'],
    df['truthfulness_encoded'],
    nan_policy='omit'
)
print(f"Spearman (Entailment vs Truthfulness): {corr_spearman:.3f}, p={p_spearman:.4f}")

# ------------------------------------------
# 2. Entailment Score vs Hallucination (Binary)
# ------------------------------------------
corr_pointbiserial, p_pointbiserial = pointbiserialr(
    df['hallucination_encoded'],
    df['entailment_score']
)
print(f"Point-Biserial (Entailment vs Hallucination): {corr_pointbiserial:.3f}, p={p_pointbiserial:.4f}")

# ------------------------------------------
# 3. NLI Evidence vs Truthfulness (Binary vs Ordinal)
# ------------------------------------------
corr_rankbiserial = rank_biserialr(
    x=df['truthfulness_encoded'],
    y=df['NLI_evidence_encoded']
)
print(f"Rank-Biserial (NLI Evidence vs Truthfulness): {corr_rankbiserial:.3f}")

# ------------------------------------------
# 4. NLI Evidence vs Hallucination (Binary vs Binary)
# ------------------------------------------
contingency_table = pd.crosstab(df['NLI_evidence'], df['hallucination'])
chi2, p_chi, _, _ = chi2_contingency(contingency_table)
n = contingency_table.sum().sum()
phi = np.sqrt(chi2 / n)  # Cramér's V for 2x2 tables = Phi coefficient
print(f"Phi Coefficient (NLI Evidence vs Hallucination): {phi:.3f}, p={p_chi:.4f}")
```

#### 3. Visualize Relationships
```python
# Plot 1: Entailment vs Truthfulness
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='truthfulness',
    y='entailment_score',
    data=df,
    order=['Fully True', 'Partially True', 'Not True']
)
plt.title("Entailment Scores by Truthfulness Category")

# Plot 2: Entailment vs Hallucination
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df,
    x='entailment_score',
    hue='hallucination',
    common_norm=False
)
plt.title("Entailment Score Distribution by Hallucination")

# Plot 3: NLI Evidence vs Hallucination
plt.figure(figsize=(6, 4))
sns.heatmap(
    contingency_table,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("NLI Evidence vs Hallucination")
plt.show()
```

---

### **Interpretation Guide**
1. **Entailment Score vs Truthfulness**:
   - **Spearman's Correlation**: 
     - A positive value (e.g., +0.5) indicates higher entailment scores correlate with more truthful summaries.
     - A significant p-value (p < 0.05) confirms the relationship is statistically meaningful.

2. **Entailment Score vs Hallucination**:
   - **Point-Biserial Correlation**:
     - A negative value (e.g., -0.3) suggests higher entailment scores correlate with fewer hallucinations.

3. **NLI Evidence vs Truthfulness**:
   - **Rank-Biserial Correlation**:
     - A positive value indicates `NLI_evidence=True` associates with higher truthfulness.

4. **NLI Evidence vs Hallucination**:
   - **Phi Coefficient**:
     - A value close to -1 implies `NLI_evidence=False` strongly associates with hallucinations.

---

### **Expected Outcome**
If your hypothesis holds:
- High `entailment_score` → "Fully True" summaries (positive Spearman correlation)
- Low `entailment_score` → "Not True" summaries (negative Point-Biserial with hallucinations)
- `NLI_evidence=True` → Fewer hallucinations (negative Phi coefficient)

Let me know if you need help interpreting your specific results!
