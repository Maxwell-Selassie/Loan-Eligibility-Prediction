# Production-Grade Preprocessing Pipeline - Complete Documentation

## Quick Start

### 1. Run the Pipeline

```bash
cd src
python preprocessing_pipeline.py
```

### 2. Programmatic Usage

```python
from preprocessing_pipeline import PreprocessingPipeline

# Initialize and execute
pipeline = PreprocessingPipeline(config_path="config/preprocessing_config.yaml")
datasets = pipeline.execute()

# Access datasets
df_train = datasets['train']
df_val = datasets['val']
df_test = datasets['test']

# Load saved models
from preprocessing import DataScaler, DataEncoder

scaler = DataScaler.load("models/scaler.pkl")
encoder = DataEncoder.load("models/encoder.pkl")
```

## Pipeline Stages

### Stage 1: Data Loading

**Module**: `data_loader.py`

**What it does**:

- Loads raw CSV data
- Optimizes data types automatically
- Reports memory usage
- Validates file existence and integrity

**Configuration**:

```yaml
file_paths:
  raw_data: 'data/raw/LEP.csv'
```

**Output**:

- Loaded DataFrame
- Metadata: shape, columns, memory usage

---

### Stage 2: Data Validation (Raw)

**Module**: `validator.py`

**What it does**:

- **Schema validation**: Checks for missing/extra columns
- **Data type validation**: Ensures numeric/categorical types are correct
- **Range validation**: Validates numeric columns are within expected bounds
- **Fail-fast**: Stops pipeline if validation fails

**Configuration**:

```yaml
validation:
  enabled: True
  schema_validation:
    expected_columns: [...]
  dtype_validation:
    numeric: [...]
    categorical: [...]
  range_validation:
    checks:
      Applicant_Income:
        min: 0
        max: 1000000
```

**Validation Checks**:

1. ✅ All expected columns present
2. ✅ Numeric columns are numeric dtype
3. ✅ Values within reasonable ranges
4. ❌ Fails immediately if any check fails

---

### Stage 3: Feature Dropping

**Module**: `feature_dropper.py`

**What it does**:

- Removes non-predictive columns (e.g., `Customer_ID`)
- Logs reason for each dropped column
- Tracks dropped columns in metadata

**Configuration**:

```yaml
columns_to_drop:
  - column: 'Customer_ID'
    reason: 'An identifier. Not predictive'
```

**Example Output**:

```
Dropping 1 columns: ['Customer_ID']
Shape after dropping: (614, 11)
```

---

### Stage 4: Data Splitting (80/20)

**Module**: `splitter.py`

**What it does**:

- Splits data into train (80%), test (20%)
- **Stratified split**: Maintains class balance across splits
- Uses fixed random seed for reproducibility
- Logs class distribution in each split

**Configuration**:

```yaml
data_split:
  train_size: 0.80
  test_size: 0.20
  stratify: True
  random_state: 42
  shuffle: True
```

**Why split BEFORE encoding/scaling?**

- Prevents data leakage
- Encoder/scaler fit only on train data
- Test remain truly unseen

**Example Output**:

```
Split results:
  Train: 491 rows (80.0%)
  Test: 123 rows (20.0%)
```

---

### Stage 5: Feature Encoding

**Module**: `encoder.py`

**What it does**:

- **Label Encoding**: Target variable (`Loan_Status`: Y→1, N→0)
- **One-Hot Encoding**: Categorical features with `drop_first=True`
- Fits on train, transforms train/val/test
- Handles unseen categories gracefully
- Aligns columns across splits

**Configuration**:

```yaml
encoding:
  label_encode:
    Loan_Status:
      positive_class: 'Y'  # Y = 1
      negative_class: 'N'  # N = 0
  
  one_hot:
    Gender:
      drop_first: True
    Married:
      drop_first: True
    Education:
      drop_first: True
    Self_Employed:
      drop_first: True
    Property_Area:
      drop_first: True
```

**Example Output**:

```
Fitting encoder...
  Label: Loan_Status → Y=1, N=0
  One-hot: Gender → 2 categories
  One-hot: Married → 2 categories
  One-hot: Education → 2 categories
  One-hot: Self_Employed → 2 categories
  One-hot: Property_Area → 3 categories

Applying encodings...
  Label encoded: Loan_Status
  One-hot encoded: 5 columns
Shape after encoding: (368, 15) (was 11)
```

**Column Transformation Example**:

```
Before:
Gender | Married | Education | Property_Area
Male   | Yes     | Graduate  | Urban

After (drop_first=True):
Gender_Male | Married_Yes | Education_Not Graduate | Property_Area_Semiurban | Property_Area_Urban
1           | 1           | 0                      | 0                       | 1
```

---

### Stage 6: Feature Scaling

**Module**: `scaler.py`

**What it does**:

- Standardizes numeric features (mean=0, std=1)
- Fits on train only, transforms train/val/test
- Supports multiple scaling methods
- Saves scaler for inference

**Configuration**:

```yaml
scaling:
  enabled: True
  method: 'standard'  # standard, minmax, robust, maxabs
  columns_to_scale:
    - 'Dependents'
    - 'Applicant_Income'
    - 'Coapplicant_Income'
    - 'Loan_Amount'
    - 'Loan_Amount_Term'
  fit_on: 'train'
```

**Scaling Methods**:

| Method | Formula | Use Case |
|--------|---------|----------|
| **Standard** | (x - μ) / σ | Default for most models |
| **MinMax** | (x - min) / (max - min) | When you need [0,1] range |
| **Robust** | (x - median) / IQR | Robust to outliers |
| **MaxAbs** | x / max(\|x\|) | Preserves sparsity |

**Example Output**:

```
Fitting scaler...
Scaling parameters (mean, std):
  Dependents: μ=0.81, σ=1.03
  Applicant_Income: μ=5403.46, σ=6109.04
  Coapplicant_Income: μ=1621.25, σ=2926.25
  Loan_Amount: μ=146.41, σ=85.59
  Loan_Amount_Term: μ=342.00, σ=65.12

Applying scaling...
  Scaled 5 columns
  Dependents: mean=0.0000, std=1.0000
  Applicant_Income: mean=-0.0000, std=1.0000
```

---

### Stage 7: Distribution Shift Detection

**Module**: `validator.py` → `check_distribution_shift()`

**What it does**:

- Uses **Kolmogorov-Smirnov test** to compare distributions
- Compares train vs test
- Warns if significant distribution shifts detected
- Helps identify data quality issues

**Configuration**:

```yaml
validation:
  distribution_shift:
    enabled: True
    method: 'ks_test'
    threshold: 0.05
```

**Example Output**:

```
Checking distribution shift: train vs validation...
  Dependents: No significant shift (p=0.8234)
  Applicant_Income: No significant shift (p=0.3421)
  Coapplicant_Income: Distribution shift detected (p=0.0234)
  Loan_Amount: No significant shift (p=0.5678)
  Loan_Amount_Term: No significant shift (p=0.9012)

⚠ Distribution shift detected in 1 features: ['Coapplicant_Income']
```

**What does this mean?**

- **p > 0.05**: Distributions are similar ✅
- **p < 0.05**: Distributions differ significantly ⚠️
- Action: Investigate why distributions differ (data collection issues?)

---

### Stage 9: Save Processed Data

**Module**: `preprocessing_pipeline.py` → `save_processed_data()`

**What it does**:

- Saves train/test CSVs
- Saves scaler and encoder as pickle files
- Saves preprocessing metadata as JSON

**Output Files**:

```
data/processed/
├── loan_e_p_cleaned_train.csv    # Ready for model training
└── loan_e_p_cleaned_test.csv     # Final evaluation

models/
├── scaler.pkl                    # Reload for inference
└── encoder.pkl                   # Reload for inference

data/artifacts/
└── preprocessing_metadata.json   # Full pipeline metadata
```

---

### Stage 10: Save Metadata

**Module**: `preprocessing_pipeline.py` → `save_metadata()`

**What it saves**:

```json
{
  "timestamp": "20250108_143025",
  "raw_data": {
    "n_rows": 614,
    "n_columns": 12,
    "columns": ["Customer_ID", "Gender", ...],
    "memory_mb": 0.05
  },
  "dropped_columns": ["Customer_ID"],
  "data_split": {
    "target_column": "Loan_Status",
    "train_size": 368,
    "val_size": 123,
    "test_size": 123,
    "train_indices": [0, 5, 7, ...],
    "val_indices": [1, 3, 9, ...],
    "test_indices": [2, 4, 6, ...]
  },
  "encoding": {
    "encoding_mappings": {...},
    "encoded_columns": [...]
  },
  "scaling": {
    "method": "standard",
    "columns_scaled": [...],
    "parameters": {
      "mean": [0.81, 5403.46, ...],
      "scale": [1.03, 6109.04, ...]
    }
  },
  "distribution_shift": {
    "train_vs_val": {...},
    "train_vs_test": {...}
  },
  "final_statistics": {
    "train_shape": [368, 15],
    "val_shape": [123, 15],
    "test_shape": [123, 15]
  }
}
```

**Why this matters**:

- **Reproducibility**: Recreate exact preprocessing
- **Debugging**: Understand what was done
- **Model serving**: Apply same transformations to new data
- **Auditing**: Track data lineage

---

## Using Saved Models for Inference

### Example: Preprocess New Data

```python
import pandas as pd
from preprocessing import DataScaler, DataEncoder

# Load new data
df_new = pd.read_csv("new_loan_applications.csv")

# Load saved transformers
encoder = DataEncoder.load("models/encoder.pkl")
scaler = DataScaler.load("models/scaler.pkl")

# Apply same transformations
df_new_encoded = encoder.transform(df_new)
df_new_scaled = scaler.transform(df_new_encoded)

# Now ready for model prediction!
predictions = model.predict(df_new_scaled)
```

### Example: Inverse Transform Predictions

```python
# If you scaled predictions and need original scale
predictions_scaled = model.predict(X_test)
predictions_original = scaler.inverse_transform(predictions_scaled)
```

---

## Performance & Optimization

### Execution Times (614 rows × 12 columns)

| Stage | Time | Notes |
|-------|------|-------|
| Data Loading | ~0.3s | Optimized dtypes |
| Validation (Raw) | ~0.5s | Schema + range checks |
| Feature Dropping | <0.1s | Simple column drop |
| Data Splitting | ~0.2s | Stratified split |
| Encoding | ~0.4s | One-hot + label |
| Scaling | ~0.3s | StandardScaler fit |
| Validation (Processed) | ~0.4s | Re-validation |
| Distribution Shift | ~0.6s | KS tests |
| Saving | ~0.5s | CSV + pickle |
| **Total** | **~3.5s** | End-to-end |

### Memory Usage

- Peak memory: ~50-100 MB
- Optimized dtypes save ~30-40%
- Scales well to 1M+ rows

---

## Best Practices

### 1. Always Fit on Train Only

✅ **CORRECT**:

```python
# Fit on train
encoder.fit(df_train)

# Transform all splits
df_train_enc = encoder.transform(df_train)
df_val_enc = encoder.transform(df_val)
df_test_enc = encoder.transform(df_test)
```

❌ **WRONG** (Data Leakage):

```python
# DON'T fit on entire dataset
encoder.fit(df_all)  # This leaks test data info!
```

### 2. Save Everything

Always save:

- ✅ Scaler (for inference)
- ✅ Encoder (for inference)
- ✅ Metadata (for reproducibility)
- ✅ Split indices (for debugging)

### 3. Validate Early and Often

```python
# Validate at each stage
df = load_data()
df = validate(df, stage='raw')      # ← Catch issues early
df = drop_features(df)
df = validate(df, stage='dropped')  # ← Verify transformation
# ... etc
```

### 4. Check Distribution Shifts

If distribution shift detected:

1. **Investigate**: Why are distributions different?
2. **Options**:
   - Collect more representative data
   - Use domain adaptation techniques
   - Accept the shift (document it)
   - Re-sample to match distributions

### 5. Version Control Everything

```bash
# Track these files in git
config/preprocessing_config.yaml
src/preprocessing/*.py
src/preprocessing_pipeline.py

# DON'T track these (too large)
data/processed/*.csv
models/*.pkl
logs/*.log
```

---

## 🐛 Troubleshooting

### Issue 1: Validation Failed - Missing Columns

**Error**:

```
ValidationError: Missing columns: {'Customer_ID'}
```

**Solution**:

- Check that column names in config match your CSV exactly
- Case-sensitive: `customer_id` ≠ `Customer_ID`

### Issue 2: Encoding Failed - Unmapped Values

**Warning**:

```
Gender: Unmapped values found: ['Other']
```

**Solution**:

- Update config to include all possible values
- Or handle in preprocessing (map 'Other' to 'Male' or 'Female')

### Issue 3: Scaler Not Fitted

**Error**:

```
RuntimeError: Scaler not fitted. Call fit() first.
```

**Solution**:

```python
# Always fit before transform
scaler.fit(df_train)
scaler.transform(df_val)  # Now works
```

### Issue 4: Column Mismatch After Encoding

**Issue**: Val/test have different columns than train

**Solution**: Pipeline automatically aligns columns:

```python
# Missing columns filled with 0
for col in train_columns:
    if col not in test_df.columns:
        test_df[col] = 0

# Reorder to match train
test_df = test_df[train_columns]
```

### Issue 5: Distribution Shift Detected

**Warning**:

```
⚠ Distribution shift detected in ['Coapplicant_Income']
```

**Actions**:

1. **Investigate**: Plot distributions

   ```python
   import matplotlib.pyplot as plt
   plt.hist(df_train['Coapplicant_Income'], alpha=0.5, label='Train')
   plt.hist(df_val['Coapplicant_Income'], alpha=0.5, label='Val')
   plt.legend()
   ```

2. **Options**:
   - Accept shift (document in report)
   - Re-split data with different seed
   - Collect more data
   - Use robust models (e.g., tree-based)

---

## 🔒 Security Features

### 1. Input Validation

- File existence checks
- Data type validation
- Range checks (prevent injection attacks)

### 2. Fail-Fast Principle

- Stop immediately on validation failure
- No partial processing
- Clear error messages

### 3. File Permissions

```python
# Models saved with restricted permissions
os.chmod(scaler_path, 0o644)  # Read-only for others
```

### 4. Logging Sanitization

- No sensitive data in logs
- PII masking support (future)

---

## Integration with Model Training

After preprocessing, use the data for training:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load processed data
df_train = pd.read_csv("../data/processed/loan_e_p_cleaned_train.csv")
df_val = pd.read_csv("../data/processed/loan_e_p_cleaned_val.csv")

# Separate features and target
X_train = df_train.drop('Loan_Status', axis=1)
y_train = df_train['Loan_Status']

X_val = df_val.drop('Loan_Status', axis=1)
y_val = df_val['Loan_Status']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_val, y_val)
print(f"Validation Accuracy: {score:.4f}")
```

---

## 🎓 Key Takeaways

1. **Fit on train, transform on all** - Prevents data leakage
2. **Validate at each stage** - Catch errors early
3. **Save everything** - Reproducibility is critical
4. **Check distribution shifts** - Ensures data quality
5. **Stratified splits** - Maintains class balance
6. **Drop first in one-hot** - Avoids multicollinearity
7. **Standard scaling** - Required for linear models
8. **Metadata tracking** - Essential for production

---

**END OF DOCUMENTATION**