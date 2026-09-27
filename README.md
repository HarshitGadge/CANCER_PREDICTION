# Breast Cancer Detection (Wisconsin Diagnostic Dataset, SVM)

Classifying breast tumours as malignant or benign from 30 cell-nucleus measurements taken from digitised biopsy images,
using a support vector machine (SVM).

## Results (held-out test set, 114 tumours: 48 malignant, 66 benign)

| Model | Accuracy | Malignant recall | Malignant precision |
|---|---|---|---|
| SVM (RBF kernel), min-max scaled | 97% | 94% | 100% |
| **SVM tuned with grid search** (C = 10, γ = 0.1) | **97%** | **96%** (46 of 48 caught) | 98% |

Recall on malignant tumours is the number that matters clinically: a missed cancer costs far more than a false alarm. The tuned model
misses 2 of the 48 malignant cases in the test set.

## Method

- Data: 569 tumours × 30 features (radius, texture, perimeter, area, smoothness and so on; mean, standard error and worst value of each).
  In the scikit-learn encoding used here, `target` 0 = malignant and 1 = benign.
- 80/20 train/test split. Features are min-max scaled using the **training** set's minimum and range.
- SVM with an RBF kernel, then `GridSearchCV` over C and γ with 5-fold cross-validation on the training set.

**Fix in this version.** The test set used to be scaled with its own minimum and range, so the test data influenced its own
preprocessing. It now uses the training-set values, as it would in production. After the fix, the tuned model's malignant recall stays
at 96% and its accuracy at 97%.

## Files

```
svm_classifier.ipynb     scaling, SVM, grid search, evaluation (executed)
breast_cancer_eda.ipynb  exploratory plots of the features
cancer_data              dataset as CSV (scikit-learn's load_breast_cancer)
data/wdbc_kaggle.csv     original Kaggle version of the dataset (diagnosis M/B)
cancerprediction.pkl     pickled model from the first version
```

## Run it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute svm_classifier.ipynb
```

Data source: Wolberg, Street & Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*, UCI Machine Learning Repository.
Tools: pandas, scikit-learn, matplotlib/seaborn.
