# Draft: Section 2 — Definitions, Technical Terms and Abbreviations
## TFG — Explainable Machine Learning for Postoperative Complication Prediction

---

## 2. Definitions, Technical Terms and Abbreviations

This section presents the key definitions, technical terms, acronyms and abbreviations used
throughout the document to facilitate a clearer understanding of the project's content.

---

## 2.1. Key Definitions

- **Postoperative complication**: Any adverse clinical event occurring within a defined time
  window following a surgical procedure, including infectious complications, unexpected
  organ failure, or death. In this project, the observation window is set to 30 days
  post-surgery.

- **Perioperative period**: The time span encompassing the preoperative, intraoperative, and
  immediate postoperative phases of care, typically extending from hospital admission for
  surgery to discharge or death.

- **Surgical site infection (SSI)**: An infection occurring at the incision site or in the
  organ or body cavity manipulated during surgery, classified according to depth and
  anatomical location. Broader infectious complications in this project also include
  systemic infections (sepsis), urinary tract infections, catheter-related infections,
  and respiratory infections.

- **Unexpected ICU admission**: Transfer of a patient to the Intensive Care Unit following
  surgery when such admission was not planned preoperatively, reflecting an unanticipated
  deterioration in clinical status.

- **30-day mortality**: Death occurring within 30 days of the surgical procedure, regardless
  of the place of death, used as a standard endpoint in perioperative outcome research.

- **Comorbidity**: A pre-existing medical condition that coexists with the primary diagnosis
  or surgical indication. Comorbidities are systematically captured in this project through
  the Charlson Comorbidity Index and additional binary clinical flags.

- **Class imbalance**: A condition in a supervised learning dataset where one class (the
  minority, or positive class) is substantially underrepresented relative to the other.
  In this project, positive outcomes (complications) account for approximately 1–5% of
  cases, creating severe imbalance that must be explicitly managed.

- **Preoperative risk assessment**: The clinical process of evaluating a patient's likelihood
  of adverse surgical outcomes prior to the intervention, typically informing decisions
  about anaesthetic choice, monitoring intensity, and surgical strategy.

- **Electronic Health Record (EHR)**: A longitudinal, digitally stored record of a patient's
  health information, including diagnoses, medications, laboratory results, vital signs,
  and clinical encounters, maintained by a healthcare provider.

- **Explainability**: The property of a model or system that allows its internal logic and
  outputs to be understood and communicated in human-interpretable terms. Distinct from
  *interpretability*, which refers to the degree to which a model's mechanics are
  transparent by design (e.g., logistic regression), explainability may also be applied
  post-hoc to complex "black-box" models.

- **Feature engineering**: The process of transforming raw data into structured numeric
  variables (features) suitable for input to machine learning models, including aggregation,
  encoding, and derivation of new variables from existing ones.

---

## 2.2. Technical and Methodological Terms

- **Supervised learning**: A machine learning paradigm in which a model is trained on
  labelled examples — pairs of input features and known output labels — to learn a
  mapping that generalises to unseen data.

- **Binary classification**: A supervised learning task where the model predicts one of two
  possible outcomes, such as complication vs. no complication.

- **Logistic Regression (LR)**: A linear classification model that estimates the probability
  of a binary outcome using a logistic sigmoid function applied to a weighted linear
  combination of input features.

- **Random Forest (RF)**: An ensemble learning method that builds multiple decision trees
  on bootstrapped subsets of the training data and aggregates their predictions, reducing
  variance and improving generalisation.

- **Gradient Boosting / XGBoost**: An ensemble technique in which decision trees are built
  sequentially, each correcting the residual errors of the previous one. XGBoost is a
  highly optimised, regularised implementation of gradient boosting widely used in
  structured data settings.

- **Extra Trees (Extremely Randomised Trees)**: A bagging-based ensemble method similar
  to Random Forest, but with an additional layer of randomness: split thresholds are
  chosen randomly rather than optimised, further reducing variance at the cost of a
  small increase in bias.

- **Gaussian Naïve Bayes (GaussianNB)**: A probabilistic classifier based on Bayes'
  theorem that assumes conditional independence between features and models each
  feature's likelihood as a Gaussian distribution. Extremely fast to train and serves
  as a lightweight probabilistic baseline.

- **LightGBM (LGBM)**: A gradient boosting framework developed by Microsoft that uses
  histogram-based split finding and leaf-wise tree growth, making it significantly
  faster and more memory-efficient than standard implementations. Used in this project
  as one of the seven candidate classifiers.

- **SMOTE (Synthetic Minority Over-sampling Technique)**: An oversampling technique that
  generates synthetic minority-class examples by interpolating between existing minority
  samples in feature space, rather than simply duplicating them.

- **SMOTE variants**: Extensions of the base SMOTE algorithm that modify the sampling
  strategy. Variants evaluated in this project include:
  - *Borderline-SMOTE*: focuses synthetic generation on minority samples near the
    decision boundary, which are the hardest to classify correctly.
  - *SVM-SMOTE*: uses a Support Vector Machine to identify the borderline region and
    generates synthetic samples along the support vectors.
  - *ADASYN (Adaptive Synthetic Sampling)*: generates more synthetic samples in regions
    of the feature space where the local density of minority examples is lower,
    adaptively concentrating effort on the hardest areas.
  - *SMOTETomek*: combines SMOTE oversampling with Tomek Links cleaning, removing
    majority-class samples that are near minority samples to create a cleaner boundary.
  - *SMOTEENN*: combines SMOTE with Edited Nearest Neighbours (ENN), applying a
    cleaning step that removes samples misclassified by their k-nearest neighbours.

- **Random Over-Sampler**: A simple oversampling strategy that duplicates randomly
  selected minority-class examples with replacement until the desired class ratio is
  reached. Serves as a baseline oversampling method for comparison.

- **Random Under-Sampler (RUS)**: An undersampling strategy that randomly discards
  majority-class examples until the desired class ratio is reached. Unlike oversampling,
  it reduces the total dataset size, which can accelerate training but may lose
  informative majority-class patterns.

- **Pipeline**: A sequential chain of data processing and modelling steps (e.g.,
  oversampling → scaling → classification) implemented as a single object to prevent
  data leakage and simplify cross-validation.

- **Data leakage**: The inadvertent inclusion of information in the training set that would
  not be available at prediction time, leading to overly optimistic evaluation results.
  In this project, leakage is prevented by (a) performing all preprocessing steps inside
  the cross-validation loop and (b) grouping folds by patient so no patient appears in
  both train and validation partitions.

- **Cross-validation (K-Fold CV)**: A resampling procedure that partitions the dataset into
  K equal folds; the model is trained on K-1 folds and evaluated on the remaining fold,
  rotating K times. Results are averaged to obtain a more reliable performance estimate.

- **StratifiedGroupKFold**: A cross-validation variant that simultaneously enforces class
  stratification (preserving the minority-class ratio in each fold) and group integrity
  (ensuring all records from the same patient are assigned to the same fold). Essential
  in this project because many patients have multiple surgical episodes, and naive
  stratification would leak patient-level information across train and validation folds.

- **Stratified split**: A train/test or cross-validation split that preserves the class
  distribution of the original dataset in each partition, critical when dealing with
  imbalanced classes.

- **Hyperparameter tuning / GridSearchCV**: The process of searching over a predefined grid
  of model configuration parameters (e.g., tree depth, learning rate, number of estimators)
  to identify the combination that maximises a validation metric.

- **Precision**: The proportion of positive predictions that are truly positive:
  TP / (TP + FP). A high precision indicates few false positives.

- **Recall (Sensitivity)**: The proportion of actual positive cases correctly identified by
  the model: TP / (TP + FN). A high recall indicates few false negatives — the
  clinically critical metric in complication screening.

- **F1-score**: The harmonic mean of precision and recall, providing a single balanced
  performance measure particularly useful under class imbalance.

- **Confusion matrix**: A tabular summary of a classifier's predictions broken into four
  counts — True Positives (TP), False Positives (FP), True Negatives (TN), and False
  Negatives (FN) — from which all classification metrics can be derived.

- **Decision threshold (classification threshold)**: The probability cut-off above which a
  model's output is mapped to the positive class. The default is 0.5, but it can be
  adjusted to trade precision for recall depending on clinical priorities (e.g., lowering
  the threshold increases sensitivity at the cost of more false alarms).

- **True Positive Rate (TPR / Sensitivity / Recall)**: TP / (TP + FN); proportion of actual
  positives correctly identified. Equivalent to Recall.

- **False Positive Rate (FPR)**: FP / (FP + TN); proportion of actual negatives incorrectly
  classified as positive. Used as the x-axis of the ROC curve.

- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve; measures the
  model's ability to discriminate between positive and negative cases across all
  classification thresholds.

- **PR-AUC (Average Precision)**: Area Under the Precision-Recall Curve; preferred over
  AUC-ROC when the positive class is rare, as it is more sensitive to model performance
  on minority-class examples.

- **SHAP (SHapley Additive exPlanations)**: A game-theoretic framework for explaining
  individual model predictions by computing each feature's marginal contribution,
  averaged across all possible feature orderings. Produces both local (per-patient) and
  global (population-level) feature importance attributions.

- **TreeExplainer**: A SHAP algorithm optimised for tree-based models (Random Forest,
  XGBoost, LightGBM) that computes exact SHAP values in polynomial time by exploiting
  the tree structure.

- **KernelExplainer**: A model-agnostic SHAP algorithm applicable to any classifier,
  including Logistic Regression and GaussianNB. It approximates SHAP values by fitting
  a weighted linear model around each prediction, at a higher computational cost than
  TreeExplainer.

- **Imputation**: The process of replacing missing values in a dataset with estimated
  substitutes, such as the median (used in this project) or mean of the observed values
  in the same feature.

- **Overfitting**: A condition in which a model learns the training data too precisely,
  capturing noise rather than generalisable patterns, resulting in degraded performance
  on unseen data.

- **Learning curve**: A diagnostic plot showing training and validation performance as a
  function of training set size. Converging curves suggest low bias; a large gap between
  them indicates overfitting (high variance).

- **Bootstrap confidence intervals**: A resampling technique that estimates the uncertainty
  of a metric (e.g., AUC-ROC, PR-AUC) by repeatedly resampling the test set with
  replacement (typically n = 1,000 iterations) and computing the 2.5th–97.5th percentile
  of the resulting distribution as a 95% confidence interval.

- **Model calibration**: The degree to which a model's predicted probabilities match the
  true empirical frequencies of the positive class. A well-calibrated model outputs
  probabilities that are directly interpretable as clinical risk estimates.

- **Calibration curve (reliability diagram)**: A graphical assessment of calibration that
  plots mean predicted probability against observed event rate across equally spaced
  probability bins; a perfectly calibrated model follows the diagonal.

- **Brier Score**: The mean squared error between predicted probabilities and true binary
  labels; ranges from 0 (perfect) to 1 (worst). Values below 0.10 are considered
  excellent and below 0.25 acceptable.

- **Isotonic Regression (post-hoc calibration)**: A non-parametric monotone fitting
  method applied after training to recalibrate raw model outputs. Used in this project
  to correct the tendency of some models (especially GaussianNB) to over- or
  under-estimate event probabilities.

- **Friedman test**: A non-parametric statistical test equivalent to a repeated-measures
  ANOVA that evaluates whether multiple algorithms differ significantly in performance
  across the same cross-validation folds. Used in this project to compare the eight
  balancing techniques.

- **Wilcoxon signed-rank test**: A non-parametric pairwise post-hoc test applied after a
  significant Friedman result to determine which pairs of techniques differ significantly.
  Each technique's fold-level scores are treated as matched pairs.

- **Bonferroni correction**: A multiple-testing correction that divides the significance
  level α by the number of pairwise comparisons performed, reducing the probability of
  false positives when testing many simultaneous hypotheses.

---

## 2.3. Acronyms and Abbreviations

- **ADASYN**: Adaptive Synthetic Sampling
- **AI**: Artificial Intelligence
- **ANOVA**: Analysis of Variance
- **AP**: Average Precision (equivalent to PR-AUC)
- **ASA-PS**: American Society of Anesthesiologists Physical Status Classification
- **AUC**: Area Under the Curve
- **CCI**: Charlson Comorbidity Index
- **CI**: Confidence Interval
- **CIE-10 / ICD-10**: International Classification of Diseases, 10th Revision
  *(Clasificación Internacional de Enfermedades)*
- **CKPT**: Checkpoint
- **CV**: Cross-Validation
- **EDA**: Exploratory Data Analysis
- **EHR**: Electronic Health Record
- **ENN**: Edited Nearest Neighbours
- **ExtraTrees**: Extremely Randomised Trees
- **FN**: False Negative
- **FP**: False Positive
- **FPR**: False Positive Rate
- **GaussianNB**: Gaussian Naïve Bayes
- **GBM / GradientBoosting**: Gradient Boosting Machine
- **ICU**: Intensive Care Unit
- **LGBM / LightGBM**: Light Gradient Boosting Machine
- **LR**: Logistic Regression
- **ML**: Machine Learning
- **PR-AUC**: Precision-Recall Area Under the Curve
- **RF**: Random Forest
- **ROC**: Receiver Operating Characteristic
- **ROS**: Random Over-Sampler
- **RUS**: Random Under-Sampler
- **SHAP**: SHapley Additive exPlanations
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **SMOTEENN**: SMOTE + Edited Nearest Neighbours
- **SMOTETomek**: SMOTE + Tomek Links
- **SSI**: Surgical Site Infection
- **SVM**: Support Vector Machine
- **SVM-SMOTE**: Support Vector Machine SMOTE
- **TFG**: *Trabajo de Fin de Grado* (Final Degree Project)
- **TN**: True Negative
- **TP**: True Positive
- **TPR**: True Positive Rate (Sensitivity / Recall)
- **WHO**: World Health Organization
- **XAI**: Explainable Artificial Intelligence
- **XGB / XGBoost**: eXtreme Gradient Boosting
