# Draft: Introduction
## TFG — Explainable Machine Learning for Postoperative Complication Prediction

---

## STRUCTURE OVERVIEW

| Block | Theme | ~Words |
|---|---|---|
| 1 | The surgical care challenge — setting the scene | 150 |
| 2 | Epidemiological burden of postoperative complications | 150 |
| 3 | Current clinical risk scores and their limits | 150 |
| 4 | ML in healthcare — promise and early evidence | 150 |
| 5 | The black-box barrier to clinical adoption | 150 |
| 6 | Class imbalance: a hidden threat to model reliability | 120 |
| 7 | Explainable AI (XAI) as the bridge | 150 |
| 8 | This project — contribution and scope | 150 |

---

## FULL DRAFT

Surgery is one of the most consequential interactions between a patient and the healthcare
system. Each year, more than 300 million surgical procedures are performed worldwide [CITE],
and while the majority result in successful outcomes, a clinically significant proportion leads to
serious postoperative complications. Among the most critical are surgical site and systemic
infectious complications, unexpected admission to the Intensive Care Unit (ICU), and 30-day
mortality — events that not only represent a profound impact on patient quality of life but also
impose a substantial burden on health systems in terms of resource consumption, prolonged
hospital stays, and preventable deaths.

Despite decades of advances in anaesthesia, surgical technique, and perioperative care, these
adverse events remain a persistent challenge. Studies estimate that postoperative complications
affect between 7% and 15% of surgical patients globally, and that a relevant fraction of
30-day deaths following non-cardiac surgery are potentially preventable [CITE]. Across diverse
healthcare systems, surgical populations span a wide spectrum of comorbidity profiles and
procedural risk levels, making uniform risk assessment inherently difficult. Identifying those
patients most likely to experience a serious adverse outcome before surgery is therefore a
central objective of modern perioperative medicine.

Clinicians currently rely on established risk stratification tools such as the American Society
of Anesthesiologists Physical Status Classification (ASA-PS) and the Charlson Comorbidity
Index (CCI) to guide preoperative assessment. Although these scores provide a valuable
structured framework for risk communication, they were developed in an era of limited data
availability and are inherently constrained by their simplicity: they aggregate complex clinical
profiles into a single ordinal value, discarding most of the granular information available in
a patient's electronic health record (EHR). As a result, they often lack the sensitivity required
to capture subtle but decisive risk factors, particularly for low-prevalence outcomes such as
postoperative infection or unexpected ICU admission, where the minority of at-risk patients
are precisely those most difficult to identify with coarse scoring systems.

The emergence of Machine Learning (ML) as an analytical paradigm has opened a new
frontier in clinical risk prediction. Unlike traditional scoring systems, ML algorithms are
capable of learning non-linear relationships across high-dimensional feature spaces, combining
sociodemographic variables, comorbidity profiles, surgical characteristics, pharmacological
history, and preoperative vital signs into a unified predictive framework. Numerous studies have
demonstrated the superior discriminative performance of ML models — particularly ensemble
methods such as Random Forests and gradient-boosted trees — compared to conventional
logistic regression or rule-based scores in the prediction of surgical outcomes [CITE]. These
results suggest that the information already recorded in hospital systems contains largely
untapped predictive signal that classical tools fail to exploit.

However, the translation of ML models into clinical practice has been markedly slower than
their technical performance would suggest. A central reason is interpretability — or rather, the
lack thereof. Tree ensembles, gradient-boosted models, and especially deep neural networks
operate as "black boxes": they produce a prediction without offering any transparent account of
how that prediction was reached. For a clinician deciding whether to escalate monitoring,
postpone a procedure, or initiate a preventive intervention, a model output of "high risk" is
of limited utility if it cannot be supported by a reasoning chain that maps to clinical intuition.
Regulatory frameworks, most notably the European AI Act and the medical device regulations
of the EU, increasingly require that AI systems used in high-stakes domains provide
meaningful explanations alongside their outputs [CITE]. This black-box barrier is not merely
a technical inconvenience — it represents a fundamental obstacle to the responsible adoption
of ML in perioperative medicine.

A further challenge that is frequently underestimated in the ML literature on clinical outcomes
is class imbalance. Postoperative complications, by their nature, are rare events: 30-day
mortality may occur in fewer than 2% of surgical episodes, unexpected ICU admissions in
perhaps 3–5%, and infectious complications in approximately 3% of cases. When the positive
class comprises such a small fraction of the training data, most standard ML algorithms will
implicitly optimise for accuracy on the majority class, effectively learning to predict "no
complication" for every patient. This results in models with high overall accuracy but
catastrophically low sensitivity — precisely the metric that matters most in a clinical screening
context. Managing this severe imbalance through principled oversampling strategies, such as
the Synthetic Minority Over-sampling Technique (SMOTE) and its variants, is therefore not
a methodological afterthought but a precondition for building clinically useful models.

The field of Explainable Artificial Intelligence (XAI) has developed a suite of techniques
designed to bridge the gap between predictive performance and clinical transparency. Among
these, SHapley Additive exPlanations (SHAP) — rooted in cooperative game theory — has
emerged as a leading framework for providing consistent, locally faithful, and globally
coherent feature attributions for any ML model [CITE Lundberg & Lee 2017]. SHAP values
quantify the contribution of each feature to an individual prediction relative to a baseline,
allowing a clinician to understand not only that a patient is flagged as high-risk, but which
specific factors — a prolonged surgery duration, elevated heart rate, the presence of
immunosuppressive comorbidities — are driving that assessment. This level of explanatory
granularity is arguably the minimum requirement for a ML tool to be genuinely actionable in
a preoperative consultation.

This project addresses all of the above challenges in an integrated, end-to-end pipeline applied
to a large-scale real-world EHR dataset provided by Osakidetza, the Basque Health Service. The
source database encompasses \textbf{103,179 surgical episodes} corresponding to more than
\textbf{90,825 unique patients} across a broad spectrum of surgical procedures. Three prediction
targets are studied in parallel: 30-day postoperative mortality, unexpected ICU admission, and
postoperative infectious complications. For 30-day mortality, the full episode-level cohort is
retained (103,179 records; 1,665 events; 1.61\% prevalence). For unexpected ICU admission, the
analysis is restricted to the \textbf{36,338 episodes} with a confirmed outcome label, since
approximately 65\% of records in the source database contain no ICU admission entry; as the
absence of a label cannot be unambiguously interpreted as a negative event, those records are
excluded to avoid systematic label noise (27,125 unique patients; 318 events; 0.88\% prevalence).
For postoperative infectious complications, clinical inclusion criteria define a study cohort of
\textbf{13,662 patients} (413 events; 3.02\% prevalence) corresponding to the surgical population
of interest, enriched with perioperative vital signs and procedure-level features. Together, these
three tasks span a broad spectrum of comorbidity profiles and outcome prevalences, enabling a
systematic comparative analysis of model behaviour across outcomes with different clinical
mechanisms and risk factor profiles. For each target, a feature engineering pipeline constructs a
rich multidimensional feature matrix from raw hospital records, multiple class-balancing strategies
are evaluated and compared, and a battery of ML algorithms is trained, cross-validated, and
optimised. The best-performing model for each outcome is then subjected to SHAP-based
interpretability analysis, producing both global feature importance rankings and individual
prediction explanations suitable for clinical review.

The contributions of this work are therefore threefold: (i) a rigorous methodological framework
for predicting rare postoperative outcomes from real-world EHR data, with explicit handling of
class imbalance; (ii) a comparative evaluation of ML algorithms and oversampling strategies
across three clinically distinct prediction tasks; and (iii) an explainability layer that translates
model outputs into clinically meaningful feature attributions, directly addressing the
interpretability gap that has historically limited the adoption of ML in perioperative risk
assessment. By combining predictive accuracy with transparency, this project contributes to
the broader agenda of human-centred artificial intelligence in healthcare — one where data-
driven tools support, rather than replace, clinical judgment.

---

## REFERENCES TO CITE (placeholders — fill with real citations)

- [CITE global surgical volume] → Weiser et al., Lancet 2015 or WHO Global Surgery 2030
- [CITE postoperative complication rates] → Healey et al. / Pearse et al. EuroSCORE / POISE
- [CITE ML vs classical scores] → Bihorac et al. (MySurgeryRisk) / Bellomo et al.
- [CITE EU AI Act / medical device regulation] → Regulation (EU) 2024/1689
- [CITE SHAP] → Lundberg & Lee, NeurIPS 2017 — "A Unified Approach to Interpreting Model Predictions"
- [CITE SMOTE] → Chawla et al., JAIR 2002

---

## STYLE NOTES

- Paragraph length: 100–180 words each (similar to example provided)
- Avoid bullet points in the body — all flowing prose
- Each paragraph should end with a sentence that "calls forward" the next block
- Target total length: ~1,100–1,300 words for the introduction body
- Do NOT start with "In recent years..." or "With the advent of..." (overused openers)
- The current draft opens with a concrete global surgical statistic — strong academic opener
