# RECIPES.md


The goal is simple:

> Given a problem fingerprint, what usually works, which models should be tried first, and what validation design is non-negotiable?

This document is the practical recipe layer: **problem signals → methods → models → validation → failure modes**.

# Recipe library

## Recipe 1 — High-cardinality categoricals

**Problem signals**
- many IDs
- many rare categories
- repeated entities
- leakage risk
- sparse one-hot explosion

**Typical competitions**
- fraud
- credit risk
- recommender logs
- ad click prediction
- transaction risk

**Strong baseline**
- CatBoost

**Winning upgrades**
- frequency encoding
- count encoding
- out-of-fold target encoding
- category interaction statistics
- CatBoost + LightGBM blend
- entity embeddings as extra ensemble member

**Validation strategy**
- GroupKFold if entities repeat
- otherwise strict out-of-fold encoding with normal CV

**Common failure modes**
- target leakage from full-data target encoding
- rare-category overfit
- category mappings changing between train and test

**Compute budget**
- low to medium

**Interpretability**
- medium

**Best models**
- CatBoost
- LightGBM / XGBoost + leak-safe encoding
- embeddings only as secondary option

**Confidence**
- High

---

## Recipe 2 — Skewed numerical features / heavy-tailed targets

**Problem signals**
- prices
- counts
- exposures
- long-tail regression targets
- large outliers dominating variance

**Typical competitions**
- price prediction
- demand prediction
- insurance severity
- sales value estimation

**Strong baseline**
- LightGBM on `log1p(target)` when valid

**Winning upgrades**
- log transforms on selected features
- winsorization / clipping
- robust scaling for linear / NN variants
- residual modeling
- blending raw-target and transformed-target models

**Validation strategy**
- time-aware split if temporal
- standard CV otherwise

**Common failure modes**
- bad inverse transform
- clipping away useful extremes
- treating negative values incorrectly in transforms

**Compute budget**
- low

**Interpretability**
- medium to high

**Best models**
- LightGBM
- XGBoost
- CatBoost
- NN usually as ensemble-only candidate

**Confidence**
- High

---

## Recipe 3 — Grouped entities / repeated units

**Problem signals**
- same customer, patient, machine, store, or device appears multiple times
- row independence assumption is broken
- aggregates over entity history are likely useful

**Typical competitions**
- recommender systems
- fraud
- medical imaging
- user behavior modeling
- industrial quality data with repeated lots or tools

**Strong baseline**
- simplest solid model + GroupKFold

**Winning upgrades**
- group aggregates
- target history features
- per-entity normalization
- entity-history recency features
- group calibration

**Validation strategy**
- GroupKFold
- StratifiedGroupKFold if label balance matters
- time-group hybrid split if grouped and temporal

**Common failure modes**
- same entity split across train and validation
- future entity history leakage
- unrealistically optimistic CV

**Compute budget**
- low

**Interpretability**
- high

**Best models**
- split choice first
- model choice second

**Confidence**
- Very high

---

## Recipe 4 — Time series with seasonality / known frequency

**Problem signals**
- repeated weekly, monthly, quarterly, or hourly patterns
- panel forecasting
- promotions / holidays / calendar structure
- multiple horizons

**Typical competitions**
- retail forecasting
- traffic
- energy
- store sales
- demand planning

**Strong baseline**
- lags + rolling stats + calendar features + LightGBM

**Winning upgrades**
- Fourier terms
- holiday indicators
- hierarchical aggregates
- per-entity lag stacks
- known-future covariates
- direct multi-horizon modeling
- TFT when setup is rich enough

**Validation strategy**
- rolling split
- walk-forward split
- blocked forward CV only

**Common failure modes**
- future leakage in rolling windows
- centered windows
- test-time covariates unavailable in practice
- random CV on temporal data

**Compute budget**
- low to medium for GBDT
- high for TFT / sequence models

**Interpretability**
- high for GBDT
- medium for deep models

**Best models**
- LightGBM / XGBoost first
- TFT second
- sequence models only when scale and structure justify them

**Confidence**
- High

---

## Recipe 5 — Sparse text classification / regression

**Problem signals**
- text is the main source of signal
- structured features are weak or limited
- labels depend on semantics or style
- duplicated passages may exist

**Typical competitions**
- toxicity
- sentiment
- essay scoring
- readability
- topic classification

**Strong baseline**
- TF-IDF + logistic regression / linear SVM

**Winning upgrades**
- BERT-family fine-tuning
- multi-seed transformer ensembles
- domain-specific pretraining if allowed
- metadata fusion
- pseudolabeling in some settings

**Validation strategy**
- stratified CV
- grouped CV if source overlap or duplicates matter

**Common failure modes**
- over-cleaning text
- duplicate leakage
- tokenization mismatch
- using leaderboard movement instead of honest validation

**Compute budget**
- medium to high

**Interpretability**
- low to medium

**Best models**
- TF-IDF + linear first
- BERT-family next

**Confidence**
- High

---

## Recipe 6 — Severe class imbalance

**Problem signals**
- tiny positive class
- asymmetric business cost
- ranking or recall matters more than global accuracy
- minority class may be noisy

**Typical competitions**
- fraud
- medical detection
- anomaly-like classification
- rare defect detection

**Strong baseline**
- weighted LightGBM / XGBoost

**Winning upgrades**
- focal loss for suitable deep setups
- hard negative mining
- threshold tuning
- prevalence-aware calibration
- ranking formulation where appropriate
- negative downsampling for training only

**Validation strategy**
- preserve realistic prevalence
- keep grouping and time structure intact
- evaluate at meaningful operating points

**Common failure modes**
- using accuracy
- tuning threshold on leaked folds
- training distribution not matching inference prior

**Compute budget**
- medium

**Interpretability**
- medium

**Best models**
- weighted GBDT for tabular
- focal-loss deep models in vision / detection settings

**Confidence**
- High

---

## Recipe 7 — Ranking / recommender systems

**Problem signals**
- MAP@K / Recall@K / NDCG
- implicit feedback
- session or user-item history
- retrieval bottleneck dominates final score

**Typical competitions**
- e-commerce recommendations
- basket prediction
- article recommendation
- session ranking

**Strong baseline**
- popularity + recency + co-occurrence candidate generation

**Winning upgrades**
- co-visitation retrieval
- multiple candidate generators
- LightGBM ranker
- graph recommenders
- sequence-based rerankers
- blending retrieval channels

**Validation strategy**
- user-aware and time-aware split
- session-aware split where applicable

**Common failure modes**
- poor candidate generation caps the whole system
- future interactions leaking into history
- evaluation mismatch between offline candidates and full ranking metric

**Compute budget**
- medium to high

**Interpretability**
- low to medium

**Best models**
- retrieval + LightGBM ranker
- graph recommender as extension
- transformer or sequence reranker when scale supports it

**Confidence**
- High

---

## Recipe 8 — Multimodal fusion

**Problem signals**
- text + image
- text + tabular
- OCR + metadata
- multiple weak signals that improve when fused

**Typical competitions**
- product matching
- multimodal search
- listing quality
- duplicate detection
- catalog enrichment

**Strong baseline**
- strongest single-modality model

**Winning upgrades**
- late fusion
- stacked logits
- similarity models
- graph clustering
- metadata fusion
- threshold optimization

**Validation strategy**
- group duplicates together
- prevent same item family across train and validation

**Common failure modes**
- duplicate leakage
- modality imbalance
- OCR brittleness
- overfitting fusion weights

**Compute budget**
- high

**Interpretability**
- low

**Best models**
- modality-specific encoders + late fusion

**Confidence**
- Medium to high

---

## Recipe 9 — Small tabular dataset

**Problem signals**
- few rows
- many features
- unstable fold metrics
- public leaderboard shifts dramatically

**Typical competitions**
- small scientific datasets
- biomedical tabular tasks
- compact risk models

**Strong baseline**
- regularized linear model and shallow GBDT

**Winning upgrades**
- repeated CV
- careful feature selection
- null importance
- model simplicity
- seed ensembling only if stable

**Validation strategy**
- repeated CV
- group or time-awareness if needed
- uncertainty tracking across folds

**Common failure modes**
- overfitting hyperparameters
- interpreting leaderboard noise as progress
- over-complex models on tiny data

**Compute budget**
- low

**Interpretability**
- high

**Best models**
- regularized linear
- shallow LightGBM / XGBoost
- CatBoost if categoricals dominate

**Confidence**
- High

---

## Recipe 10 — Computer vision transfer learning

**Problem signals**
- moderate labeled dataset
- image classification / segmentation / detection
- label noise or source bias possible
- pretrained backbone likely helpful

**Typical competitions**
- image classification
- medical imaging
- defect detection
- satellite imagery
- object detection

**Strong baseline**
- pretrained backbone + standard augmentations

**Winning upgrades**
- stronger augmentations
- TTA
- fold ensembling
- pseudo-labeling
- source-aware validation
- architecture diversity

**Validation strategy**
- grouped CV when patient, video, scene, or source repeats
- stratified folds for class balance

**Common failure modes**
- image leakage via near-duplicates
- mismatched augmentations
- public leaderboard overfit
- label-noise amplification

**Compute budget**
- medium to high

**Interpretability**
- low to medium

**Best models**
- pretrained CNN / ViT family
- UNet-style models for segmentation
- detection architectures for object tasks

**Confidence**
- High

---

## 8. Decision flow

Use this sequence on a new competition-style problem.

### Step 1 — Is the problem temporal?

If yes:
- use time-aware validation first
- start with lag features, rolling features, calendar features
- add Fourier terms if periodicity exists
- only add TFT / deep sequence models after the GBDT baseline is strong

### Step 2 — Do entities repeat?

If yes:
- switch to GroupKFold or StratifiedGroupKFold
- create entity history and aggregate features
- check future-history leakage before adding complex models

### Step 3 — Are there high-cardinality categoricals?

If yes:
- start with CatBoost
- compare against LightGBM / XGBoost with out-of-fold target or frequency encoding
- never encode with target information using the full dataset

### Step 4 — Is text the main signal?

If yes:
- begin with TF-IDF + linear model
- move to BERT-family fine-tuning
- guard against duplicate leakage

### Step 5 — Is the metric ranking-based?

If yes:
- build candidate generation first
- then rerank
- graph or sequential models become attractive only after the retrieval layer is strong

### Step 6 — Is it standard non-temporal tabular?

If yes:
- start with LightGBM / XGBoost / CatBoost
- add feature engineering and calibrated blends
- only test tabular deep learning after the tree baseline is competitive

### Step 7 — Is there severe imbalance?

If yes:
- use weighted objectives
- monitor PR-oriented or ranking-oriented metrics if appropriate
- tune thresholds separately from model training

---

## 9. Evidence-strength rubric

Use this when rating tactics.

### Dimensions

- **Frequency**: how often it appears in strong solutions
- **Strength**: how much it usually helps
- **Generality**: how broadly it transfers
- **Leakage risk**: how easy it is to misuse
- **Compute cost**
- **Interpretability**

### Example ratings

| Tactic | Frequency | Strength | Generality | Leakage Risk | Notes |
|---|---:|---:|---:|---:|---|
| Group-aware CV | 5 | 5 | 5 | 1 | Often more important than model choice |
| CatBoost for high-cardinality categoricals | 4 | 4 | 4 | 2 | Strong default for category-heavy tabular |
| BERT fine-tuning for NLP | 5 | 5 | 4 | 3 | Strong once validation is honest |
| Lag features + LightGBM for forecasting | 5 | 5 | 5 | 2 | Extremely hard to beat |
| TFT for rich forecasting | 3 | 4 | 3 | 2 | Useful but not default |
| Retrieval + reranker for recsys | 5 | 5 | 4 | 3 | Candidate generation is critical |
| Pseudo-labeling | 3 | 3 | 2 | 4 | Can help, but fragile |
| Public leaderboard probing tricks | 2 | 2 | 1 | 5 | Usually not reusable |

---

## 10. Compact template for future use

Use this block for each new problem.

**Problem type:**  
**Metric:**  
**Core signals:**  
**Leakage risks:**  
**Best baseline:**  
**Best model families:**  
**Winning upgrades:**  
**Validation strategy:**  
**Post-processing:**  
**Common failure modes:**  
**Evidence strength:**  
**Recommended starting stack:**  

---

## 11. Recommended starting stacks by family

### Tabular

**Start with**
- CatBoost
- LightGBM
- XGBoost

**Then add**
- leak-safe categorical encoding
- group aggregates
- calibration
- multi-seed / multi-model blending

### NLP

**Start with**
- TF-IDF + logistic regression / linear SVM

**Then add**
- BERT-family fine-tuning
- metadata fusion
- multi-seed ensemble

### Time series

**Start with**
- lag features
- rolling statistics
- calendar features
- Fourier terms if needed
- LightGBM / XGBoost

**Then add**
- hierarchical features
- TFT or sequence models if justified

### Computer vision

**Start with**
- pretrained backbone
- moderate augmentations
- honest grouped validation if necessary

**Then add**
- stronger augmentations
- TTA
- pseudo-labeling
- fold / architecture ensembling

### Recommender / ranking

**Start with**
- popularity / co-occurrence / recency retrieval
- candidate generation

**Then add**
- LightGBM ranker
- graph recommender
- sequence reranker
- metric-aware post-processing

### Multimodal

**Start with**
- strongest unimodal baseline

**Then add**
- late fusion
- similarity features
- stacked logits
- threshold or calibration tuning

---

## 12. General principles that transfer well

Across many competitions, these are repeatedly valuable:

- honest validation beats clever modeling on a broken split
- strong baselines win a shocking amount of the time
- feature engineering still matters, especially in tabular and forecasting
- ensembling is often the final 10–20% of leaderboard improvement
- metric alignment matters: optimize for the actual evaluation objective
- reproducibility should be first-class:
  - dataset version
  - split seed
  - recipe hash
  - model seed
  - feature block hashes
  - evaluator version
  - exact mission snapshot

---

## 13. What not to over-generalize

Do not blindly elevate these into universal recipes:

- competition-specific hacks that exploit quirks of the test set
- methods that depend on external data unavailable in real projects
- public leaderboard probing behavior
- highly compute-heavy ensembles that give tiny marginal gains
- tricks that only worked because of leakage or split artifacts

The playbook should preserve:
- what is robust
- what is transferable
- what is compute-efficient
- what is honest under proper validation

---

## 14. Recommended use in practice

When starting a new problem:

1. classify the problem using the signal taxonomy
2. pick the matching baseline recipe
3. enforce the right CV before feature engineering
4. test only a few model families that fit the signal shape
5. escalate to advanced tricks only after the baseline is trustworthy
6. record every change with reproducibility metadata
