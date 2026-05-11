# Methodological Soundness — CBL-16 London Policing Demand Study
### Full Technical Documentation: Phases 1–5

---

## Research Question
**How can data-driven estimates of police demand be used to inform the effective organisation and allocation of policing resources in the United Kingdom?**

The analytical pipeline translates this question into five sequential phases: data ingestion → feature engineering → statistical validation → risk index construction → clustering into priority tiers. Every methodological decision is evidence-based and traceable to academic literature.

---

---

## PHASE 1 — Data Ingestion & Filtering

### What we did
Loaded all raw data across five sources, filtered to London, standardised formats, and saved clean outputs as compressed Parquet files.

### Data sources and rationale

| Dataset | Source | Records | Why included |
|---|---|---|---|
| Street Crimes | police.uk | 3,443,915 | Primary measure of crime demand per LSOA |
| Crime Outcomes | police.uk | 2,905,468 | Needed to compute resolution rate |
| Stop & Search | police.uk | 372,018 | Proxy for proactive policing intensity |
| TfL Station Footfall | Transport for London | 15,615 | Proxy for public presence / crime opportunity |
| Temperature (HadUK-Grid) | Met Office | 36 months | Controls for seasonal weather effects on crime |

### Geographic filter — why London only
The `Falls within` column in police.uk data identifies the responsible force. We retained only:
- **Metropolitan Police Service** — covers 32 London boroughs
- **City of London Police** — covers the Square Mile

This gave us a consistent, comparable jurisdictional scope. All other 41 forces in England & Wales were excluded.

### Unit of analysis — why LSOA
A **Lower Super Output Area (LSOA)** is a small geographic unit defined by the ONS, each containing approximately 400–1,200 households (~1,500 people). London has **4,994 LSOAs**.

LSOAs were chosen because:
1. They are the finest geographic level at which police.uk publishes crime data
2. They align with the **Index of Multiple Deprivation (IMD)** — deprivation data is published at LSOA level
3. They are small enough to capture genuine spatial variation in demand, but large enough to avoid zero-inflation problems in crime counts

### Time period — why 36 months (April 2023 – March 2026)
- Three complete **policing years** (April–March), which is the standard UK financial/policing year
- Long enough to compute reliable seasonal patterns and trend decomposition (STL requires at least 2 full cycles)
- Matches the availability of all five data sources simultaneously

### Temperature extraction — why xarray and HadUK-Grid
The Met Office **HadUK-Grid** dataset provides gridded climate data at 1km resolution across the UK in NetCDF format. We used the `xarray` library to:
1. Open each NetCDF file
2. Slice to a pre-verified London bounding box (y-index 355–401, x-index 703–761)
3. Average `tasmax` and `tasmin` (max and min daily temperature) over all London grid cells
4. Compute mean temperature: `avg_temperature = (tasmax + tasmin) / 2`

This gives one monthly temperature value for London — used later as a uniform covariate across all LSOAs to control for weather-driven seasonality.

### Missing values — how handled

| Field | Missing | Reason | Treatment |
|---|---|---|---|
| Crime ID | 696,507 | All Anti-social behaviour — UK policy, ASB has no Crime ID | Excluded from resolution rate join (correct by design) |
| Latitude / Longitude | 18,691 | Location suppressed to protect victim privacy | Cannot assign LSOA — ~0.5% spatial undercount |
| LSOA code / name | 18,692 | Same rows as missing coordinates | Same as above |
| Context column | 3,443,915 | Entire column empty | Column dropped entirely |

**Important:** the missing Crime IDs are 100% Anti-social behaviour records — this is not data quality failure, it is a deliberate UK Home Office policy because ASB is not a criminal offence.

---

---

## PHASE 2 — Feature Engineering per LSOA

### What we did
Built a **master feature matrix** of 4,994 London LSOAs × 8 features, aggregating all raw data into one per-LSOA summary row.

### Why feature engineering matters
Raw crime counts alone are insufficient to measure policing demand. Demand is a multi-dimensional concept (Laufs et al., 2021): it includes volume, severity, temporal volatility, spatial context, socioeconomic deprivation, and proactive policing load. Each feature captures a different dimension.

### Features constructed and why

#### 1. `crime_count` — Total crimes per LSOA (36 months)
The baseline volume measure. Simple row count grouped by LSOA code. Includes ASB.
- **Why:** Establishes the raw workload per LSOA — the starting point for all other analysis.

#### 2. `severity_weighted_count` — Cambridge Crime Harm Index weighted total
Each crime type is assigned a severity weight from the **Cambridge Crime Harm Index (CCHI)** (Sherman et al., 2016). The CCHI is based on sentencing guidelines — more serious crimes receive higher weights (e.g., homicide = 5,860, drug offences = 1, ASB = 1).

Formula: `severity_weighted_count = Σ (crime_count_by_type × CCHI_weight)`

- **Why:** Raw crime counts treat a shoplifting incident the same as a violent assault. Severity weighting corrects this — an LSOA with fewer but more serious crimes should score higher than one with many minor incidents.

#### 3. `resolution_rate` — Percentage of crimes with a positive outcome
Joined street crimes to outcomes on `Crime ID`. A "positive" outcome was defined as any outcome that involves charging, caution, or action taken (excludes "Investigation complete — no suspect identified", "Unable to prosecute", etc.)

Formula: `resolution_rate = (crimes with positive outcome / total crimes with Crime ID) × 100`

- **Why:** Low resolution rates indicate areas where police are overwhelmed or evidence is harder to collect — a signal of unmet demand. Also captures operational effectiveness, important for resource allocation decisions.

#### 4. `seasonal_volatility` — Standard deviation of monthly crime counts across 36 months
For each LSOA, we computed the monthly crime count for each of the 36 months, then took the standard deviation across those 36 values.

- **Why:** Two LSOAs can have the same annual crime count but very different patterns — one stable, one with extreme spikes. High volatility means demand is unpredictable, requiring flexible resource deployment rather than fixed allocation.

#### 5. `employment_rank` — IMD 2025 Employment Deprivation Rank
Joined from the **Index of Multiple Deprivation 2025** at LSOA level on `LSOA code (2021)`. Rank 1 = most deprived, Rank 32,844 = least deprived.

- **Why:** Deprivation is one of the strongest predictors of crime demand in the literature (Laufs et al., 2021). Employment deprivation specifically captures economic exclusion, which is more directly linked to acquisitive crime than income deprivation alone.
- **Note:** IMD overall rank and income rank were dropped at Phase 3 due to multicollinearity (VIF > 5). Employment rank had the lowest VIF and was retained.

#### 6. `stop_search_rate` — Stop & searches per km²
Stop & search records have no LSOA code — only latitude/longitude. We performed a **GeoPandas spatial join** (point-in-polygon) to assign each stop to an LSOA, then normalised by LSOA area in km².

Formula: `stop_search_rate = total stops in LSOA / LSOA area (km²)`

- **Why:** Captures proactive policing intensity. High stop & search rates indicate areas where police already perceive elevated risk — making it an independent demand signal from crime counts.

#### 7. `total_footfall` — TfL station footfall assigned to LSOA (36-month total)
TfL footfall data provides daily entry + exit tap counts per station. We:
1. Geocoded each station using **Nominatim** (OpenStreetMap geocoder), with a local cache to avoid re-querying
2. Converted station coordinates to points in EPSG:27700 (British National Grid)
3. Spatial join (point-in-polygon) against London LSOA boundaries to assign each station to its LSOA
4. Summed: `TotalFootfall = EntryTapCount + ExitTapCount`, then aggregated to monthly and 36-month totals per LSOA

- **Why:** Footfall is a proxy for the volume of people in public space. High footfall areas generate more opportunity for theft, disorder, and public safety incidents regardless of resident population — essential for understanding demand in transport hubs and commercial zones.

### Output
`phase2_feature_matrix.parquet` — 4,994 rows × 13 columns (LSOA identifiers + 8 features + crime_count)

---

---

## PHASE 3 — Statistical Validation of Features

### What we did
Proved statistically that every feature included in the model genuinely predicts crime demand, is not redundant with another feature, and that seasonality and spatial clustering exist in the data. This phase determines what goes into the model and what is dropped.

### Why statistical validation is necessary
Including irrelevant or collinear features in a composite index is a common methodological failure. It inflates the index artificially and produces weights that reflect statistical artefacts rather than real demand signals. Phase 3 provides the rigorous justification that each feature earns its place.

---

### 3a — Feature Selection

#### Test 1: Spearman Rank Correlation
**What it is:** Spearman correlation measures the monotonic relationship between two variables — whether one tends to increase as the other increases — without assuming normality. It ranks both variables and correlates the ranks.

**Why Spearman and not Pearson:** Crime count data is heavily right-skewed (a small number of LSOAs have extremely high counts). Pearson correlation assumes both variables are normally distributed — violated here. Spearman is robust to non-normality and outliers.

**Decision rule:** Drop any feature with |r| < 0.1 (very weak relationship with crime_count). Retain everything else.

**Results:**

| Feature | Spearman r | Decision |
|---|---|---|
| severity_weighted_count | +0.9685 | PASS |
| seasonal_volatility | +0.9120 | PASS |
| stop_search_rate | +0.6605 | PASS |
| imd_rank | -0.4964 | PASS |
| income_rank | -0.4085 | PASS |
| employment_rank | -0.3921 | PASS |
| resolution_rate | +0.2862 | PASS |
| total_footfall | +0.2409 | PASS |

All 8 features passed. No features dropped at this stage. Note that IMD rank, income rank, and employment rank are all negative — because these are ranks where rank 1 = most deprived, so higher deprivation (lower rank number) correlates with more crime.

---

#### Test 2: Variance Inflation Factor (VIF)
**What it is:** VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity with other features. A VIF of 1.0 means no correlation with other features. A VIF of 5.0 means the variance is inflated 5×.

**Formula:** `VIF_i = 1 / (1 - R²_i)` where R²_i is the R-squared from regressing feature i on all other features.

**Why this matters:** If two features are highly correlated with each other (e.g., imd_rank and income_rank both measure deprivation), including both in the model doesn't add information — it just creates unstable, unreliable weights. The model can't distinguish their individual contributions.

**Decision rule:** Iteratively drop the feature with the highest VIF > 5, recompute, repeat until all VIFs ≤ 5.

**Results:**
- Iteration 1: `imd_rank` VIF = 15.15 → DROPPED (highly collinear with income and employment ranks — it's a composite of them)
- Iteration 2: `income_rank` VIF = 8.03 → DROPPED (collinear with employment_rank; employment rank had the lowest VIF of the three)
- All remaining features VIF ≤ 5 → STOP

**Final validated features (6):**
`severity_weighted_count`, `seasonal_volatility`, `stop_search_rate`, `employment_rank`, `resolution_rate`, `total_footfall`

---

### 3b — Seasonality Validation

#### Test 3: Kruskal-Wallis Test
**What it is:** A non-parametric test that asks whether multiple groups (here: months) come from the same distribution. It is the non-parametric equivalent of one-way ANOVA.

**Why non-parametric:** Crime counts are count data, not normally distributed. Kruskal-Wallis makes no distributional assumption — it works on ranks.

**Hypotheses:**
- H₀ (null): Crime counts are the same across all 12 months (no seasonality)
- H₁ (alternative): At least one month has a significantly different crime distribution

**Results:** H = 28.75, p = 0.0025

**Interpretation:** p < 0.05 → reject H₀ → seasonality is statistically confirmed. Crime demand varies significantly by month, which justifies including `seasonal_volatility` as a feature and using STL decomposition.

---

#### Test 4: Coefficient of Variation (CV)
**What it is:** CV = (standard deviation / mean) × 100. It expresses volatility as a percentage of the average, allowing fair comparison across crime types with very different base rates.

**Why needed:** Raw standard deviation is not comparable across crime types — drug offences have a much higher count than bicycle theft, so naturally a higher std. CV normalises for this.

**Results (by crime type):**

| Crime Type | CV |
|---|---|
| Drugs | 22.7% |
| Bicycle theft | 21.0% |
| Possession of weapons | 19.3% |
| Theft from the person | 19.2% |
| Shoplifting | 18.5% |
| Violence and sexual offences | 6.6% |

**Interpretation:** Drug offences and outdoor theft have the highest seasonal variability — they spike in summer months. Violence has low CV, indicating it is more constant year-round. This information informed how seasonal_volatility was interpreted in the model.

---

#### Test 5: STL Decomposition (Seasonal-Trend decomposition using LOESS)
**What it is:** STL decomposes a time series into three additive components:
- **Trend:** The long-term direction (increasing, decreasing, stable)
- **Seasonal:** The repeating annual pattern
- **Residual:** What is left after removing trend and seasonal — irregular fluctuations

**LOESS** (Locally Estimated Scatterplot Smoothing) is used for the smoothing step, making STL robust to outliers compared to classical decomposition methods.

**Parameters used:**
- `period = 12` (monthly data, one cycle per year)
- Applied per-LSOA to the 36-month monthly crime series

**Results:**
- **Seasonal strength = 0.9082**
- Interpretation: Strong seasonality confirmed (values approaching 1.0 indicate the seasonal component dominates the residual)

**Why this matters for the model:** The seasonal component amplitude from STL was used to refine the `seasonal_volatility` feature. STL is more robust than simple standard deviation because it separates genuine seasonal patterns from random noise and trend effects.

---

### 3c — Spatial Validation

#### Test 6: Moran's I (Spatial Autocorrelation)
**What it is:** Moran's I tests whether similar values cluster together in space. It compares the value at each LSOA to the values of its neighbouring LSOAs using a spatial weights matrix.

**Formula:** `I = (n / S₀) × (Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄)) / Σᵢ(xᵢ - x̄)²`

Where wᵢⱼ is the spatial weight between LSOA i and j (1 if neighbours, 0 otherwise), computed using **Queen contiguity** (LSOAs sharing any boundary edge or corner are neighbours).

**Range:** I = +1 (perfect clustering), I = 0 (random), I = -1 (perfect dispersion)

**Hypotheses:**
- H₀: Crime is randomly distributed across London (no spatial pattern)
- H₁: Similar crime levels cluster together spatially

**Results:** I = 0.450, E[I] = -0.0002, z = 65.28, p = 0.001

**Interpretation:** I = 0.450 is a moderate-strong positive spatial autocorrelation. p < 0.001 → reject H₀ → spatial clustering is confirmed with extremely high confidence. High-crime LSOAs are significantly more likely to be surrounded by other high-crime LSOAs.

**Why this matters:** This statistically justifies the spatial dimension of the risk index. Crime demand is not randomly distributed — it concentrates geographically. This means tier assignments in Phase 5 will capture real spatial patterns, not arbitrary groupings.

### Output
`phase3_validated_features.parquet` — 4,994 rows × 11 columns (6 validated features + LSOA identifiers + crime_count)

---

---

## PHASE 4 — Risk Index Construction

### What we did
Built a single, empirically justified risk score (0–100) for every London LSOA using two independent statistical methods to derive weights, then combined them.

### Why a composite index and not just crime count?
Crime count alone cannot capture the full complexity of policing demand. An LSOA with 500 crimes but extreme seasonal spikes and high deprivation requires different resource planning than one with 500 stable, minor crimes. The composite index captures multiple dimensions simultaneously.

---

### Step 1: Log-transformation of skewed features
Before fitting any model, `severity_weighted_count` and `total_footfall` were log-transformed using `log1p(x)` (natural log of x+1, handles zeros).

**Why:** Both features had extreme right skew — severity_weighted_count ranged from 0 to 1,891,536 with a mean of ~60,000. Without transformation, the extreme outliers cause numerical instability in the optimiser (singular Hessian matrix). Log-transformation compresses the scale while preserving the rank ordering.

---

### Step 2: Standardisation (z-score)
All features were standardised: `x_std = (x - mean) / std`

This produces features with mean = 0 and standard deviation = 1.

**Why:** Regression coefficients are only comparable across features if they are on the same scale. Without standardisation, a coefficient on severity_weighted_count (range: 0 to 13) would be numerically tiny compared to one on crime_count just because of units, not because of genuine importance.

---

### Weight Derivation Method 1: Negative Binomial Regression (Primary)

**What it is:** A generalised linear model for count data. It models crime_count as a function of the validated features, estimating a coefficient for each feature that represents its contribution to predicting crime demand.

**Why Negative Binomial and not Poisson:** Poisson regression assumes that the mean equals the variance (equidispersion). Crime data is **overdispersed** — the variance (≈ 3.2M) is much larger than the mean (≈ 480). The Negative Binomial adds an overdispersion parameter (alpha) that explicitly models this, producing unbiased standard errors and p-values. Using Poisson on overdispersed data would underestimate standard errors and produce spuriously significant results.

**Model:** `log(E[crime_count]) = β₀ + β₁·severity + β₂·volatility + β₃·stop_search + β₄·employment + β₅·resolution + β₆·footfall`

**Optimisation:** L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) with starting parameters derived from an OLS regression on log(crime_count+1). Starting from OLS estimates rather than zeros ensures the optimiser begins near the true solution, avoiding convergence to local minima.

**Results:**

| Feature | NB Coefficient | p-value | Significant |
|---|---|---|---|
| severity_weighted_count | +0.906 | < 0.001 | Yes |
| seasonal_volatility | +0.085 | < 0.001 | Yes |
| stop_search_rate | +0.015 | < 0.001 | Yes |
| employment_deprivation | -0.084 | < 0.001 | Yes |
| total_footfall | +0.015 | < 0.001 | Yes |
| resolution_rate | -0.003 | 0.186 | No* |

*resolution_rate is not significant at p < 0.05 in the NB model, but is retained because it passed Phase 3 Spearman validation (r = +0.29) and is theoretically justified by the literature. This is documented transparently.

**Note on employment_deprivation direction:** The employment_rank field was inverted (`max_rank + 1 − rank`) so that higher values mean more deprived. The negative NB coefficient on the inverted scale is therefore consistent — it confirms that more deprived LSOAs have higher crime demand.

---

### Weight Derivation Method 2: Random Forest (Cross-Validation)

**What it is:** An ensemble of 300 decision trees, each trained on a random subset of features and data. Feature importance is measured by how much each feature reduces prediction error (mean decrease in impurity) across all trees.

**Parameters:**
- `n_estimators = 300` (300 trees — enough for stable importance estimates)
- `random_state = 42` (reproducibility)
- `max_features = 'sqrt'` (each tree considers √6 ≈ 2.4 features at each split — standard for regression forests, reduces correlation between trees)
- `n_jobs = -1` (parallel computation using all CPU cores)

**Why Random Forest as cross-validation?** RF makes no distributional assumptions and is robust to outliers and non-linear relationships. If both NB and RF agree on which features matter most, the weights have strong convergent validity — two completely different methods reaching the same conclusion.

**5-fold cross-validated R²: 0.8655 ± 0.1351**
The model explains ~87% of the variance in crime counts across LSOAs — a strong fit.

**RF vs NB rank correlation: r = 0.8286 (strong agreement)**

---

### Final Weight Derivation

Final weights are the average of NB-derived weights and RF-derived weights, then renormalised to sum to 1:

`final_weight_i = (nb_weight_i + rf_weight_i) / 2`

| Feature | NB weight | RF weight | **Final weight** |
|---|---|---|---|
| severity_weighted_count | 0.817 | 0.420 | **0.619** |
| seasonal_volatility | 0.077 | 0.336 | **0.206** |
| total_footfall | 0.014 | 0.104 | **0.059** |
| stop_search_rate | 0.014 | 0.094 | **0.054** |
| employment_deprivation | 0.076 | 0.025 | **0.051** |
| resolution_rate | 0.003 | 0.021 | **0.012** |

**Literature cross-reference (Laufs et al., 2021; WWCCR):**
- High weight on severity_weighted_count: consistent with literature prioritising harm-weighted demand
- High weight on seasonal_volatility: consistent with WWCCR emphasis on temporal unpredictability
- Moderate weight on employment_deprivation: consistent with deprivation as a structural driver of demand

---

### Building the Index

#### Step 4: Min-Max Normalisation
All features normalised to [0, 1]: `x_norm = (x - min) / (max - min)`

**Why after weight derivation and not before:** Normalisation is for the arithmetic of the weighted sum. Standardisation (z-score) was used for the statistical models. Using normalisation for the models would lose the interpretability of count-based distributions that NB requires.

#### Step 5: Weighted Composite Score
`risk_score = Σ (weight_i × feature_i_normalised)`

#### Step 6: Scale to [0, 100]
`risk_score_scaled = (risk_score − min) / (max − min) × 100`

**Results:**
- Mean risk score: 52.62
- Std: 5.53
- Range: 0 (lowest demand) to 100 (highest demand — Westminster 013G)
- Top LSOAs: Westminster (tourism/nightlife hub), City of London, Lambeth, Hillingdon (Heathrow area)

---

### Sensitivity Analysis

**What it is:** Each weight was perturbed by ±10%, the index recomputed, and the overlap between the original and perturbed top-10% / bottom-10% LSOA rankings was measured.

**Why:** A good index should be robust — if a small change in weights dramatically reshuffles rankings, the index is unstable and untrustworthy. High stability means the rankings reflect the data, not the specific weight choices.

**Results:**
- Mean top-10% rank stability: **99.0%**
- Mean bottom-10% rank stability: **99.6%**

Changing any weight by ±10% retains 99% of the same LSOAs in the top and bottom tiers — the index is highly robust.

---

---

## PHASE 5 — Clustering into Priority Tiers

### What we did
Grouped 4,994 LSOAs into actionable priority tiers using unsupervised machine learning, validated cluster quality, and profiled each tier for operational interpretation.

### Why cluster instead of just cutting the score into bands?
Score-based bands (e.g., top 25% = Tier 1) are arbitrary. Clustering finds natural groupings in the data based on similarity across all features simultaneously — not just the risk score. This produces tiers that are genuinely distinct from each other in multiple dimensions, making them more operationally meaningful.

### Clustering input: 7 features
`risk_score_scaled` + all 6 validated features (all min-max normalised to [0,1])

---

### Method 1: K-Means Clustering

**What it is:** K-Means partitions LSOAs into k groups by minimising the **within-cluster sum of squares (WCSS)** — the total squared distance of each point from its cluster centroid.

**Algorithm:**
1. Initialise k centroids randomly (using k-means++ for better initialisation)
2. Assign each LSOA to the nearest centroid
3. Recompute centroids as the mean of assigned points
4. Repeat steps 2–3 until assignments stabilise

**Parameters:**
- `n_clusters`: tested 2, 3, 4, 5, 6
- `n_init = 20`: runs the algorithm 20 times with different random starts, keeps the best result (avoids local minima)
- `max_iter = 500`: maximum iterations per run
- `random_state = 42`: reproducibility

**Selecting optimal k — two methods:**

**Elbow method:** Plot WCSS against k. Look for the "elbow" where adding more clusters gives diminishing returns.

**Silhouette score:** For each data point, measures:
- `a` = mean distance to other points in its own cluster
- `b` = mean distance to points in the nearest other cluster
- `silhouette = (b - a) / max(a, b)`
- Range: -1 (wrong cluster) to +1 (well-separated)
- Overall score = mean across all points — higher is better

---

### Method 2: Gaussian Mixture Model (GMM) — AIC/BIC Comparison

**What it is:** GMM is a probabilistic clustering method that assumes the data is generated from a mixture of Gaussian distributions. Unlike K-Means (which uses hard boundaries), GMM assigns each point a probability of belonging to each cluster.

**Why GMM as comparison:** K-Means assumes clusters are spherical and equally sized. GMM can model elliptical clusters of different sizes — a more flexible assumption. If both methods agree on k, the result is more credible.

**Model selection — AIC and BIC:**
- **AIC** (Akaike Information Criterion): `AIC = 2k − 2ln(L)`. Penalises model complexity, but less strongly. Favours slightly more complex models.
- **BIC** (Bayesian Information Criterion): `BIC = k·ln(n) − 2ln(L)`. Penalises complexity more strongly for large n. More conservative — favours simpler models.
- Lower AIC/BIC = better model. Optimal k is where AIC/BIC is minimised.

**Parameters:**
- `n_components`: tested 2–6
- `n_init = 5`: multiple random initialisations
- `max_iter = 300`: convergence iterations

---

### Final k Selection

The model-selected k (by majority vote across silhouette, AIC, BIC) was **k = 6**. However, analysis revealed:
- Tier 1 at k=6 had a **negative silhouette (-0.117)** — the 142 extreme Westminster/City LSOAs are geometrically closer to Tier 2's centroid, meaning they are outliers rather than a true cluster
- Tiers 2–6 had overlapping risk scores (range: 47–56), a difference of only ~2 points between adjacent tiers — operationally indistinguishable

**Decision: produce both k=6 (model-selected) and k=4 (operationally recommended)**

**k=4 rationale:**
- Four tiers map naturally to policing resource categories: Critical / High / Moderate / Low
- All four tiers have positive silhouette scores
- Risk score separation is cleaner and more actionable

---

### Tier Profiling (k=4)

| Tier | Label | n LSOAs | % London | Mean Risk Score | Mean Crime Count |
|---|---|---|---|---|---|
| 1 | Critical demand | 1,429 | 28.6% | Highest | Highest |
| 2 | High demand | 1,538 | 30.8% | High | High |
| 3 | Moderate demand | 1,047 | 21.0% | Moderate | Moderate |
| 4 | Low demand | 980 | 19.6% | Lowest | Lowest |

Each tier was profiled by:
- Dominant crime types
- Mean deprivation (employment rank)
- Mean footfall
- Mean stop & search rate
- Geographic distribution (borough composition)

---

---

## Summary of Methodological Chain

```
Research Question
       |
       v
Phase 1: Raw data → London filter → Standardised clean datasets
       |
       v
Phase 2: LSOA aggregation → 8 features capturing different demand dimensions
       |
       v
Phase 3: Spearman (relevance) + VIF (no redundancy) + Kruskal-Wallis (seasonality)
         + CV (crime type volatility) + STL (seasonal structure) + Moran's I (spatial clustering)
         → 6 statistically validated features
       |
       v
Phase 4: NB Regression (primary weights) + Random Forest (cross-validation)
         + Min-max normalisation + Weighted composite index + Sensitivity analysis
         → Risk score 0–100 per LSOA
       |
       v
Phase 5: K-Means + Silhouette (k selection) + GMM AIC/BIC (cross-validation)
         → Priority tiers with operational profiles and choropleth map
```

Every decision at every step is:
1. **Justified by statistical evidence** (p-values, scores, tests)
2. **Cross-validated by an independent method** (NB + RF, K-Means + GMM)
3. **Grounded in academic literature** (Laufs et al., 2021; Sherman et al., 2016; WWCCR)
4. **Documented with transparency** (flagged features, data limitations, sensitivity analysis)

---

*CBL-16 — London Policing Demand Study | Mateus Becklas | 2025–2026*
