# Project 03 — Wholesale Customer Segmentation

**Intern:** Darshan Suresh S
**Level:** Intermediate | **Tools:** Pandas, NumPy, Matplotlib, Scikit-learn, SciPy

## Problem

A wholesale distributor serves customers with very different purchasing patterns. Without a labelled target,
this project builds a data-driven unsupervised learning workflow that discovers customer segments from annual
category spending and translates them into understandable, actionable business profiles.

## Dataset

- **Source:** UCI Machine Learning Repository — [Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers)
- **Size:** 440 customers, annual spending across 6 product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen), plus `Channel` and `Region` fields
- Download the dataset from the link above and place it at `data/wholesale_customers.csv`.

## Approach

1. **EDA** — audited distributions, outliers, and skewness across all 6 spending categories. Every column showed severe right-skew (skewness 2.5–11.1), driven by a small number of very large-spending customers.
2. **Feature strategy** — excluded `Channel`/`Region` from clustering inputs (to avoid trivially rediscovering the existing 2-category split), retained them for post-hoc validation.
3. **Transformation** — applied `log1p` to compress skew, then `StandardScaler` to equalize scale across the 6 dimensions.
4. **Cluster count selection** — compared the elbow method and silhouette score across k=2–10; selected **k=3** (second-best silhouette, richer segments than the binary k=2 split).
5. **Clustering** — fit K-Means (primary) and Agglomerative Clustering (comparison) at k=3; silhouette scores were nearly identical (0.259 vs 0.255), but the two algorithms showed only partial agreement — documented honestly as a stability limitation.
6. **PCA** — reduced to 2D for visualization (71.3% of variance explained by the first two components).
7. **Cluster profiling** — original-unit and normalized spending profiles, cluster sizes.
8. **Validation** — cross-tabulated clusters against Channel/Region; Cluster 2 showed a very strong (98.6%) alignment with the Horeca channel.
9. **Business labeling** — named each segment and proposed one action per segment.
10. **Reusable assignment function** — classifies a new customer record into a segment using the saved preprocessing and clustering objects.

## Repository Structure

## How to Run

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter notebook notebooks/03_customer_segmentation.ipynb
# Kernel > Restart & Run All
```

## Key Results

| Cluster | Size | Profile | Label |
|---|---|---|---|
| 0 | 80 | High Grocery & Detergents/Paper, low Fresh/Frozen | Grocery & Household Retailers |
| 1 | 147 | High across nearly every category | High-Volume Bulk Buyers |
| 2 | 213 | High Fresh, very low Milk/Grocery/Detergents | Fresh-Focused Food Service (Horeca) |

Cluster 2 aligns with the Horeca channel in **210 of 213 cases (98.6%)** — strong evidence this segment reflects
genuine business structure. Region showed no strong relationship with any cluster.

## Limitations

- K-Means and Agglomerative Clustering showed only partial agreement — one large segment is consistently found by both methods, but the boundary between the remaining two is less crisp.
- With only 440 customers, smaller/rarer purchasing patterns may not form their own distinct cluster.
- PCA visualization captures 71.3% of variance in 2D — a useful but imperfect simplification of the true 6D structure.

## Dataset Citation

Cardoso, M. (2014). Wholesale customers [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X