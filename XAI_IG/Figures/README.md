# Figures

This folder contains the plots generated from the Integrated Gradients attribution results.

All figures are produced from the aggregated CSV files stored in the `results/` folder.

---

## Contents

- `global_feature_importance.png`  
  Global average importance of all input variables.

- `timestep_importance.png`  
  Mean attribution per historical time step (t, t−1, t−2, t−3).

- `importance_by_horizon.png`  
  Feature importance across different prediction horizons.

- `importance_by_street.png`  
  Feature importance across different streets.

- `attribution_pca.png`  
  PCA projection of normalized attribution vectors.

- `importance_by_variable_group.png`  
  Average importance grouped by variable type (calendar, flow, weather).

---

## Notes

- Figures are generated using aggregated attribution values.
- Each dataset is normalized independently.
- Comparisons across datasets should be interpreted in relative terms only.
