# Explainability Code

This folder contains the reference Jupyter notebook used to compute Integrated Gradients attributions for the CNN–BiLSTM model.

The notebook implements the full explainability pipeline for a single dataset and serves as a template for all other experimental configurations.

The analysis is performed as a post-hoc procedure and does not modify or retrain the original predictive model.

---

## Notebook

A single reference notebook is provided:

```XAI_AS12.ipynb```


This notebook demonstrates:

- Loading pretrained CNN–BiLSTM models  
- Preparing input data  
- Computing Integrated Gradients attributions  
- Aggregating attribution values across samples and time steps  
- Exporting results to CSV files in the `results/` folder  

Additional experiments (other streets and horizons) follow the same structure and can be obtained by adapting the input paths and configuration parameters.

---

## Output

The notebook generates one aggregated CSV file per dataset, containing attribution values organized by variable and historical time step.

These CSV files are later used to produce the figures included in the `figures/` folder.

---

## Notes

- Integrated Gradients is computed over pretrained models.
- A zero baseline is used for attribution.
- Absolute attribution values are reported.
- Results correspond to the same datasets and experimental setup as the original CNN–BiLSTM model.

This code is provided for research and reproducibility purposes.
