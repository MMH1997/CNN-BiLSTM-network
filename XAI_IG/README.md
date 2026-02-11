# Integrated Gradients Explainability

This folder contains additional code and experimental results for applying Integrated Gradients (IG) to the CNN–BiLSTM model provided in this repository.

The goal is to analyse feature relevance and temporal contributions of the model predictions, following the methodology described in our ongoing research work on explainability for hybrid CNN–RNN architectures.

This module is implemented as a post-hoc analysis and does not modify or retrain the original predictive model.

## Structure

- `code/` – Integrated Gradients and analysis scripts  
- `results/` – Raw and aggregated attribution outputs  
- `figures/` – Generated plots  
- `paper/` – Draft / PDF of the associated paper  
- `README.md`

## Notes

- Integrated Gradients is computed over pretrained models.
- A zero baseline is used for attribution.
- Results correspond to the same datasets and experimental setup as the original CNN–BiLSTM model.

This folder is intended for research and reproducibility purposes.
