# Publish VECAS to Streamlit Community Cloud

Application name: **VECAS — Volcanic Earthquake Classification and Analysis System**

Tagline: **Intelligent Seismic Analysis for Volcanic Earthquake Classification**

Repository:
https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier

## Current deployment

The public Streamlit app is deployed from branch `main`, with `app.py` as its entry point.

## Architecture naming

VECAS is the name of the application. The model names remain:
- SeismicNet (CNN 1D)
- VOISS-Net (CNN 2D)
- Seismic-VOISS-Net Multimodal CNN

## Streamlit Community Cloud settings

- Repository: `Jfkevin/voiss-volcanic-earthquake-classifier`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

## Model input contract

- Sampling rate after preprocessing: 100 Hz.
- One inference segment: 60 seconds / 6,000 samples.
- Waveform input: `(6000, 1)`.
- Spectrogram input: `(96, 60, 1)`.
- Supported formats: CSV, TXT, WAV, MiniSEED, H5, HDF5.
- Signals longer than 60 seconds can be segmented; shorter signals are rejected.
- Waveform amplitude is not normalized automatically; its scale must be consistent with the training data.

## Researcher profile photo

The About page reads an optional `profile_photo_base64` value from Streamlit Community Cloud Secrets. This keeps the photo out of the public GitHub repository. Because the photo is displayed on the public About page, visitors can still view it.

## Notes

Prediction History is temporary and scoped to the current browser session. Softmax values are displayed as model scores/confidence and have not been established as calibrated probabilities. VECAS is an academic research and portfolio prototype, not an operational volcano early-warning system.
