# VECAS — Volcanic Earthquake Classification and Analysis System

**VECAS** is a Streamlit research prototype for volcanic seismic-event classification and signal analysis. Its tagline is **Intelligent Seismic Analysis for Volcanic Earthquake Classification**.

The application demonstrates three CNN-based model approaches:

1. **SeismicNet (CNN 1D)** — processes waveform input.
2. **VOISS-Net (CNN 2D)** — processes spectrogram input.
3. **Seismic-VOISS-Net Multimodal CNN** — combines waveform and spectrogram features through feature fusion.

The application includes waveform and spectrogram visualization, four-class outputs (LP, TC, TR, VT), single-model inference, Compare All model agreement, event information, session-scoped prediction history, researcher profile, and model documentation.

> VECAS is the application/system name. SeismicNet, VOISS-Net, and Seismic-VOISS-Net Multimodal CNN are model/architecture names used in the research implementation.

## Repository contents

- `app.py` — Streamlit application entry point.
- `requirements.txt` — Python dependencies.
- `runtime.txt` — Python 3.12 runtime.
- `artifacts/` — metadata, preprocessing information, and Git LFS pointers for the three trained models.
- `PUSH_MODELS_TO_GITHUB.ps1` — Windows helper for uploading model binaries to Git LFS.
- `PUBLISH_TO_GITHUB_STREAMLIT.md` — publishing and deployment guide.

## Model artifacts

The three trained Keras model binaries are stored using Git LFS. The repository includes their LFS pointers under `artifacts/`.

## Deploy to Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Select repository `Jfkevin/voiss-volcanic-earthquake-classifier`.
3. Branch: `main`.
4. Main file path: `app.py`.
5. Python version: `3.12`.
6. Deploy and monitor the build logs.

## Model input contract

- Sampling rate after resampling: 100 Hz.
- Inference segment: 60 seconds / 6,000 samples.
- Waveform shape (without batch): `(6000, 1)`.
- Spectrogram shape (without batch): `(96, 60, 1)`.
- Supported uploads: CSV, TXT, WAV, MiniSEED, H5, HDF5.
- Signals shorter than 60 seconds are rejected; longer signals can be segmented.
- Waveform amplitude is not normalized automatically. The input amplitude scale must be consistent with training data.

## Classification labels

- LP — Long Period
- TC — Tectonic
- TR — Tremor
- VT — Volcano-Tectonic

## Researcher and project context

VECAS was developed by **Farrelius Kevin** as an academic and portfolio prototype exploring the intersection of Informatics, Artificial Intelligence, and Earth Science. The project investigates deep-learning approaches to volcanic seismic signal classification using the Llaima volcanic seismic dataset from Chile.

## Notes

- Prediction history is temporary and scoped to the current Streamlit browser session.
- Softmax scores are displayed as model scores/confidence; they have not been established as calibrated probabilities.
- Generalization to other volcanoes, stations, sensors, or acquisition conditions requires external validation.
- VECAS is not an operational volcano early-warning system.

© 2026 Farrelius Kevin. VECAS — Volcanic Earthquake Classification and Analysis System.
