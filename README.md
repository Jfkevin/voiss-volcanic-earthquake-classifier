# VOISS — Volcanic Earthquake Classification System

**VOISS** is a Streamlit research prototype for volcanic seismic-event classification using three trained CNN architectures:

1. **SeismicNet (CNN 1D)** — waveform input.
2. **VOISS-Net (CNN 2D)** — spectrogram input.
3. **Seismic-VOISS-Net Multimodal CNN** — waveform + spectrogram with feature fusion.

The app includes waveform/spectrogram visualization, four-class outputs (LP, TC, TR, VT), Compare All model agreement, event information, session-scoped prediction history, researcher profile, and model documentation.

## Repository contents

- `app.py` — Streamlit application entry point.
- `requirements.txt` — Python dependencies.
- `runtime.txt` — Python 3.12 runtime.
- `artifacts/` — metadata, preprocessing information, and Git LFS pointers for the three trained models.
- `PUSH_MODELS_TO_GITHUB.ps1` — one-time Windows helper to upload the model binaries to Git LFS.
- `PUBLISH_TO_GITHUB_STREAMLIT.md` — publishing guide.

## Important: upload the model binaries

The three model binaries are large (about 296 MiB total); two are larger than GitHub's 100 MB regular-file limit. Their Git LFS pointer files are already committed, but the actual LFS binary objects still need to be uploaded from your downloaded `seismic_deployment_artifacts.zip`.

Install [Git for Windows](https://git-scm.com/download/win) and [Git LFS](https://git-lfs.com/), then open PowerShell and run:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd voiss-volcanic-earthquake-classifier
Set-ExecutionPolicy -Scope Process Bypass
.\PUSH_MODELS_TO_GITHUB.ps1 "$env:USERPROFILE\Downloads\seismic_deployment_artifacts.zip"
```

The `GIT_LFS_SKIP_SMUDGE` setting is important because the repository currently contains LFS pointers before their binary objects are uploaded. The helper copies the three model files from the ZIP and uploads their contents to Git LFS.

Check that all three files appear in `git lfs ls-files` and wait for the helper to report success.

## Deploy to Streamlit Community Cloud

After the model upload succeeds:

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Choose **Create app**.
3. Repository: `Jfkevin/voiss-volcanic-earthquake-classifier`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. In Advanced settings, choose **Python 3.12**.
7. Click Deploy and monitor the build logs.

Community Cloud supports Git LFS repositories, so the app can read the model files from the repository at startup. See [Streamlit's file-organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Model input contract

- Sampling rate after resampling: 100 Hz.
- Inference segment: 60 seconds / 6,000 samples.
- Waveform shape (without batch): `(6000, 1)`.
- Spectrogram shape (without batch): `(96, 60, 1)`.
- Supported uploads: CSV, TXT, WAV, MiniSEED, H5, HDF5.
- Shorter-than-60-second signals are rejected; longer signals can be segmented.
- Waveform amplitude is not normalized automatically. The input amplitude scale must be consistent with training data.

## Classes

- LP — Long Period
- TC — Tectonic
- TR — Tremor
- VT — Volcano-Tectonic

## Notes and limitations

- Prediction history is temporary and scoped to the current Streamlit browser session.
- Softmax scores are shown as model scores/confidence; they have not been established as calibrated probabilities.
- The models were developed using the Llaima volcanic seismic dataset. Generalization to other volcanoes, stations, sensors, or acquisition conditions requires external validation.
- This is an academic research prototype, not an operational volcano early-warning system.

© 2026 Farrelius Kevin. VOISS — Volcanic Earthquake Classification System.
