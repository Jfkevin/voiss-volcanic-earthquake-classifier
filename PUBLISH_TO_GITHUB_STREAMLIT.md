# Publish VOISS to Streamlit Community Cloud

Repository: https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier

## Model files (one-time action)

The three model files are tracked with Git LFS. GitHub's normal repository API cannot upload the LFS binary objects for you, so run the included PowerShell helper after cloning the repository:

1. Install Git for Windows and Git LFS.
2. Download your `seismic_deployment_artifacts.zip` from Colab.
3. Clone the repo.
4. Run `PUSH_MODELS_TO_GITHUB.ps1` from the cloned repo, passing the full path to that ZIP.

Example:

```powershell
git clone https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier.git
cd voiss-volcanic-earthquake-classifier
Set-ExecutionPolicy -Scope Process Bypass
.\PUSH_MODELS_TO_GITHUB.ps1 "$env:USERPROFILE\Downloads\seismic_deployment_artifacts.zip"
```

The helper extracts the 3 .keras files to artifacts/, tracks them with Git LFS, verifies all three paths, then commits and pushes the model artifacts to main.

## Deploy the web app

1. Sign in at https://share.streamlit.io/ using your GitHub account.
2. Click Create app.
3. Select repository `Jfkevin/voiss-volcanic-earthquake-classifier`, branch `main`, entrypoint `app.py`.
4. Select Python 3.12 in Advanced settings.
5. Deploy and wait for the app to install dependencies and retrieve Git LFS assets.

## Input expectations

- 100 Hz, 60 s, 6,000 samples.
- Waveform input: (6000, 1).
- Spectrogram input: (96, 60, 1).
- Formats: CSV, TXT, WAV, MiniSEED, H5, HDF5.
- Signals longer than 60 seconds can be segmented; shorter signals are rejected.
- Waveform amplitude scale must be consistent with the training data. No amplitude normalization is applied.

## Notes

Prediction history is session-scoped, not persistent. Softmax values are model scores and are not guaranteed calibrated probabilities. This is an academic research prototype, not an operational warning system.
