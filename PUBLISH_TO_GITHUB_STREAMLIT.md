# Panduan VOISS: unggah model ke GitHub lalu deploy ke Streamlit Cloud

Repository yang sudah disiapkan:
https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier

## Status saat ini

- Source code Streamlit, dependencies, metadata, konfigurasi, dan dokumentasi sudah berada di repository.
- Tiga file .keras sudah memiliki pointer Git LFS di folder artifacts.
- Objek biner Git LFS belum dikirim. Langkah satu kali ini harus dijalankan dari komputer pemilik file ZIP model.

## Satu kali unggah model (Windows)

### Persiapan
Install:
- Git for Windows: https://git-scm.com/download/win
- Git LFS: https://git-lfs.com/

Pastikan file seismic_deployment_artifacts.zip tersedia di Downloads.

### Jalankan perintah ini di PowerShell

Salin seluruh blok berikut dan jalankan:

```powershell
git lfs install
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd voiss-volcanic-earthquake-classifier
Set-ExecutionPolicy -Scope Process Bypass
.\PUSH_MODELS_TO_GITHUB.ps1 "$env:USERPROFILE\Downloads\seismic_deployment_artifacts.zip"
```

Jika nama/path ZIP berbeda, ganti bagian terakhir dengan lokasi sebenarnya.

**Mengapa GIT_LFS_SKIP_SMUDGE diperlukan?** Repository saat ini sudah berisi pointer model LFS, tetapi objek biner model baru akan diunggah oleh script. Variabel itu mencegah Git mencoba mengambil model sebelum objek tersedia.

Script akan:
1. Mengekstrak tiga model dari ZIP ke artifacts/.
2. Memastikan ketiga model terdaftar pada Git LFS.
3. Mengunggah objek biner melalui Git LFS ke repository.
4. Menampilkan pesan sukses jika unggahan selesai.

Jangan upload ZIP model ke repository. Yang diunggah adalah tiga file .keras sebagai objek LFS.

## Deploy ke Streamlit Community Cloud

Setelah script model menampilkan pesan sukses:

1. Buka https://share.streamlit.io/ dan login dengan GitHub.
2. Klik Create app.
3. Pilih repository Jfkevin/voiss-volcanic-earthquake-classifier.
4. Branch: main.
5. Main file path: app.py.
6. Pada Advanced settings pilih Python 3.12.
7. Klik Deploy dan pantau build logs.

## Kontrak input model

- Sampling rate setelah preprocessing: 100 Hz.
- Satu segmen inferensi: 60 detik / 6.000 sampel.
- Waveform: (6000, 1).
- Spectrogram: (96, 60, 1).
- Format: CSV, TXT, WAV, MiniSEED, H5, HDF5.
- Rekaman lebih dari 60 detik dapat dipilih segmennya; rekaman lebih pendek ditolak.
- Amplitudo waveform tidak dinormalisasi otomatis; skalanya harus konsisten dengan data training.

Prediction History pada versi demo bersifat sementara per sesi browser. Skor Softmax belum dibuktikan terkalibrasi. VOISS adalah prototipe penelitian akademik, bukan sistem peringatan dini operasional.
