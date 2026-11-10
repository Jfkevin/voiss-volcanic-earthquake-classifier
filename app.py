from __future__ import annotations

import io
import base64
import html
import json
import gc
import os
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from fractions import Fraction

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from scipy.signal.windows import tukey

# Keep TensorFlow CPU thread usage modest on shared cloud runtimes.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"

FS = 100.0
N_SAMPLES = 6000
N_CLASSES = 4
CODES = ["LP", "TC", "TR", "VT"]
CLASS_NAMES = {
    "LP": "Long Period",
    "TC": "Tectonic",
    "TR": "Tremor",
    "VT": "Volcano-Tectonic",
}
COLORS = {"LP":"#e45b5b", "TC":"#3985d8", "TR":"#20a36b", "VT":"#e5a13a"}
MODEL_FILES = {
    "SeismicNet": "SeismicNet_final.keras",
    "VOISS-Net": "VOISSNet_final.keras",
    "Seismic-VOISS-Net Multimodal CNN": "SeismicNet_VOISSNet_Multimodal_final.keras",
}
MODEL_LABELS = {
    "SeismicNet": "SeismicNet (CNN 1D)",
    "VOISS-Net": "VOISS-Net (CNN 2D)",
    "Seismic-VOISS-Net Multimodal CNN": "Seismic-VOISS-Net Multimodal CNN",
}
ACCEPTED = ["csv","txt","wav","mseed","msd","h5","hdf5"]

st.set_page_config(
    page_title="VOISS | Volcanic Earthquake Classification",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#10243f; --navy2:#17385d; --blue:#1768b5; --pale:#eef5fd; --line:#dbe6f2; }
.stApp { background:linear-gradient(180deg,#f7f9fc 0%,#edf3fa 100%); color:#132b49; }
[data-testid="stHeader"] { background:rgba(0,0,0,0); height:0.25rem; }
[data-testid="stToolbar"] { right:1rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d2039 0%,#142f50 65%,#10243f 100%); border-right:1px solid #1a3b5f; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.1rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span { color:#e7f0fa; }
[data-testid="stSidebar"] button { border:1px solid transparent !important; border-radius:9px !important; min-height:42px; text-align:left !important; justify-content:flex-start !important; color:#e9f2fb !important; background:transparent !important; }
[data-testid="stSidebar"] button[kind="primary"] { background:linear-gradient(90deg,#1768b5,#2c78c7) !important; color:white !important; border-color:#3c84d1 !important; box-shadow:0 5px 14px rgba(0,0,0,.16); }
[data-testid="stSidebar"] button:hover { background:rgba(83,145,203,.22) !important; border-color:rgba(155,197,235,.22) !important; }
.block-container { max-width:1600px; padding-top:1.05rem; padding-bottom:2rem; }
.voiss-header { position:relative; overflow:hidden; min-height:112px; border-radius:15px; padding:20px 25px; margin:0 0 16px 0; color:#fff; background-size:cover; background-position:center 54%; box-shadow:0 5px 16px rgba(16,45,79,.12); }
.voiss-header:after { content:""; position:absolute; inset:0; background:linear-gradient(90deg,rgba(12,31,55,.94) 0%,rgba(17,48,79,.80) 49%,rgba(17,48,79,.18) 100%); pointer-events:none; }
.voiss-header-inner { position:relative; z-index:1; display:flex; justify-content:space-between; align-items:center; gap:18px; min-height:70px; }
.voiss-header h1 { color:white; font-size:27px; line-height:1.12; margin:0; font-weight:780; letter-spacing:-.4px; }
.voiss-header p { color:#e4eef8; font-size:13px; margin:7px 0 0 0; }
.voiss-time { color:#fff; white-space:nowrap; text-align:right; font-size:13px; font-weight:700; text-shadow:0 1px 5px rgba(0,0,0,.3); }
.sidebar-brand { display:flex; align-items:center; gap:12px; padding:4px 4px 18px 4px; margin-bottom:18px; border-bottom:1px solid rgba(217,234,249,.16); }
.sidebar-brand img { width:49px; height:49px; object-fit:contain; }
.sidebar-brand-title { color:#fff; font-size:24px; line-height:1; font-weight:800; letter-spacing:.5px; }
.sidebar-brand-sub { color:#bfd1e3; font-size:10px; line-height:1.4; margin-top:5px; }
.voiss-card { background:#fff; border:1px solid #dce6f2; border-radius:13px; padding:15px 17px; margin-bottom:12px; box-shadow:0 2px 9px rgba(20,55,95,.045); }
.voiss-muted { color:#63758c; font-size:12px; }
.voiss-panel { background:#fff; border:1px solid #dce6f2; border-radius:13px; padding:15px 17px; margin-bottom:13px; box-shadow:0 2px 9px rgba(20,55,95,.045); }
.panel-title { color:#102b4d; font-size:17px; font-weight:780; margin:0 0 12px 0; }
.hero-card { position:relative; overflow:hidden; border:1px solid #d5e2f1; border-radius:17px; min-height:260px; padding:29px 31px; margin-bottom:17px; background-size:cover; background-position:center 45%; box-shadow:0 4px 15px rgba(16,45,79,.09); }
.hero-card:before { content:""; position:absolute; inset:0; background:linear-gradient(90deg,rgba(248,251,255,.98) 0%,rgba(244,249,255,.92) 42%,rgba(229,240,251,.10) 100%); }
.hero-copy { position:relative; z-index:1; max-width:560px; }
.hero-kicker { font-size:12px; font-weight:800; color:#1c68ad; text-transform:uppercase; letter-spacing:1.1px; }
.hero-title { font-size:38px; color:#123b66; font-weight:850; line-height:1.06; margin:7px 0 8px; }
.hero-subtitle { color:#274c72; font-size:18px; font-weight:700; margin:0 0 9px; }
.hero-text { color:#405b78; max-width:540px; font-size:14px; line-height:1.55; }
.feature-card { background:linear-gradient(180deg,#fff 0%,#f4f8fd 100%); border:1px solid #d8e4f2; border-radius:12px; padding:16px 14px; min-height:130px; height:100%; box-shadow:0 2px 8px rgba(20,55,95,.04); }
.feature-icon { color:#1768b5; font-size:27px; margin-bottom:7px; }
.feature-title { color:#132f51; font-size:15px; font-weight:760; margin-bottom:4px; }
.feature-text { color:#5a6f86; font-size:12px; line-height:1.45; }
.class-chip { background:#fff; border:1px solid #dbe6f2; border-radius:10px; padding:13px 15px; border-top:4px solid var(--chip-color); }
.class-chip-code { color:var(--chip-color); font-size:25px; font-weight:850; line-height:1.1; }
.class-chip-name { color:#213d5d; font-size:12px; font-weight:650; margin-top:5px; }
.section-heading { font-size:20px; font-weight:800; color:#102b4d; margin:6px 0 12px; }
.flow-scroll { width:100%; overflow-x:auto; padding:7px 2px 12px; }
.flow-row { display:flex; flex-wrap:nowrap; align-items:center; gap:7px; min-width:max-content; }
.flow-step { display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; width:112px; min-height:79px; padding:9px 7px; border-radius:9px; border:1px solid #a9c9e7; background:linear-gradient(180deg,#f5faff,#e6f1fc); color:#183d64; box-shadow:0 2px 4px rgba(20,55,95,.04); }
.flow-step.pink { background:linear-gradient(180deg,#fff7f8,#ffe8eb); border-color:#edb3bc; color:#773b46; }
.flow-step.purple { background:linear-gradient(180deg,#faf8ff,#eee9fb); border-color:#c7b8ed; color:#4b3d78; }
.flow-step.green { background:linear-gradient(180deg,#f4fff9,#e2f7eb); border-color:#a9d9bd; color:#285b42; }
.flow-step b { font-size:11px; line-height:1.25; }
.flow-step small { font-size:9px; line-height:1.3; margin-top:4px; opacity:.85; }
.flow-arrow { color:#264b71; font-size:18px; font-weight:800; }
.architecture-card { background:#fff; border:1px solid #dce6f2; border-radius:13px; overflow:hidden; margin-bottom:15px; box-shadow:0 2px 9px rgba(20,55,95,.045); }
.architecture-band { padding:10px 16px; font-size:17px; font-weight:800; color:#102e52; }
.architecture-band.blue { background:linear-gradient(90deg,#d8ebff,#edf6ff); border-bottom:1px solid #c6def4; }
.architecture-band.purple { background:linear-gradient(90deg,#e9e0ff,#f6f1ff); border-bottom:1px solid #d9c9f4; }
.architecture-band.green { background:linear-gradient(90deg,#dff7e9,#effcf4); border-bottom:1px solid #c8ead5; }
.architecture-body { padding:13px 16px 14px; }
.architecture-description { color:#405a75; font-size:13px; line-height:1.55; }
.doc-toc { background:linear-gradient(180deg,#fff,#f6f9fd); border:1px solid #dbe6f2; border-radius:13px; padding:14px; position:sticky; top:8px; }
.doc-toc a { display:block; padding:8px 9px; margin:2px 0; border-radius:7px; color:#21466e !important; text-decoration:none !important; font-size:12px; font-weight:650; }
.doc-toc a:hover { background:#e9f2fc; }
.doc-section { background:#fff; border:1px solid #dce6f2; border-radius:13px; padding:17px 18px; margin-bottom:14px; box-shadow:0 2px 9px rgba(20,55,95,.04); scroll-margin-top:15px; }
.doc-section h3 { color:#143a63; margin:0 0 10px; font-size:18px; }
.profile-hero { background:linear-gradient(115deg,#fff 0%,#eff6ff 55%,#dceafa 100%); border:1px solid #d7e4f3; border-radius:16px; padding:22px; }
.profile-name { color:#123b66; font-size:29px; font-weight:850; margin:0; }
.profile-role { color:#24659f; font-weight:700; margin:5px 0 9px; }
.profile-bio { color:#425d79; line-height:1.6; font-size:13px; }
.profile-placeholder { width:100%; min-height:235px; border-radius:14px; background:linear-gradient(145deg,#dbeafa,#eef5fc); display:flex; align-items:center; justify-content:center; color:#4d7095; font-size:52px; border:1px solid #ccdeef; }
.about-highlight { border:1px solid #dbe6f2; background:#fff; border-radius:12px; padding:15px; min-height:145px; }
.about-highlight h4 { margin:0 0 6px; color:#173b62; font-size:15px; }
.about-highlight p { color:#61758c; font-size:12px; line-height:1.5; margin:0; }
.result-card { background:#fff; border:1px solid #d9e5f2; border-radius:13px; padding:15px; box-shadow:0 2px 8px rgba(20,55,95,.04); height:100%; }
.result-class { border-radius:8px; background:#fff0ef; color:#a52828; font-size:18px; font-weight:800; padding:10px; text-align:center; margin:8px 0; }
.agreement-card { background:linear-gradient(90deg,#e8f9ef,#f4fcf7); border:1px solid #c8ebd6; padding:16px 18px; border-radius:12px; }
div.stButton > button[kind="primary"] { background:#1768c2; border-color:#1768c2; }
div[data-testid="stFileUploader"] { border:1px dashed #9bb8d6; background:#f8fbff; border-radius:10px; padding:7px; }
div[data-testid="stRadio"] > div { gap:8px; }
div[data-testid="stRadio"] label { background:#f4f8fd; border:1px solid #d7e4f2; padding:8px 11px; border-radius:9px; }
@media (max-width: 800px) {
 .voiss-header-inner { align-items:flex-start; flex-direction:column; }
 .voiss-time { text-align:left; }
 .hero-title { font-size:29px; }
 .hero-card { min-height:300px; }
}
</style>
""", unsafe_allow_html=True)
def now_wib():
    return datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%d %b %Y · %H:%M:%S WIB")

def asset_data_uri(filename: str) -> str:
    path = ASSETS / filename
    if not path.exists():
        return ""
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def page_header(title: str, subtitle: str = ""):
    banner = asset_data_uri("volcano-banner.svg")
    style = f'background-image:linear-gradient(90deg,rgba(10,30,54,.95) 0%,rgba(15,47,80,.80) 52%,rgba(15,47,80,.20) 100%),url("{banner}");' if banner else 'background:linear-gradient(100deg,#10243f,#1d4f80);'
    st.markdown(
        f'<div class="voiss-header" style="{style}">'
        f'<div class="voiss-header-inner"><div><h1>VOISS: {html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle)}</p></div>'
        f'<div class="voiss-time">{now_wib()}</div></div></div>',
        unsafe_allow_html=True
    )


def model_status():
    missing = []
    for fn in MODEL_FILES.values():
        path = ARTIFACTS / fn
        if not path.is_file():
            missing.append(fn + " (file belum tersedia)")
            continue
        # GitHub stores a small LFS pointer in Git; the real model must be fetched by Git LFS.
        with path.open("rb") as handle:
            header = handle.read(200)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            missing.append(fn + " (Git LFS object belum diunduh)")
        elif path.stat().st_size < 1024:
            missing.append(fn + " (ukuran file tidak wajar)")
    return missing

@st.cache_resource(show_spinner="Memuat model AI...", max_entries=1)
def load_model_cached(path_str: str, mtime_ns: int, size: int):
    return tf.keras.models.load_model(path_str, compile=False, safe_mode=True)

def load_model(name: str):
    p = ARTIFACTS / MODEL_FILES[name]
    if not p.is_file():
        raise FileNotFoundError(f"File model belum ditemukan: {p.name}")
    stt = p.stat()
    return load_model_cached(str(p), stt.st_mtime_ns, stt.st_size)

def validate_signal(x):
    x = np.asarray(x, dtype=np.float32).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Waveform harus 1D, diterima shape {x.shape}.")
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Waveform kosong atau mengandung NaN/Inf.")
    return x

def resample_to_100(x, source_fs):
    x = validate_signal(x)
    if source_fs <= 0:
        raise ValueError("Sampling rate harus lebih besar dari nol.")
    if abs(float(source_fs)-FS) < 1e-9:
        return x.astype(np.float32)
    ratio = Fraction(FS/float(source_fs)).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)

def get_hdf5_datasets(raw):
    result = {}
    with h5py.File(io.BytesIO(raw), "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
                result[name] = tuple(obj.shape)
        h5.visititems(visit)
    return result

def read_signal(raw, filename, source_fs=100.0, channel=0, event=0, trace=0, dataset=None):
    ext = Path(filename).suffix.lower().lstrip(".")
    meta = {"file_name":filename, "extension":ext}
    if ext in {"csv","txt"}:
        # Infer common comma/tab/semicolon/space-delimited numeric tables.
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", header=None, comment="#")
            arr = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="all").to_numpy(dtype=float)
        except Exception:
            arr = np.genfromtxt(io.BytesIO(raw), delimiter=None, comments="#")
        arr = np.asarray(arr)
        if arr.ndim == 1: arr = arr[:,None]
        if arr.ndim != 2 or channel < 0 or channel >= arr.shape[1]:
            raise ValueError("Tabel tidak memiliki channel numerik yang dipilih.")
        x = arr[:,channel]
        sr = float(source_fs)
        meta["channel_index"] = channel
    elif ext == "wav":
        sr, arr = wavfile.read(io.BytesIO(raw))
        if arr.ndim == 1:
            x = arr
        else:
            if channel < 0 or channel >= arr.shape[1]: raise ValueError("Indeks channel WAV tidak valid.")
            x = arr[:,channel]
        sr = float(sr)
        meta["channel_index"] = channel
    elif ext in {"mseed","msd","miniseed"}:
        from obspy import read
        stream = read(io.BytesIO(raw))
        if len(stream) == 0 or trace < 0 or trace >= len(stream):
            raise ValueError("Trace MiniSEED tidak tersedia.")
        tr = stream[trace]
        x = tr.data
        sr = float(tr.stats.sampling_rate)
        meta.update({"trace_index":trace,"station":str(getattr(tr.stats,"station","")),"channel":str(getattr(tr.stats,"channel",""))})
    elif ext in {"h5","hdf5"}:
        if not dataset: raise ValueError("Pilih dataset HDF5 terlebih dahulu.")
        with h5py.File(io.BytesIO(raw), "r") as h5:
            if dataset not in h5: raise ValueError(f"Dataset {dataset} tidak ditemukan.")
            ds = h5[dataset]
            arr = np.asarray(ds[...])
            sr = None
            for k in ("sampling_rate_hz","sampling_rate","sample_rate","fs"):
                if k in ds.attrs: sr = float(ds.attrs[k]); break
                if k in h5.attrs: sr = float(h5.attrs[k]); break
        if sr is None: sr = float(source_fs)
        if arr.ndim == 1: x = arr
        elif arr.ndim == 2:
            if arr.shape[1] == N_SAMPLES and arr.shape[0] != N_SAMPLES:
                if event >= arr.shape[0]: raise ValueError("Event index HDF5 di luar rentang.")
                x = arr[event,:]
            else:
                if channel >= arr.shape[1]: raise ValueError("Channel HDF5 di luar rentang.")
                x = arr[:,channel]
        elif arr.ndim == 3:
            if event >= arr.shape[0] or channel >= arr.shape[2]: raise ValueError("Event/channel HDF5 di luar rentang.")
            x = arr[event,:,channel]
        else: raise ValueError(f"Dimensi HDF5 tidak didukung: {arr.shape}")
        meta.update({"dataset":dataset,"event_index":event,"channel_index":channel})
    else:
        raise ValueError(f"Format .{ext} belum didukung.")
    x = validate_signal(x)
    meta["source_sampling_rate_hz"] = float(sr)
    meta["source_samples"] = int(len(x))
    return x, float(sr), meta

def make_segment(x, source_fs, start_seconds=0.0):
    original = validate_signal(x)
    duration = len(original)/float(source_fs)
    resampled = resample_to_100(original, source_fs)
    max_start = max(0.0, len(resampled)/FS-60.0)
    if start_seconds < 0 or start_seconds > max_start+1e-6:
        raise ValueError(f"Awal segmen harus 0 hingga {max_start:.2f} detik.")
    if len(resampled) < N_SAMPLES:
        raise ValueError(f"Durasi hanya {duration:.2f} detik. Model membutuhkan minimal 60 detik.")
    i0 = int(round(start_seconds*FS))
    seg = resampled[i0:i0+N_SAMPLES]
    if len(seg) != N_SAMPLES: raise ValueError("Segmen tidak tepat 6.000 sampel.")
    meta = {
        "original_duration_seconds":duration,
        "processed_duration_seconds":60.0,
        "original_sampling_rate_hz":float(source_fs),
        "processed_sampling_rate_hz":FS,
        "sample_count":int(len(seg)),
        "segment_start_seconds":float(start_seconds),
        "amplitude_min":float(np.min(seg)),
        "amplitude_max":float(np.max(seg)),
    }
    return seg.astype(np.float32), meta

def preprocess_waveform(x):
    x = validate_signal(x)
    if len(x) != N_SAMPLES: raise ValueError("SeismicNet memerlukan 6.000 sampel.")
    return x.reshape(1,6000,1).astype(np.float32)

def preprocess_spectrogram(x):
    x = validate_signal(x)
    if len(x) != N_SAMPLES: raise ValueError("Spectrogram memerlukan 6.000 sampel.")
    centered = x - np.mean(x)
    tapered = centered * tukey(len(centered), alpha=0.1)
    f,t,z = stft(tapered, fs=FS, window="hann", nperseg=1000, noverlap=900, boundary="zeros", padded=True)
    db = 20.0*np.log10(np.abs(z)+1e-10)
    mask = (f>=0.5)&(f<=10.0)
    cropped = db[mask,:60]
    if cropped.shape != (96,60): raise ValueError(f"Shape spectrogram {cropped.shape}, diharapkan (96,60).")
    lo,hi=float(cropped.min()),float(cropped.max())
    normalized=(cropped-lo)/(hi-lo+1e-10)
    return normalized.astype(np.float32)[None,:,:,None]

def predict_model(name, x):
    model = load_model(name)
    wave = preprocess_waveform(x)
    spec = preprocess_spectrogram(x)
    if name == "SeismicNet": inp = wave
    elif name == "VOISS-Net": inp = spec
    else: inp = [wave,spec]
    scores = np.asarray(model.predict(inp, verbose=0)[0], dtype=float)
    del model
    gc.collect()
    if scores.shape != (4,) or not np.all(np.isfinite(scores)):
        raise ValueError(f"Output {name} tidak sesuai: {scores.shape}")
    idx = int(np.argmax(scores))
    return {
        "model":name,
        "predicted_code":CODES[idx],
        "predicted_class":CLASS_NAMES[CODES[idx]],
        "confidence":float(scores[idx]),
        "probabilities":{CODES[i]:float(scores[i]) for i in range(4)}
    }

def agreement_for(results):
    if len(results)<2: return {"available":False,"message":"Model Agreement hanya tersedia saat minimal dua model dijalankan."}
    counts={}
    for r in results: counts[r["predicted_code"]]=counts.get(r["predicted_code"],0)+1
    top,count=max(counts.items(), key=lambda x:x[1])
    majority=count>=len(results)//2+1
    return {"available":True,"agreement_count":count,"model_count":len(results),
        "consensus_code":top if majority else None,
        "message":f"Mayoritas {count}/{len(results)} model memprediksi {top}." if majority else "Tidak ada mayoritas: prediksi model berbeda."}

def ensure_history():
    if "voiss_history" not in st.session_state: st.session_state["voiss_history"]=[]
    return st.session_state["voiss_history"]

def save_history(row):
    h=ensure_history()
    h.insert(0,row)
    del h[200:]

def plot_signals(x):
    t=np.arange(len(x))/FS
    a,b=st.columns(2)
    with a:
        fig=go.Figure(go.Scatter(x=t,y=x,mode="lines",line=dict(color="#2476cc",width=1)))
        fig.update_layout(title="Waveform (Raw Seismic Signal)",xaxis_title="Waktu (detik)",yaxis_title="Amplitudo",height=330,template="plotly_white",margin=dict(l=20,r=15,t=50,b=35))
        st.plotly_chart(fig,use_container_width=True)
    with b:
        sp=preprocess_spectrogram(x)[0,:,:,0]
        fig=go.Figure(go.Heatmap(z=sp,x=np.linspace(0,60,60),y=np.linspace(0.5,10,96),colorscale="Turbo",colorbar=dict(title="Skala 0–1")))
        fig.update_layout(title="Spectrogram (Input VOISS-Net)",xaxis_title="Waktu (detik)",yaxis_title="Frekuensi (Hz)",height=330,template="plotly_white",margin=dict(l=20,r=15,t=50,b=35))
        st.plotly_chart(fig,use_container_width=True)

def render_result_cards(results):
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        code = result["predicted_code"]
        color = COLORS[code]
        with col:
            st.markdown(
                f'<div class="result-card"><div style="font-size:14px;font-weight:800;color:#1a3d64">{html.escape(result["model"])}</div>'
                f'<div class="result-class">{code} · {html.escape(result["predicted_class"])}</div></div>',
                unsafe_allow_html=True
            )
            st.metric("Confidence", f'{result["confidence"]*100:.2f}%')
            st.markdown("**Class Probability**")
            for label in CODES:
                score = result["probabilities"][label]
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-top:5px">'
                    f'<span>{label} · {CLASS_NAMES[label]}</span><b>{score*100:.2f}%</b></div>',
                    unsafe_allow_html=True
                )
                st.progress(float(np.clip(score, 0, 1)), text=None)


def dashboard():
    page_header(
        "Volcanic Earthquake Classification",
        "SeismicNet (CNN 1D)  |  VOISS-Net (CNN 2D)  |  Seismic-VOISS-Net Multimodal CNN"
    )
    banner = asset_data_uri("volcano-banner.svg")
    bg = f'background-image:linear-gradient(90deg,rgba(250,252,255,.98) 0%,rgba(246,250,255,.91) 47%,rgba(236,245,254,.10) 100%),url("{banner}");' if banner else 'background:linear-gradient(110deg,#fff,#d9eafb);'
    st.markdown(
        f'<div class="hero-card" style="{bg}"><div class="hero-copy">'
        f'<div class="hero-kicker">Research-based intelligent seismic system</div>'
        f'<div class="hero-title">VOISS</div>'
        f'<div class="hero-subtitle">Volcanic Earthquake Classification System</div>'
        f'<div class="hero-text">Sistem klasifikasi gempa vulkanik berbasis deep learning yang menggabungkan analisis waveform, spektrogram, dan pendekatan multimodal untuk mengidentifikasi empat jenis kejadian seismik.</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    features = [
        ("〽️","Klasifikasi Otomatis","Mengidentifikasi empat kelas: LP, TC, TR, dan VT."),
        ("▱","Tiga Model CNN","SeismicNet, VOISS-Net, dan Seismic-VOISS-Net Multimodal CNN."),
        ("▥","Visualisasi Sinyal","Menampilkan waveform dan spektrogram hasil preprocessing."),
        ("⌘","Berbasis Penelitian","Menggunakan model dan konfigurasi dari eksperimen penelitian."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-heading">Empat Kelas Klasifikasi</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, code in zip(cols, CODES):
        col.markdown(
            f'<div class="class-chip" style="--chip-color:{COLORS[code]}">'
            f'<div class="class-chip-code">{code}</div><div class="class-chip-name">{CLASS_NAMES[code]}</div></div>',
            unsafe_allow_html=True
        )

    a, b, c = st.columns(3)
    a.metric("Arsitektur CNN", "3")
    b.metric("Kelas keluaran", "4")
    c.metric("Input standar", "100 Hz · 60 detik")

    missing = model_status()
    if missing:
        st.warning("Ada artefak model yang belum siap.")
        st.code("\n".join(missing))
    else:
        st.markdown(
            '<div class="agreement-card"><b style="color:#157347">✓ Sistem siap untuk inference</b>'
            '<div style="color:#476b59;font-size:12px;margin-top:4px">Ketiga model ditemukan pada direktori artefak.</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-heading">Mulai Analisis</div>', unsafe_allow_html=True)
    left, right = st.columns([2,1])
    with left:
        st.markdown(
            '<div class="voiss-panel"><b>Alur penggunaan</b><p style="color:#5d7289;font-size:13px">'
            'Unggah satu rekaman seismik, pilih model yang ingin digunakan, kemudian jalankan klasifikasi. '
            'Hasil prediksi akan dilengkapi visualisasi sinyal dan distribusi skor untuk setiap kelas.</p></div>',
            unsafe_allow_html=True
        )
    with right:
        if st.button("Mulai Klasifikasi  →", type="primary", use_container_width=True):
            st.session_state["active_page"] = "Classification"
            st.rerun()


def classification():
    page_header(
        "Volcanic Earthquake Classification",
        "Input seismic data  |  Signal visualization  |  Classification result"
    )
    missing = model_status()
    if missing:
        st.error("Artefak model belum lengkap. Periksa folder artifacts.")
        st.code("\n".join(missing))
        return

    st.markdown('<div class="section-heading">1. Input Seismic Data</div>', unsafe_allow_html=True)
    up = st.file_uploader(
        "Upload Seismic File",
        type=ACCEPTED,
        help="Format: CSV, TXT, WAV, MiniSEED, H5, dan HDF5. Model menggunakan satu segmen berdurasi 60 detik."
    )
    if up is None:
        st.info("Drag and drop file seismik di sini atau klik Browse files. Input model: 100 Hz, 6.000 sampel, 60 detik.")
        return

    raw = up.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    ext = Path(up.name).suffix.lower().lstrip(".")
    dataset = None
    event = 0
    channel = 0
    trace = 0
    manual_fs = 100.0

    if ext in {"csv", "txt", "h5", "hdf5"}:
        manual_fs = st.number_input("Sampling rate (Hz)", min_value=0.1, max_value=10000.0, value=100.0, step=1.0)

    if ext in {"h5", "hdf5"}:
        try:
            names = get_hdf5_datasets(raw)
            if not names:
                st.error("Tidak ditemukan dataset numerik pada HDF5.")
                return
            dataset = st.selectbox("Pilih dataset HDF5", list(names))
            shape = names[dataset]
            if len(shape) == 3:
                event = st.number_input("Indeks event", 0, max(0, shape[0]-1), 0)
                channel = st.number_input("Indeks channel", 0, max(0, shape[2]-1), 0)
            elif len(shape) == 2 and shape[1] == N_SAMPLES and shape[0] != N_SAMPLES:
                event = st.number_input("Indeks event", 0, max(0, shape[0]-1), 0)
            elif len(shape) == 2:
                channel = st.number_input("Indeks channel", 0, max(0, shape[1]-1), 0)
        except Exception as exc:
            st.error(f"Gagal membaca struktur HDF5: {exc}")
            return

    if ext == "wav":
        try:
            sr0, array = wavfile.read(io.BytesIO(raw))
            if array.ndim == 2:
                channel = st.selectbox("Pilih channel WAV", list(range(array.shape[1])))
            st.caption(f"Sampling rate dari WAV: {sr0} Hz")
        except Exception as exc:
            st.error(f"WAV tidak valid: {exc}")
            return

    if ext in {"mseed", "msd"}:
        try:
            from obspy import read
            stream = read(io.BytesIO(raw))
            trace = st.selectbox("Pilih trace MiniSEED", list(range(len(stream))))
        except Exception as exc:
            st.error(f"MiniSEED tidak valid: {exc}")
            return

    try:
        preview, preview_sr, file_meta = read_signal(raw, up.name, manual_fs, int(channel), int(event), int(trace), dataset)
        duration = len(preview) / preview_sr
        st.caption(f"File terbaca: {len(preview):,} sampel · {preview_sr:g} Hz · durasi {duration:.2f} detik")
        start = 0.0
        if duration > 60:
            start = st.slider("Pilih awal segmen 60 detik", 0.0, float(duration-60), 0.0, 1.0)
        elif duration < 60:
            st.warning("Durasi rekaman kurang dari 60 detik. Model membutuhkan minimal 60 detik.")
    except Exception as exc:
        st.error(f"File tidak dapat dibaca: {exc}")
        return

    st.markdown('<div class="section-heading">2. Model Selection</div>', unsafe_allow_html=True)
    choice = st.radio(
        "Pilih model",
        ["SeismicNet (CNN 1D)", "VOISS-Net (CNN 2D)", "Seismic-VOISS-Net Multimodal CNN", "Compare All Models"],
        horizontal=True,
        label_visibility="collapsed"
    )
    selected = {
        "SeismicNet (CNN 1D)": ["SeismicNet"],
        "VOISS-Net (CNN 2D)": ["VOISS-Net"],
        "Seismic-VOISS-Net Multimodal CNN": ["Seismic-VOISS-Net Multimodal CNN"],
        "Compare All Models": list(MODEL_FILES),
    }[choice]

    if st.button("▶  Run Classification", type="primary", use_container_width=True):
        try:
            x, segment_meta = make_segment(preview, preview_sr, float(start))
            with st.spinner("Model sedang melakukan klasifikasi..."):
                results = [predict_model(name, x) for name in selected]
            agreement = agreement_for(results)
            run = now_wib()
            row = {
                "run_at": run, "file_name": up.name, "mode": choice,
                "results": results, "agreement": agreement,
                "duration": 60.0, "sample_rate": 100, "samples": 6000,
                "source_duration": duration, "source_fs": preview_sr,
                "segment_start": float(start), "file_meta": file_meta,
            }
            save_history(row)
            st.session_state["last_prediction"] = {"hash": digest, "waveform": x, "row": row}
            st.success("Klasifikasi selesai.")
        except Exception as exc:
            st.error(f"Klasifikasi gagal: {exc}")
            st.exception(exc)

    last = st.session_state.get("last_prediction")
    if last and last["hash"] == digest:
        row = last["row"]
        st.markdown('<div class="section-heading">3. Signal Visualization</div>', unsafe_allow_html=True)
        plot_signals(last["waveform"])

        st.markdown('<div class="section-heading">4. Classification Result</div>', unsafe_allow_html=True)
        render_result_cards(row["results"])

        st.markdown('<div class="section-heading">5. Model Agreement</div>', unsafe_allow_html=True)
        if row["agreement"].get("available"):
            ag = row["agreement"]
            if ag.get("consensus_code"):
                st.markdown(
                    f'<div class="agreement-card"><b style="color:#167348">✓ Model Agreement</b>'
                    f'<div style="color:#315b47;margin-top:4px">{html.escape(ag["message"])} '
                    f'Final prediction: <b>{ag["consensus_code"]} · {CLASS_NAMES[ag["consensus_code"]]}</b></div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(ag["message"])
            st.metric("Model Agreement", f'{ag["agreement_count"]}/{ag["model_count"]} models')
        else:
            st.info("Model Agreement tersedia pada mode Compare All Models.")

        st.markdown('<div class="section-heading">6. Event Information</div>', unsafe_allow_html=True)
        meta_left, meta_right = st.columns(2)
        event_rows = [
            ("File Name", row["file_name"]),
            ("Original Duration", f'{row.get("source_duration",60):.2f} seconds'),
            ("Input Sampling Rate", f'{row.get("source_fs",100):g} Hz'),
            ("Processed Duration", "60.00 seconds"),
            ("Processed Sampling Rate", "100 Hz"),
            ("Number of Samples", "6,000"),
            ("Segment Start", f'{row.get("segment_start",0):.2f} seconds'),
            ("Analysis Time", row["run_at"]),
        ]
        with meta_left:
            st.dataframe(pd.DataFrame(event_rows[:4], columns=["Information","Value"]), use_container_width=True, hide_index=True)
        with meta_right:
            st.dataframe(pd.DataFrame(event_rows[4:], columns=["Information","Value"]), use_container_width=True, hide_index=True)
        st.caption("Confidence ditampilkan sebagai skor Softmax tertinggi. Skor ini belum tentu merupakan probabilitas yang terkalibrasi.")


def history_page():
    page_header("Prediction History", "Riwayat klasifikasi yang dilakukan pada sesi browser ini.")
    hist = ensure_history()
    if not hist:
        st.markdown(
            '<div class="voiss-panel"><h3 style="color:#173d66">Belum ada riwayat prediksi</h3>'
            '<p style="color:#64788e;font-size:13px">Hasil klasifikasi akan muncul di sini setelah Anda menjalankan analisis.</p></div>',
            unsafe_allow_html=True
        )
        return

    top1, top2, top3 = st.columns([2,1,1])
    with top1:
        query = st.text_input("Cari nama file", placeholder="Cari berdasarkan nama file...")
    with top2:
        model_filter = st.selectbox("Model", ["Semua Model"] + sorted({r["mode"] for r in hist}))
    with top3:
        if st.button("Clear History", use_container_width=True):
            st.session_state["voiss_history"] = []
            st.rerun()

    rows = []
    for i, record in enumerate(hist, start=1):
        predicted = ", ".join(item["predicted_code"] for item in record["results"])
        rows.append({
            "No": i,
            "Date & Time": record["run_at"],
            "File Name": record["file_name"],
            "Model": record["mode"],
            "Prediction": predicted,
            "Confidence": max(item["confidence"] for item in record["results"]),
            "Agreement": record["agreement"].get("message", "—"),
        })
    df = pd.DataFrame(rows)
    if query:
        df = df[df["File Name"].str.contains(query, case=False, na=False)]
    if model_filter != "Semua Model":
        df = df[df["Model"] == model_filter]

    display = df.copy()
    if not display.empty:
        display["Confidence"] = display["Confidence"].map(lambda value: f"{value*100:.2f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)
    if not df.empty:
        st.download_button("↓ Export CSV", df.to_csv(index=False).encode("utf-8-sig"), "voiss_prediction_history.csv", "text/csv")
    st.markdown("### Detail Prediksi")
    for record in hist:
        with st.expander(f'{record["run_at"]} · {record["file_name"]} · {record["mode"]}'):
            render_result_cards(record["results"])
            if record["agreement"].get("available"):
                st.write(record["agreement"]["message"])
    st.caption("Riwayat tersimpan sementara pada sesi browser ini.")


def get_profile_photo():
    # Keep the photo out of public GitHub: read it from Streamlit Cloud Secrets.
    try:
        encoded = st.secrets.get("profile_photo_base64", "")
        if encoded:
            raw = base64.b64decode(str(encoded).strip())
            return io.BytesIO(raw)
    except Exception:
        pass
    local_photo = ASSETS / "profile.jpg"
    return str(local_photo) if local_photo.exists() else None


def about_page():
    page_header("About the Researcher", "Research background · Machine learning · Volcanic seismic classification")
    photo = get_profile_photo()

    st.markdown('<div class="section-heading">Meet the Researcher</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 2.25], gap="large")
    with left:
        if photo is not None:
            st.image(photo, use_container_width=True)
        else:
            st.markdown(
                '<div class="profile-placeholder"><div style="text-align:center"><div style="font-size:60px;font-weight:850">FK</div>'
                '<div style="font-size:12px;font-weight:700">Researcher Portrait</div></div></div>',
                unsafe_allow_html=True
            )
    with right:
        st.markdown(
            '<div class="profile-hero"><div class="hero-kicker">Researcher & Developer</div>'
            '<div class="profile-name">Farrelius Kevin</div>'
            '<div class="profile-role">S1 Informatika · Universitas Multimedia Nusantara · Data Science & AI</div>'
            '<div class="profile-bio">Saya memiliki ketertarikan pada pemanfaatan data science dan artificial intelligence untuk mengolah data kompleks menjadi informasi yang lebih mudah dipahami. Melalui VOISS, saya mengembangkan prototipe klasifikasi gempa vulkanik dengan mengeksplorasi pendekatan CNN 1D, CNN 2D, dan multimodal learning.</div>'
            '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:13px">'
            '<span style="background:#e4f0fc;color:#1b5d99;border-radius:20px;padding:6px 10px;font-size:11px;font-weight:700">Python</span>'
            '<span style="background:#e4f0fc;color:#1b5d99;border-radius:20px;padding:6px 10px;font-size:11px;font-weight:700">TensorFlow</span>'
            '<span style="background:#e4f0fc;color:#1b5d99;border-radius:20px;padding:6px 10px;font-size:11px;font-weight:700">Deep Learning</span>'
            '<span style="background:#e4f0fc;color:#1b5d99;border-radius:20px;padding:6px 10px;font-size:11px;font-weight:700">Streamlit</span>'
            '</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-heading">About the VOISS Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="voiss-panel"><h3 style="color:#173c63;margin:0 0 8px">From Seismic Signals to Intelligent Classification</h3>'
        '<p style="color:#4a627d;font-size:13px;line-height:1.65">VOISS (Volcanic Earthquake Classification System) merupakan prototipe aplikasi berbasis web yang membawa eksperimen penelitian ke dalam pengalaman interaktif. Sistem ini memproses rekaman seismik melalui waveform dan representasi spektrogram, kemudian membandingkan tiga pendekatan CNN untuk mengenali karakteristik kejadian gempa vulkanik.</p>'
        '<p style="color:#4a627d;font-size:13px;line-height:1.65">Pengembangan VOISS bertujuan menjembatani proses eksperimen machine learning dengan demonstrasi yang dapat dicoba oleh pengguna, sehingga alur input data, transformasi sinyal, dan hasil klasifikasi dapat diamati dalam satu aplikasi.</p></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading">Three Model Approaches</div>', unsafe_allow_html=True)
    model_cards = [
        ("01", "SeismicNet", "CNN 1D · Waveform", "Mempelajari pola temporal langsung dari urutan amplitudo sinyal seismik.", "#dceeff", "#2376bd"),
        ("02", "VOISS-Net", "CNN 2D · Spectrogram", "Menganalisis pola waktu-frekuensi dari representasi spektrogram hasil STFT.", "#eee8ff", "#7956b6"),
        ("03", "Seismic-VOISS-Net", "Multimodal CNN", "Menggabungkan fitur waveform dan spektrogram melalui feature fusion.", "#e2f6e9", "#26845a"),
    ]
    cols = st.columns(3)
    for col, (num, name, tag, desc, bg, accent) in zip(cols, model_cards):
        with col:
            st.markdown(
                f'<div class="about-highlight" style="border-top:4px solid {accent}">'
                f'<div style="font-size:11px;color:{accent};font-weight:800">{num} · {tag}</div>'
                f'<h4>{name}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-heading">Research Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "Llaima, Chile")
    c2.metric("Event Classes", "4")
    c3.metric("Sampling Rate", "100 Hz")
    c4.metric("Input Window", "60 seconds")

    st.markdown(
        '<div class="voiss-panel"><b style="color:#173c63">Classification Labels</b>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">'
        + "".join([f'<span style="background:{COLORS[k]}18;border:1px solid {COLORS[k]}55;color:{COLORS[k]};border-radius:8px;padding:8px 13px;font-size:12px;font-weight:800">{k} · {CLASS_NAMES[k]}</span>' for k in CODES])
        + '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading">Research & Technology</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="about-highlight"><h4>Research Focus</h4><p>Comparative CNN architectures, waveform analysis, time-frequency representation, multimodal feature fusion, dan explainable AI untuk interpretasi model.</p></div>',
            unsafe_allow_html=True
        )
    with right:
        st.markdown(
            '<div class="about-highlight"><h4>Technology Stack</h4><p>Python · TensorFlow / Keras · NumPy · SciPy · Pandas · Plotly · Streamlit · h5py · ObsPy</p></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-heading">Explore the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="voiss-panel"><p style="color:#4a627d;font-size:13px">Jelajahi halaman Documentation untuk memahami arsitektur ketiga model, alur preprocessing, serta cara menggunakan aplikasi VOISS. Source code aplikasi tersedia pada repository publik berikut.</p>'
        '<a href="https://github.com/Jfkevin/voiss-volcanic-earthquake-classifier" target="_blank" style="display:inline-block;background:#1768b5;color:white!important;text-decoration:none;padding:9px 15px;border-radius:8px;font-weight:700;font-size:12px">⌘ View GitHub Repository ↗</a></div>',
        unsafe_allow_html=True
    )
    st.caption("VOISS is an academic research and portfolio prototype.")


def flow_step(title, detail="", tone="blue"):
    return (
        f'<div class="flow-step {tone}"><b>{html.escape(title)}</b>'
        f'<small>{html.escape(detail)}</small></div>'
    )


def pipeline_html(steps, tone="blue"):
    pieces = []
    for index, (title, detail) in enumerate(steps):
        pieces.append(flow_step(title, detail, tone))
        if index < len(steps) - 1:
            pieces.append('<div class="flow-arrow">→</div>')
    return '<div class="flow-scroll"><div class="flow-row">' + "".join(pieces) + '</div></div>'


def documentation_page():
    page_header("Documentation", "Panduan penggunaan, data & preprocessing, serta dokumentasi arsitektur model VOISS.")
    toc, content = st.columns([0.25, 0.75], gap="medium")
    with toc:
        st.markdown(
            '<div class="doc-toc"><div style="font-size:17px;font-weight:800;color:#153b64;margin:3px 5px 10px">▤ Daftar Isi</div>'
            '<a href="#about-voiss">1. Tentang VOISS</a>'
            '<a href="#architecture">2. Arsitektur Model</a>'
            '<a href="#seismicnet" style="padding-left:22px">• 2.1 SeismicNet (CNN 1D)</a>'
            '<a href="#voissnet" style="padding-left:22px">• 2.2 VOISS-Net (CNN 2D)</a>'
            '<a href="#multimodal" style="padding-left:22px">• 2.3 Seismic-VOISS-Net Multimodal CNN</a>'
            '<a href="#labels">3. Label Klasifikasi</a>'
            '<a href="#preprocessing">4. Data & Preprocessing</a>'
            '<a href="#usage">5. Cara Penggunaan</a></div>',
            unsafe_allow_html=True
        )
    with content:
        st.markdown(
            '<div class="doc-section" id="about-voiss"><h3>1. Tentang VOISS</h3>'
            '<p style="font-size:13px;color:#4d647e;line-height:1.65">VOISS (Volcanic Earthquake Classification System) adalah aplikasi demonstrasi penelitian untuk klasifikasi gempa vulkanik menggunakan tiga arsitektur Convolutional Neural Network (CNN). Sistem mengolah sinyal seismik dalam bentuk waveform dan spektrogram, kemudian menghasilkan prediksi ke dalam empat kategori: Long Period (LP), Tectonic (TC), Tremor (TR), dan Volcano-Tectonic (VT).</p></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="doc-section" id="architecture"><h3>2. Arsitektur Model</h3>'
            '<p style="font-size:13px;color:#4d647e;line-height:1.6">VOISS mengimplementasikan dua model unimodal dan satu model multimodal. SeismicNet menerima waveform, VOISS-Net menerima spektrogram, sedangkan Seismic-VOISS-Net Multimodal CNN menggabungkan representasi dari kedua cabang.</p></div>',
            unsafe_allow_html=True
        )

        steps1 = [
            ("Input Waveform", "6000 × 1"),
            ("Conv1D · 32", "kernel 64 · stride 2"),
            ("MaxPool1D", "pool 4 · dropout .15"),
            ("Conv1D · 64", "kernel 32 · stride 2"),
            ("MaxPool1D", "pool 4 · dropout .35"),
            ("Conv1D · 128", "kernel 16 · stride 2"),
            ("MaxPool1D", "pool 4 · stride 4"),
            ("Conv1D · 256", "kernel 8 · stride 2"),
            ("Conv1D · 1401", "kernel 16 · stride 2"),
            ("Flatten", "vektor fitur"),
            ("Dense · 1500", "ReLU"),
            ("Dropout", "rate 0.75"),
            ("Dense · 4", "Softmax"),
        ]
        st.markdown(
            '<div class="architecture-card" id="seismicnet"><div class="architecture-band blue">〽️ 2.1 SeismicNet (CNN 1D)</div>'
            '<div class="architecture-body"><div class="architecture-description">SeismicNet merupakan model CNN 1D yang memproses sinyal seismik dalam bentuk waveform (deret waktu). Arsitektur ini dirancang untuk mengekstraksi pola temporal dari perubahan amplitudo sinyal.</div>'
            + pipeline_html(steps1, "blue")
            + '<div style="font-size:12px;color:#45617d"><b>Input:</b> (6000, 1) · <b>Output:</b> 4 kelas Softmax · <b>Total parameters:</b> 12.513.181</div></div></div>',
            unsafe_allow_html=True
        )

        steps2 = [
            ("Input Spectrogram", "96 × 60 × 1"),
            ("Conv2D · 32", "kernel 3 × 3"),
            ("MaxPool2D", "pool 3 × 3"),
            ("Conv2D · 64", "kernel 3 × 3"),
            ("MaxPool2D", "pool 3 × 3"),
            ("Conv2D · 128", "kernel 3 × 3"),
            ("MaxPool2D", "pool 3 × 3"),
            ("Flatten", "vektor fitur"),
            ("Dropout", "rate 0.2"),
            ("Dense · 128", "ReLU"),
            ("Dropout", "rate 0.5"),
            ("Dense · 64", "ReLU"),
            ("Dropout", "rate 0.5"),
            ("Dense · 4", "Softmax"),
        ]
        st.markdown(
            '<div class="architecture-card" id="voissnet"><div class="architecture-band purple">▧ 2.2 VOISS-Net (CNN 2D)</div>'
            '<div class="architecture-body"><div class="architecture-description">VOISS-Net merupakan model CNN 2D yang memproses spektrogram hasil Short-Time Fourier Transform (STFT). Model mengekstraksi pola spektral dan temporal secara simultan melalui operasi konvolusi dua dimensi.</div>'
            + pipeline_html(steps2, "purple")
            + '<div style="font-size:12px;color:#5d4b82"><b>Input:</b> (96, 60, 1) · <b>Output:</b> 4 kelas Softmax · <b>Total parameters:</b> 199.620</div></div></div>',
            unsafe_allow_html=True
        )

        wave_steps = [
            ("Input Waveform", "6000 × 1"),
            ("Conv1D stack", "32 · 64 · 128 · 256 · 1401"),
            ("Pooling", "MaxPool1D blocks"),
            ("Flatten", "vektor waveform"),
            ("Dense · 1500", "ReLU · Dropout .75"),
        ]
        spec_steps = [
            ("Input Spectrogram", "96 × 60 × 1"),
            ("Conv2D stack", "32 · 64 · 128"),
            ("Pooling", "MaxPool2D blocks"),
            ("Flatten", "vektor spektrogram"),
            ("Dense layers", "128 → 64"),
        ]
        fusion_steps = [
            ("Concatenate", "gabungkan fitur"),
            ("Dense · 256", "ReLU"),
            ("Dropout", "rate 0.5"),
            ("Dense · 4", "Softmax"),
        ]
        st.markdown(
            '<div class="architecture-card" id="multimodal"><div class="architecture-band green">⑂ 2.3 Seismic-VOISS-Net Multimodal CNN</div>'
            '<div class="architecture-body"><div class="architecture-description">Model multimodal ini memiliki dua cabang pemrosesan. Cabang pertama menggunakan SeismicNet untuk waveform, sedangkan cabang kedua menggunakan VOISS-Net untuk spektrogram. Vektor fitur dari kedua cabang digabungkan melalui feature fusion sebelum lapisan klasifikasi akhir.</div>'
            '<div style="background:#eff7ff;border:1px dashed #9dc7ed;border-radius:10px;padding:10px 12px;margin-top:9px"><b style="font-size:12px;color:#1b5b95">Cabang 1 — SeismicNet (CNN 1D)</b>'
            + pipeline_html(wave_steps, "blue")
            + '</div><div style="background:#fff3f5;border:1px dashed #e8aeb8;border-radius:10px;padding:10px 12px;margin-top:9px"><b style="font-size:12px;color:#9b4654">Cabang 2 — VOISS-Net (CNN 2D)</b>'
            + pipeline_html(spec_steps, "pink")
            + '</div><div style="background:#effaf3;border:1px dashed #9ed8b5;border-radius:10px;padding:10px 12px;margin-top:9px"><b style="font-size:12px;color:#28764b">Feature Fusion & Classification Head</b>'
            + pipeline_html(fusion_steps, "green")
            + '</div><div style="font-size:12px;color:#356b4d;margin-top:8px"><b>Input:</b> waveform (6000, 1) + spectrogram (96, 60, 1) · <b>Total parameters:</b> 13.108.205</div></div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="doc-section"><h3>Komponen & Konfigurasi Utama</h3>', unsafe_allow_html=True)
        config = pd.DataFrame([
            ["Jenis input", "Waveform (1D)", "Spectrogram (2D)", "Waveform + Spectrogram"],
            ["Input tanpa batch", "(6000, 1)", "(96, 60, 1)", "[(6000, 1), (96, 60, 1)]"],
            ["Jumlah parameter", "12.513.181", "199.620", "13.108.205"],
            ["Optimizer", "Adam", "Adam", "Adam"],
            ["Learning rate", "1 × 10⁻⁴", "1 × 10⁻⁴", "1 × 10⁻⁴"],
            ["Loss function", "Categorical Crossentropy", "Categorical Crossentropy", "Categorical Crossentropy"],
            ["Batch size", "32", "32", "32"],
        ], columns=["Komponen", "SeismicNet", "VOISS-Net", "Seismic-VOISS-Net"])
        st.dataframe(config, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="doc-section" id="labels"><h3>3. Label Klasifikasi</h3>'
            '<p style="color:#536a83;font-size:13px">Ketiga model menghasilkan skor untuk empat kelas kejadian seismik vulkanik berikut.</p>',
            unsafe_allow_html=True
        )
        cols = st.columns(4)
        for col, code in zip(cols, CODES):
            with col:
                st.markdown(
                    f'<div class="class-chip" style="--chip-color:{COLORS[code]}"><div class="class-chip-code">{code}</div><div class="class-chip-name">{CLASS_NAMES[code]}</div></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="doc-section" id="preprocessing"><h3>4. Data & Preprocessing</h3>'
            '<p style="font-size:13px;color:#4d647e;line-height:1.6">Satu input inferensi dipersiapkan menjadi waveform sepanjang 60 detik pada sampling rate 100 Hz. Dari waveform yang sama, sistem membentuk spektrogram menggunakan urutan preprocessing berikut:</p>'
            '<ol style="color:#425b75;font-size:13px;line-height:1.8">'
            '<li>Validasi waveform satu dimensi dengan 6.000 sampel.</li>'
            '<li>Mean removal untuk mengurangi komponen rata-rata sinyal.</li>'
            '<li>Tukey window dengan alpha 0,1.</li>'
            '<li>STFT menggunakan Hann window, nperseg 1000, dan noverlap 900.</li>'
            '<li>Konversi magnitudo menjadi desibel: 20 log10(|Zxx| + 1e-10).</li>'
            '<li>Pemotongan frekuensi 0,5–10 Hz dan pengambilan 60 kolom waktu pertama.</li>'
            '<li>Normalisasi Min-Max per spektrogram ke rentang [0, 1].</li></ol>'
            '<div style="background:#f2f7fd;border-radius:9px;padding:10px 12px;color:#244c73;font-size:12px"><b>Format input model:</b> Waveform (6000, 1) · Spectrogram (96, 60, 1) · Multimodal menggunakan keduanya dari event yang sama.</div></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="doc-section" id="usage"><h3>5. Cara Penggunaan</h3>'
            '<ol style="color:#425b75;font-size:13px;line-height:1.8">'
            '<li>Buka halaman <b>Classification</b> dan unggah satu file seismik.</li>'
            '<li>Untuk CSV/TXT/HDF5, tentukan sampling rate dan channel/dataset apabila diperlukan.</li>'
            '<li>Pilih SeismicNet, VOISS-Net, Seismic-VOISS-Net Multimodal CNN, atau Compare All Models.</li>'
            '<li>Pastikan rekaman menyediakan segmen minimal 60 detik, lalu tekan <b>Run Classification</b>.</li>'
            '<li>Amati waveform, spektrogram, hasil prediksi, class probability, dan Model Agreement apabila lebih dari satu model dijalankan.</li></ol>'
            '<p style="font-size:12px;color:#6b7d91">Format berkas yang diterima: CSV, TXT, WAV, MiniSEED, H5, dan HDF5.</p></div>',
            unsafe_allow_html=True
        )


# Sidebar navigation

st.session_state.setdefault("active_page", "Dashboard")
page_icons = {
    "Dashboard": "⌂",
    "Classification": "▦",
    "History": "◷",
    "About": "ⓘ",
    "Documentation": "▤",
}
with st.sidebar:
    logo_uri = asset_data_uri("voiss-logo.svg")
    if logo_uri:
        st.markdown(
            f'<div class="sidebar-brand"><img src="{logo_uri}"><div>'
            f'<div class="sidebar-brand-title">VOISS</div>'
            f'<div class="sidebar-brand-sub">Volcanic Earthquake<br>Classification System</div></div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("## 🌋 VOISS")
        st.caption("Volcanic Earthquake Classification System")

    for item, icon in page_icons.items():
        if st.button(
            f"{icon}   {item}",
            key=f"nav_{item}",
            type="primary" if st.session_state["active_page"] == item else "secondary",
            use_container_width=True
        ):
            st.session_state["active_page"] = item
            st.rerun()

    st.markdown(
        '<div style="margin-top:38px;padding:12px 4px;border-top:1px solid rgba(222,237,250,.16);'
        'color:#b8cde1;font-size:11px;line-height:1.6">SeismicNet · VOISS-Net<br>Seismic-VOISS-Net Multimodal CNN<br><br>© 2026 Farrelius Kevin</div>',
        unsafe_allow_html=True
    )

page = st.session_state["active_page"]
if page == "Dashboard":
    dashboard()
elif page == "Classification":
    classification()
elif page == "History":
    history_page()
elif page == "About":
    about_page()
else:
    documentation_page()
