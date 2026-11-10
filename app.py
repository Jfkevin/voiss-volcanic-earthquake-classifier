from __future__ import annotations

import io
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
:root { --navy:#10243f; --blue:#1768b5; --pale:#eef5fd; }
.stApp { background:linear-gradient(180deg,#f5f8fc 0%,#eef3f9 100%); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10243f 0%,#142f50 100%); }
[data-testid="stSidebar"] * { color:#f5f8fd !important; }
.voiss-header { background:linear-gradient(100deg,#10243f 0%,#1d4f80 65%,#193a60 100%);
  color:white;padding:20px 24px;border-radius:16px;margin-bottom:18px; }
.voiss-header h1 { color:white;margin:0;font-size:27px; }
.voiss-header p { color:#d9e8f7;margin:6px 0 0 0; }
.voiss-card { background:white;border:1px solid #dce6f2;border-radius:13px;
  padding:15px 17px;margin-bottom:12px;box-shadow:0 2px 9px rgba(20,55,95,.045); }
.voiss-muted { color:#63758c;font-size:12px; }
div.stButton > button[kind="primary"] { background:#1768c2;border-color:#1768c2; }
</style>
""", unsafe_allow_html=True)

def now_wib():
    return datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%d %b %Y · %H:%M:%S WIB")

def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="voiss-header"><div style="display:flex;justify-content:space-between;gap:16px;align-items:center;">'
        f'<div><h1>VOISS · {title}</h1><p>{subtitle}</p></div>'
        f'<div style="text-align:right;white-space:nowrap;font-weight:650">{now_wib()}</div></div></div>',
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
    cols=st.columns(len(results))
    for col,r in zip(cols,results):
        code=r["predicted_code"]
        with col:
            st.markdown('<div class="voiss-card">',unsafe_allow_html=True)
            st.subheader(r["model"])
            st.markdown(f'<div style="background:#fff0f0;color:#a32323;padding:9px;border-radius:8px;text-align:center;font-size:19px;font-weight:800;">{code} · {r["predicted_class"]}</div>',unsafe_allow_html=True)
            st.metric("Confidence (Softmax score)",f'{r["confidence"]*100:.2f}%')
            st.markdown("**Class Probability**")
            for c in CODES:
                score=r["probabilities"][c]
                st.write(f"{c} · {CLASS_NAMES[c]} — {score*100:.2f}%")
                st.progress(float(np.clip(score,0,1)))
            st.markdown("</div>",unsafe_allow_html=True)

def dashboard():
    page_header("Volcanic Earthquake Classification","SeismicNet (CNN 1D) · VOISS-Net (CNN 2D) · Seismic-VOISS-Net Multimodal CNN")
    st.markdown('<div class="voiss-card" style="background:linear-gradient(110deg,#fff 0%,#e7f1fc 58%,#d8e8f8 100%);"><h1 style="color:#153c67;margin:0;">VOISS</h1><h3 style="color:#244b76;margin:2px 0 8px;">Volcanic Earthquake Classification System</h3><p style="color:#425b76;">Sistem demonstrasi klasifikasi sinyal gempa vulkanik berbasis deep learning menggunakan waveform, spektrogram, dan pendekatan multimodal.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Arsitektur CNN","3")
    b.metric("Kelas keluaran","4")
    c.metric("Input model","100 Hz · 60 detik")
    st.markdown("### Empat kelas klasifikasi")
    cols=st.columns(4)
    for col,code in zip(cols,CODES):
        col.markdown(f'<div class="voiss-card" style="border-top:4px solid {COLORS[code]};"><h2 style="color:{COLORS[code]};margin:0;">{code}</h2><b>{CLASS_NAMES[code]}</b></div>',unsafe_allow_html=True)
    missing=model_status()
    if missing:
        st.warning("Model belum seluruhnya tersedia di folder artifacts.")
        st.code("\n".join(missing))
    else:
        st.success("Ketiga model ditemukan. Aplikasi siap untuk inference.")
    st.info("Buka Classification untuk mengunggah file seismik dan menjalankan prediksi.")

def classification():
    page_header("Classification","Unggah satu rekaman, pilih model, dan jalankan klasifikasi.")
    missing=model_status()
    if missing:
        st.error("Artefak model belum lengkap. Tiga file .keras harus berada di folder artifacts.")
        st.code("\n".join(missing))
        st.stop()
    st.markdown("### 1. Input Seismic Data")
    up=st.file_uploader("Upload seismic file",type=ACCEPTED,help="CSV, TXT, WAV, MiniSEED, H5, HDF5. Satu prediksi menggunakan segmen 60 detik.")
    if up is None:
        st.info("Upload satu waveform untuk memulai. Input model: 6.000 sampel, 100 Hz, satu channel.")
        return
    raw=up.getvalue()
    digest=hashlib.sha256(raw).hexdigest()
    ext=Path(up.name).suffix.lower().lstrip(".")
    dataset=None
    event=0
    channel=0
    trace=0
    manual_fs=100.0
    if ext in {"csv","txt","h5","hdf5"}:
        manual_fs=st.number_input("Sampling rate (Hz)",min_value=0.1,max_value=10000.0,value=100.0,step=1.0)
    if ext in {"h5","hdf5"}:
        try:
            names=get_hdf5_datasets(raw)
            if not names: st.error("Tidak ditemukan dataset numerik pada HDF5."); return
            dataset=st.selectbox("Dataset HDF5",list(names))
            shape=names[dataset]
            if len(shape)==3:
                event=st.number_input("Indeks event",0,max(0,shape[0]-1),0)
                channel=st.number_input("Indeks channel",0,max(0,shape[2]-1),0)
            elif len(shape)==2 and shape[1]==6000 and shape[0]!=6000:
                event=st.number_input("Indeks event",0,max(0,shape[0]-1),0)
            elif len(shape)==2:
                channel=st.number_input("Indeks channel",0,max(0,shape[1]-1),0)
        except Exception as e: st.error(f"Gagal membaca struktur HDF5: {e}"); return
    if ext=="wav":
        try:
            sr0,arr=wavfile.read(io.BytesIO(raw))
            if arr.ndim==2: channel=st.selectbox("Channel WAV",list(range(arr.shape[1])))
            st.caption(f"Sampling rate dari WAV: {sr0} Hz")
        except Exception as e: st.error(f"WAV tidak valid: {e}"); return
    if ext in {"mseed","msd"}:
        try:
            from obspy import read
            stream=read(io.BytesIO(raw))
            trace=st.selectbox("Trace MiniSEED",list(range(len(stream))))
        except Exception as e: st.error(f"MiniSEED tidak valid: {e}"); return
    try:
        preview,preview_sr,_=read_signal(raw,up.name,manual_fs,int(channel),int(event),int(trace),dataset)
        duration=len(preview)/preview_sr
        st.caption(f"Durasi terdeteksi: {duration:.2f} detik · {len(preview):,} sampel · {preview_sr:g} Hz")
        start=0.0
        if duration>60:
            start=st.slider("Awal segmen 60 detik (detik)",0.0,float(duration-60),0.0,1.0)
    except Exception as e:
        st.error(f"File tidak dapat dibaca: {e}")
        return
    st.markdown("### 2. Model Selection")
    choice=st.radio("Pilih pendekatan",["SeismicNet (CNN 1D)","VOISS-Net (CNN 2D)","Seismic-VOISS-Net Multimodal CNN","Compare All Models"],horizontal=True)
    selected={
        "SeismicNet (CNN 1D)":["SeismicNet"],
        "VOISS-Net (CNN 2D)":["VOISS-Net"],
        "Seismic-VOISS-Net Multimodal CNN":["Seismic-VOISS-Net Multimodal CNN"],
        "Compare All Models":list(MODEL_FILES),
    }[choice]
    if st.button("▶ Run Classification",type="primary",use_container_width=True):
        try:
            x,meta=make_segment(preview,preview_sr,float(start))
            with st.spinner("Model sedang melakukan klasifikasi..."):
                results=[]
                for name in selected:
                    results.append(predict_model(name,x))
            agreement=agreement_for(results)
            run=now_wib()
            row={"run_at":run,"file_name":up.name,"mode":choice,"results":results,"agreement":agreement,
                 "duration":60.0,"sample_rate":100,"samples":6000}
            save_history(row)
            st.session_state["last_prediction"]={"hash":digest,"waveform":x,"row":row}
            st.success("Klasifikasi selesai.")
        except Exception as e:
            st.error(f"Klasifikasi gagal: {e}")
            st.exception(e)
    last=st.session_state.get("last_prediction")
    if last and last["hash"]==digest:
        row=last["row"]
        st.markdown("### 3. Signal Visualization")
        plot_signals(last["waveform"])
        st.markdown("### 4. Classification Result")
        render_result_cards(row["results"])
        st.markdown("### 5. Model Agreement")
        if row["agreement"].get("available"):
            ag=row["agreement"]
            if ag.get("consensus_code"):
                st.success(f'{ag["message"]} Consensus: {ag["consensus_code"]} — {CLASS_NAMES[ag["consensus_code"]]}')
            else: st.warning(ag["message"])
            st.metric("Model Agreement",f'{ag["agreement_count"]}/{ag["model_count"]} models')
        else:
            st.info("Model Agreement tersedia pada Compare All Models.")
        st.markdown("### 6. Event Information")
        st.dataframe(pd.DataFrame([
            {"Informasi":"File Name","Nilai":row["file_name"]},
            {"Informasi":"Duration","Nilai":"60.00 seconds"},
            {"Informasi":"Sampling Rate","Nilai":"100 Hz"},
            {"Informasi":"Number of Samples","Nilai":"6,000"},
            {"Informasi":"Analysis Time","Nilai":row["run_at"]},
        ]),use_container_width=True,hide_index=True)
        st.caption("Softmax scores are model scores and have not been demonstrated to be calibrated probabilities.")

def history_page():
    page_header("Prediction History","Riwayat prediksi yang tersimpan sementara pada sesi browser ini.")
    hist=ensure_history()
    if not hist:
        st.info("Belum ada prediksi pada sesi ini.")
        return
    df=pd.DataFrame([{"No":i+1,"Tanggal & Waktu":r["run_at"],"Nama File":r["file_name"],"Model":r["mode"],
                      "Prediksi":", ".join(x["predicted_code"] for x in r["results"]),
                      "Agreement":r["agreement"].get("message","N/A")} for i,r in enumerate(hist)])
    q=st.text_input("Cari nama file")
    if q: df=df[df["Nama File"].str.contains(q,case=False,na=False)]
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Export History CSV",df.to_csv(index=False).encode("utf-8-sig"),"voiss_history.csv","text/csv")
    with st.expander("Detail prediksi"):
        for r in hist:
            st.markdown(f"**{r['run_at']} · {r['file_name']} · {r['mode']}**")
            for p in r["results"]:
                st.write(p["model"],"—",p["predicted_code"],f'{p["confidence"]*100:.2f}%')
    if st.button("Hapus history"):
        st.session_state["voiss_history"]=[]
        st.rerun()
    st.caption("History hanya berlaku di sesi browser saat ini. Riwayat permanen membutuhkan database eksternal.")

def about_page():
    page_header("About the Researcher","Profil peneliti, latar belakang, dan identitas proyek VOISS.")
    a,b=st.columns([1,2])
    with a:
        photo=ASSETS/"profile.jpg"
        if photo.exists(): st.image(str(photo),use_container_width=True)
        else: st.markdown('<div class="voiss-card" style="height:230px;display:grid;place-items:center;font-size:52px;">👤<br><small style="font-size:13px">Foto profil dapat ditambahkan</small></div>',unsafe_allow_html=True)
    with b:
        st.title("Farrelius Kevin")
        st.write("Informatics · Data Science · Machine Learning · Deep Learning")
        st.write("VOISS dikembangkan sebagai proyek penelitian dan portofolio untuk mengeksplorasi CNN 1D, CNN 2D, dan multimodal feature fusion pada klasifikasi gempa vulkanik.")
        st.write("Dataset: Llaima volcano seismic dataset, Chile.")
    st.markdown("### Tentang penelitian")
    st.write("SeismicNet memproses waveform, VOISS-Net memproses spektrogram, dan Seismic-VOISS-Net Multimodal CNN menggabungkan kedua representasi untuk mengklasifikasikan Long Period (LP), Tectonic (TC), Tremor (TR), dan Volcano-Tectonic (VT).")
    st.info("Foto profil dan tautan email, LinkedIn, serta GitHub dapat ditambahkan pada folder assets dan bagian halaman About.")

def documentation_page():
    page_header("Documentation","Arsitektur model, input, preprocessing, evaluasi, dan batasan VOISS.")
    st.markdown("## Arsitektur tiga model")
    with st.container(border=True):
        st.subheader("1. SeismicNet (CNN 1D)")
        st.write("Model unimodal yang memproses waveform berdimensi satu untuk mengekstraksi pola temporal sinyal.")
        st.code("""Waveform (6000, 1)
→ Conv1D 32, kernel 64, stride 2 + ReLU
→ MaxPool1D 4, stride 4 → Dropout 0.15
→ Conv1D 64, kernel 32, stride 2 + ReLU
→ MaxPool1D 4, stride 4 → Dropout 0.35
→ Conv1D 128, kernel 16, stride 2 + ReLU
→ MaxPool1D 4, stride 4
→ Conv1D 256, kernel 8, stride 2 + ReLU
→ Conv1D 1401, kernel 16, stride 2 + ReLU
→ Flatten → Dense 1500 + ReLU → Dropout 0.75
→ Dense 4 + Softmax""")
        st.metric("Total parameters","12,513,181")
    with st.container(border=True):
        st.subheader("2. VOISS-Net (CNN 2D)")
        st.write("Model unimodal yang memproses spektrogram hasil STFT untuk mengekstraksi pola waktu-frekuensi.")
        st.code("""Spectrogram (96, 60, 1)
→ Conv2D 32, kernel 3×3 + ReLU → MaxPool2D 3×3
→ Conv2D 64, kernel 3×3 + ReLU → MaxPool2D 3×3
→ Conv2D 128, kernel 3×3 + ReLU → MaxPool2D 3×3
→ Flatten → Dropout 0.2
→ Dense 128 + ReLU → Dropout 0.5
→ Dense 64 + ReLU → Dropout 0.5
→ Dense 4 + Softmax""")
        st.metric("Total parameters","199,620")
    with st.container(border=True):
        st.subheader("3. Seismic-VOISS-Net Multimodal CNN")
        st.write("Model multimodal dengan cabang SeismicNet untuk waveform dan cabang VOISS-Net untuk spektrogram. Kedua vektor fitur digabungkan melalui Concatenate sebelum classification head.")
        a,b=st.columns(2)
        a.markdown("**Cabang waveform**")
        a.code("Input (6000,1) → Conv1D 32/64/128/256/1401 → Pooling → Flatten → Dense 1500 → Dropout 0.75")
        b.markdown("**Cabang spektrogram**")
        b.code("Input (96,60,1) → Conv2D 32/64/128 → Pooling → Flatten → Dense 128 → Dense 64")
        st.markdown("**Feature fusion dan output**")
        st.code("Concatenate → Dense 256 + ReLU → Dropout 0.5 → Dense 4 + Softmax")
        st.metric("Total parameters","13,108,205")
    st.markdown("## Input dan preprocessing")
    c1,c2,c3=st.columns(3)
    c1.metric("Sampling rate","100 Hz")
    c2.metric("Durasi","60 detik")
    c3.metric("Sampel","6.000")
    st.write("Spectrogram: mean removal → Tukey window alpha=0.1 → STFT Hann (nperseg=1000, noverlap=900) → dB → crop 0.5–10 Hz → 60 time columns → per-spectrogram Min-Max [0,1].")
    st.write("Input tanpa batch: waveform (6000,1), spectrogram (96,60,1). Multimodal memerlukan kedua input dari event yang sama.")
    st.markdown("## Label klasifikasi")
    for code in CODES: st.markdown(f'<div class="voiss-card" style="border-left:5px solid {COLORS[code]};"><b>{code}</b> — {CLASS_NAMES[code]}</div>',unsafe_allow_html=True)
    st.markdown("## Evaluasi dan keterbatasan")
    st.write("Metrik utama penelitian: Macro F1-score, Macro Precision, dan Macro Recall. Accuracy tetap ditampilkan sebagai metrik pelengkap. Evaluasi menggunakan prediksi out-of-fold dari Stratified 5-Fold Cross Validation.")
    st.warning("Skor Softmax belum terbukti terkalibrasi. Model dikembangkan menggunakan dataset Llaima; generalisasi ke gunung, stasiun, instrumen, atau kondisi akuisisi lain memerlukan validasi eksternal. VOISS adalah prototipe riset, bukan pengganti interpretasi ahli seismologi.")

# Sidebar and navigation
with st.sidebar:
    st.markdown("## 🌋 VOISS")
    st.caption("Volcanic Earthquake Classification System")
    st.markdown("---")
    page=st.radio("Navigasi",["Dashboard","Classification","History","About","Documentation"],label_visibility="collapsed")
    st.markdown("---")
    st.caption("SeismicNet · VOISS-Net · Multimodal CNN")
    st.caption("© 2026 Farrelius Kevin")

if page=="Dashboard": dashboard()
elif page=="Classification": classification()
elif page=="History": history_page()
elif page=="About": about_page()
else: documentation_page()
