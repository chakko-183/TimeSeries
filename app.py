import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.interpolate import interp1d
import os
import time

# ==============================================================================
# 1. SETUP MODEL & CONFIG
# ==============================================================================

MODEL_FILES = {
    "Random Forest (RF)": 'rf_shapeall.joblib',
    "Support Vector Machine (SVM)": 'svm_shapeall.joblib',
    "1-NN DTW": '1nn_dtw_shapeall.joblib'
}

# Metrik Kinerja (Sesuaikan dengan data Anda)
MODEL_METRICS = {
    "Random Forest (RF)": {"Akurasi": "74.00%", "F1": 0.73},
    "Support Vector Machine (SVM)": {"Akurasi": "80.50%", "F1": 0.81},
    "1-NN DTW": {"Akurasi": "85.20%", "F1": 0.84}
}

TIME_STEPS = 512

@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")
        return None

# ==============================================================================
# 2. FUNGSI EKSTRAKSI GAMBAR KE TIME SERIES
# ==============================================================================

def image_to_timeseries(uploaded_file):
    """Mengubah gambar menjadi 512 titik data berdasarkan kontur."""
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing (Binarization)
    # Kita asumsikan objek lebih gelap dari background atau sebaliknya
    # Menggunakan Otsu's thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Cari Kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    # Ambil kontur terbesar
    cnt = max(contours, key=cv2.contourArea)
    
    # Hitung Jarak dari Centroid ke setiap titik di kontur (Centroid Distance Function)
    M = cv2.moments(cnt)
    if M['m00'] == 0: return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    distances = []
    for point in cnt:
        x, y = point[0]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        distances.append(dist)
    
    # Interpolasi agar menjadi tepat 512 titik
    distances = np.array(distances)
    x_old = np.linspace(0, 1, len(distances))
    x_new = np.linspace(0, 1, TIME_STEPS)
    f = interp1d(x_old, distances, kind='linear')
    signal_512 = f(x_new)
    
    # Normalisasi Z-Score (seperti pada data training)
    mean = np.mean(signal_512)
    std = np.std(signal_512)
    signal_scaled = (signal_512 - mean) / (std if std > 0 else 1)
    
    return signal_scaled

# ==============================================================================
# 3. ANTARMUKA STREAMLIT
# ==============================================================================

st.set_page_config(page_title="ShapeAll Image Classifier", layout="wide")
st.title("🚀 Klasifikasi Bentuk dari Foto (ShapesAll)")

# --- Sidebar ---
st.sidebar.header("Konfigurasi Model")
selected_model_name = st.sidebar.selectbox("Pilih Model:", list(MODEL_FILES.keys()))
metrics = MODEL_METRICS.get(selected_model_name)
st.sidebar.info(f"Akurasi: {metrics['Akurasi']} | F1: {metrics['F1']}")

model = load_model(MODEL_FILES[selected_model_name])
is_dtw = "DTW" in selected_model_name

# --- Main Page ---
if model:
    st.header("1. Upload Foto")
    uploaded_file = st.file_uploader("Unggah foto bentuk (lingkaran, kotak, segitiga, dll)", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Foto yang diunggah", use_container_width=True)
        
        if st.button("Proses & Prediksi", type="primary"):
            # Ekstraksi sinyal
            ts_data = image_to_timeseries(uploaded_file)
            
            if ts_data is not None:
                # Reshape untuk model
                if is_dtw:
                    X_input = ts_data.reshape(1, TIME_STEPS, 1)
                else:
                    X_input = ts_data.reshape(1, TIME_STEPS)
                
                # Prediksi
                with st.spinner('Menganalisis bentuk...'):
                    prediction = model.predict(X_input)
                
                # --- HASIL ---
                st.header("2. Hasil Klasifikasi")
                st.balloons()
                st.success(f"Bentuk terdeteksi sebagai: **{prediction[0]}**")
                
                # --- GRAFIK TIME SERIES ---
                st.subheader("3. Grafik Time Series (Hasil Ekstraksi Foto)")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                ax.plot(ts_data, color='#1f77b4', linewidth=2)
                ax.set_title(f"Sinyal Bentuk - Prediksi: {prediction[0]}")
                ax.set_xlabel(f"Titik Waktu ({TIME_STEPS})")
                ax.set_ylabel("Nilai (Z-Score Scaled)")
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
            else:
                st.error("Gagal mendeteksi objek dalam gambar. Pastikan gambar memiliki latar belakang yang bersih.")
else:
    st.error("Model tidak ditemukan. Pastikan file .joblib ada di folder aplikasi.")