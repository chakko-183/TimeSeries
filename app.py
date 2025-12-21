import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.interpolate import interp1d
import os

# ==============================================================================
# 1. SETUP MODEL
# ==============================================================================

MODEL_FILES = {
    "Random Forest (RF)": 'rf_shapeall.joblib',
    "Support Vector Machine (SVM)": 'svm_shapeall.joblib',
    "1-NN DTW": '1nn_dtw_shapeall.joblib'
}

TIME_STEPS = 512

@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")
        return None

# ==============================================================================
# 2. FUNGSI PEMROSESAN GAMBAR (IMAGE TO TIME SERIES)
# ==============================================================================

def process_image_to_sequence(uploaded_file):
    """Mengubah gambar menjadi urutan 512 titik data (Time Series)."""
    # 1. Load Gambar
    image = Image.open(uploaded_file).convert('L') # Convert ke Grayscale
    img_array = np.array(image)

    # 2. Thresholding (Memisahkan bentuk dari background)
    # Asumsi: Background terang, bentuk gelap. Kita balik (INV) agar bentuk jadi putih
    _, thresh = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)

    # 3. Cari Kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None

    # Ambil kontur terbesar (objek utama)
    cnt = max(contours, key=cv2.contourArea)
    
    # Ambil koordinat Y sebagai sinyal data (representasi bentuk)
    # Kita ambil 1D array dari titik-titik kontur
    points = cnt[:, 0, 1].astype(float) 

    # 4. Resampling ke 512 titik (Sangat Penting!)
    # Karena model Anda dilatih dengan tepat 512 titik
    x_old = np.linspace(0, 1, len(points))
    x_new = np.linspace(0, 1, TIME_STEPS)
    interpolation_func = interp1d(x_old, points, kind='linear')
    sequence_512 = interpolation_func(x_new)

    # 5. Normalisasi (Z-Score Scaling) sesuai fungsi preprocess asli Anda
    mean = np.mean(sequence_512)
    std = np.std(sequence_512)
    final_sequence = (sequence_512 - mean) / (std if std > 0 else 1)
    
    return final_sequence

# ==============================================================================
# 3. ANTARMUKA STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Image to Shape Classifier", layout="wide")
st.title("📸 Klasifikasi Bentuk via Unggah Foto")
st.write("Unggah foto bentuk (lingkaran, segitiga, dll) untuk diprediksi oleh model.")

# Sidebar Model
selected_model_name = st.sidebar.selectbox("Pilih Model:", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[selected_model_name])
is_dtw = "DTW" in selected_model_name

if model:
    # --- Input Foto ---
    st.header("1. Unggah Foto Bentuk")
    uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan preview gambar
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Gambar Original", use_container_width=True)

        if st.button("Lakukan Prediksi", type="primary"):
            with st.spinner('Mengekstraksi bentuk dan memprediksi...'):
                # Proses gambar jadi data numerik
                processed_data = process_image_to_sequence(uploaded_file)

                if processed_data is not None:
                    # Siapkan format input untuk model
                    if is_dtw:
                        X_input = processed_data.reshape(1, TIME_STEPS, 1)
                    else:
                        X_input = processed_data.reshape(1, TIME_STEPS)

                    # Prediksi
                    prediction = model.predict(X_input)

                    # --- Output (Sama dengan format sebelumnya) ---
                    st.header("2. Hasil Klasifikasi")
                    st.balloons()
                    st.success(f"Prediksi Kelas Bentuk Adalah: **{prediction[0]}**")

                    # Visualisasi Hasil Ekstraksi
                    st.subheader("Visualisasi Fitur dari Gambar")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(processed_data, color='firebrick')
                    ax.set_title(f"Sinyal Ekstraksi Gambar (Input Model)")
                    ax.set_xlabel("Titik Koordinat (512)")
                    ax.set_ylabel("Nilai Normalisasi")
                    st.pyplot(fig)
                else:
                    st.error("Gagal mendeteksi bentuk dalam gambar. Pastikan gambar memiliki kontras yang jelas.")
else:
    st.error("Model tidak ditemukan.")