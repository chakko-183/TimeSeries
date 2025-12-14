import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ==============================================================================
# 1. SETUP MODEL & METRIK KINERJA
# ==============================================================================

# Definisikan nama file model (path relatif)
MODEL_FILES = {
    # File-file ini HARUS ada di folder yang sama dengan app.py
    "Random Forest (RF)": 'rf_shapeall.joblib',
    "Support Vector Machine (SVM)": 'svm_shapeall.joblib',
    "1-NN DTW": '1nn_dtw_shapeall.joblib'
}

# Metrik Kinerja (Diambil dari hasil evaluasi data uji Anda)
MODEL_METRICS = {
    # Sesuaikan nilai Akurasi dan F1-Score ini berdasarkan laporan teknis Anda
    "Random Forest (RF)": {"Akurasi": "74.00%", "F1": 0.73},
    "Support Vector Machine (SVM)": {"Akurasi": "80.50%", "F1": 0.81}, # Angka Contoh
    "1-NN DTW": {"Akurasi": "85.20%", "F1": 0.84} # Angka Contoh
}

TIME_STEPS = 512

@st.cache_resource
def load_model(model_path):
    """Memuat model yang sudah dilatih dari file .joblib."""
    model_name = os.path.basename(model_path)
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.sidebar.error(f"File model '{model_name}' tidak ditemukan di folder deployment.")
        return None
    except Exception as e:
        # Menangkap error dependensi (tslearn, sklearn version)
        st.sidebar.error(f"Gagal memuat {model_name}. Cek dependensi (tslearn/sklearn version). Error: {e}")
        return None

def preprocess_input(time_series_data, is_dtw_model):
    """Melakukan preprocessing Z-Score Scaling untuk RF/SVM atau Reshape untuk DTW."""
    data_np = np.array(time_series_data)
    
    if is_dtw_model:
        # DTW memerlukan format 3D (samples, timesteps, dimension)
        return data_np.reshape(1, TIME_STEPS, 1)
    else:
        # RF/SVM memerlukan format 2D
        data_2D = data_np.reshape(1, TIME_STEPS)
        
        # Standard Scaling (Z-Score) pada sampel tunggal
        mean = np.mean(data_2D)
        std = np.std(data_2D)
        
        # Hindari pembagian dengan nol
        X_scaled = (data_2D - mean) / (std if std > 0 else 1)
        return X_scaled

# ==============================================================================
# 2. ANTARMUKA STREAMLIT
# ==============================================================================

st.set_page_config(page_title="ShapeAll TS Classifier", layout="wide")
st.title("🚀 Deployment Klasifikasi Bentuk Time Series (ShapesAll)")
st.subheader("Uji Kinerja Model Klasifikasi di Lingkungan Lokal") 

# --- Sidebar: Pemilihan Model ---
st.sidebar.header("Pilih Model untuk Prediksi")
selected_model_name = st.sidebar.selectbox(
    "Model Klasifikasi:",
    list(MODEL_FILES.keys())
)
selected_model_path = MODEL_FILES[selected_model_name]

# Memuat model
model = load_model(selected_model_path)
is_dtw = "DTW" in selected_model_name

# --- Tampilkan Status Model dan Metrik ---
if model:
    st.sidebar.success(f"Model {selected_model_path} berhasil dimuat.")
    
    # Tampilkan metrik kinerja di sidebar
    metrics = MODEL_METRICS.get(selected_model_name)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Metrik Kinerja Uji")
    st.sidebar.write(f"**Akurasi Uji:** {metrics['Akurasi']}")
    st.sidebar.write(f"**Weighted F1:** {metrics['F1']}")
    st.sidebar.markdown("---")
    
    st.markdown(f"**Model Aktif:** **{selected_model_name}**")
    
    st.header("1. Input Data Time Series Baru")
    st.info(f"Dataset ShapesAll memerlukan **{TIME_STEPS}** titik waktu.")
    
    # --- Input Data dan Tombol Generate ---
    if st.button("Generate Contoh Time Series Acak"):
        simulated_ts = np.random.randn(TIME_STEPS) * 5 + 10 
        ts_text = ', '.join(map(str, simulated_ts))
    else:
        ts_text = ""
        
    ts_input = st.text_area(
        f"Masukkan {TIME_STEPS} nilai (dipisahkan koma):",
        value=ts_text,
        height=150
    )

    # --- Bagian Prediksi ---
    if st.button("Lakukan Prediksi", type="primary"):
        if ts_input:
            try:
                # 1. Parsing Input
                ts_list = [float(x.strip()) for x in ts_input.split(',') if x.strip()]
                
                if len(ts_list) != TIME_STEPS:
                    st.error(f"Input harus memiliki {TIME_STEPS} nilai, Anda memasukkan {len(ts_list)}.")
                else:
                    # 2. Preprocessing
                    X_processed = preprocess_input(ts_list, is_dtw)
                    
                    # 3. Prediksi
                    with st.spinner('Memproses dan memprediksi...'):
                        time.sleep(0.5) 
                        prediction = model.predict(X_processed)
                    
                    st.header("2. Hasil Klasifikasi")
                    st.balloons()
                    st.success("Prediksi Kelas Bentuk Adalah:")
                    st.markdown(f"## **{prediction[0]}**")
                    
                    # 4. Visualisasi
                    st.subheader("Visualisasi Input yang Diprediksi")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    if is_dtw:
                        ax.plot(X_processed[0, :, 0])
                    else:
                        ax.plot(X_processed[0])
                        
                    ax.set_title(f"Input Diprediksi sebagai: {prediction[0]}")
                    ax.set_xlabel(f"Titik Waktu ({TIME_STEPS})")
                    ax.set_ylabel("Nilai (Scaled)")
                    st.pyplot(fig) 
                    
            except ValueError:
                st.error("Pastikan semua input adalah angka dan dipisahkan oleh koma.")
            except Exception as e:
                st.exception(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        else:
            st.warning("Silakan masukkan data time series atau generate contoh acak.")
            
# Tampilkan pesan jika model gagal dimuat
else:
    st.error("Aplikasi tidak berjalan. Pastikan file model yang dibutuhkan berada di direktori yang sama.")