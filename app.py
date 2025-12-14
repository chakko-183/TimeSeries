
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Definisikan path model di Drive
MODEL_FILES = {
    # Gunakan nama file saja (relative path)
    "Random Forest (RF)": 'rf_shapeall.joblib',
    "Support Vector Machine (SVM)": 'svm_shapeall.joblib',
    "1-NN DTW": '1nn_dtw_shapeall.joblib'
}

TIME_STEPS = 512

@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.sidebar.success(f"Model {model_path.split('/')[-1]} berhasil dimuat.")
        return model
    except FileNotFoundError:
        st.sidebar.error(f"File model '{model_path.split('/')[-1]}' tidak ditemukan di Drive.")
        return None

def preprocess_input(time_series_data, is_dtw_model):
    data_np = np.array(time_series_data)
    
    if is_dtw_model:
        return data_np.reshape(1, TIME_STEPS, 1)
    else:
        data_2D = data_np.reshape(1, TIME_STEPS)
        mean = np.mean(data_2D)
        std = np.std(data_2D)
        X_scaled = (data_2D - mean) / (std if std > 0 else 1)
        return X_scaled

# --- ANTARMUKA STREAMLIT ---
st.set_page_config(page_title="ShapeAll TS Classifier", layout="wide")
st.title("🚀 Deployment Klasifikasi Bentuk Time Series")
st.subheader("Menjalankan di Lingkungan Lokal (D:\)") 

st.sidebar.header("Pilih Model untuk Prediksi")
selected_model_name = st.sidebar.selectbox(
    "Model Klasifikasi:",
    list(MODEL_FILES.keys())
)
selected_model_path = MODEL_FILES[selected_model_name]

model = load_model(selected_model_path)
is_dtw = "DTW" in selected_model_name

if model:
    st.markdown(f"**Model Aktif:** **{selected_model_name}**")
    st.header("1. Input Data Time Series Baru")
    st.info(f"Dataset ShapeAll memerlukan {TIME_STEPS} titik waktu.")
    
    if st.button("Generate Contoh Time Series Acak"):
        simulated_ts = np.random.randn(TIME_STEPS) * 5 + 10 
        ts_text = ', '.join(map(str, simulated_ts))
    else:
        ts_text = ""
        
    ts_input = st.text_area(f"Masukkan {TIME_STEPS} nilai:", value=ts_text, height=150)

    if st.button("Lakukan Prediksi", type="primary"):
        if ts_input:
            try:
                ts_list = [float(x.strip()) for x in ts_input.split(',') if x.strip()]
                if len(ts_list) != TIME_STEPS:
                    st.error(f"Input harus memiliki {TIME_STEPS} nilai, Anda memasukkan {len(ts_list)}.")
                else:
                    X_processed = preprocess_input(ts_list, is_dtw)
                    with st.spinner('Memproses dan memprediksi...'):
                        time.sleep(0.5) 
                        prediction = model.predict(X_processed)
                    
                    st.header("2. Hasil Klasifikasi")
                    st.success("Prediksi Kelas Bentuk Adalah:")
                    st.markdown(f"## **{prediction[0]}**")
                    
                    st.subheader("Visualisasi Input")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    if is_dtw:
                        ax.plot(X_processed[0, :, 0])
                    else:
                        ax.plot(X_processed[0])
                    ax.set_title(f"Input Diprediksi sebagai: {prediction[0]}")
                    st.pyplot(fig)
                    
            except Exception as e:
                st.exception(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Silakan masukkan data time series atau generate contoh acak.")
