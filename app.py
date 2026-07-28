import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Cek ketersediaan XGBoost
try:
    from xgboost import XGBClassifier
    xgb = True
except ImportError:
    xgb = False

# ==========================================================
# KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================================
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection CSS untuk Tampilan Modern & Sidebar
st.markdown("""
    <style>
    /* Styling Header Utama */
    .main-header {
        background: linear-gradient(135deg, #e53935 0%, #e35d5b 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        margin-top: 8px;
        margin-bottom: 0;
        opacity: 0.9;
    }
    
    /* Custom Styling Sidebar Branding */
    .sidebar-brand {
        text-align: center;
        padding: 10px;
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #ef9a9a;
    }
    .sidebar-brand h2 {
        color: #c62828 !important;
        margin: 0;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    /* Custom Alert Badges */
    .badge-danger {
        background-color: #ffebee;
        color: #c62828;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #c62828;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .badge-success {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 5px solid #2e7d32;
        font-weight: 600;
        font-size: 1.1rem;
    }

    /* Subheader Styling */
    .sub-title {
        color: #1e293b;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Utama Berwarna
st.markdown("""
    <div class="main-header">
        <h1>Prediksi Risiko Penyakit Jantung</h1>
        <p>Aplikasi analisis medis berbasis Machine Learning untuk deteksi dini risiko penyakit jantung secara presisi.</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR NAVIGATION & CONTROLS (RAPAT & TERTATA)
# ==========================================================
with st.sidebar:
    # Branding Mini
    st.markdown("""
        <div class="sidebar-brand">
            <h2>❤️ Heart Predict ML</h2>
            <small style="color: #555;">Sistem Analisis Medis</small>
        </div>
    """, unsafe_allow_html=True)

    # Pengelompokan Menu Navigasi
    st.markdown("### 📍 Navigasi Utamanya")
    
    menu = st.radio(
        "Pilih Modul Sistem:",
        [
            "🏠 Home",
            "📊 Dataset Overview",
            "📈 Exploratory Data Analysis (EDA)",
            "⚙️ Data Preprocessing",
            "🤖 Model Training & Comparison",
            "🔍 Prediksi Manual Pasien",
            "📁 Prediksi Massal (CSV Upload)"
        ]
    )

    st.markdown("---")

    # Fitur Dinamis di Sidebar (Hanya Muncul jika Memilih Prediksi Manual)
    selected_model_name = None
    if menu == "🔍 Prediksi Manual Pasien":
        st.markdown("### ⚙️ Konfigurasi Prediksi")
        if os.path.exists("Semua_Model.pkl"):
            models_dict = joblib.load("Semua_Model.pkl")
            selected_model_name = st.selectbox(
                "Pilih Algoritma Model:",
                list(models_dict.keys()),
                help="Pilih algoritma classification yang ingin digunakan untuk memprediksi data pasien."
            )
        else:
            st.warning("⚠️ Lakukan training terlebih dahulu untuk memilih model.")

    # Status Model Artifacts di Bagian Bawah Sidebar
    st.markdown("### 🛡️ Status Sistem")
    model_ready = os.path.exists("Semua_Model.pkl")
    if model_ready:
        st.success("🟢 Model Trained & Ready")
    else:
        st.caption("🔴 Status: Model Belum Ditingkatkan / Disimpan")

    st.caption("---")
    st.caption("© 2026 Medical Data Science Project")


# ==========================================================
# DATASET TRAINING (Heart Disease UCI)
# ==========================================================
TRAINING_DATASET = "dataset heart_disease_uci.csv"

@st.cache_data
def load_training_data():
    if os.path.exists(TRAINING_DATASET):
        try:
            df_data = pd.read_csv(TRAINING_DATASET, sep=';')
            if df_data.shape[1] <= 1:
                df_data = pd.read_csv(TRAINING_DATASET, sep=',')
        except:
            df_data = pd.read_csv(TRAINING_DATASET)
    else:
        dummy_data = {
            'age': [63, 67, 67, 37, 41, 56, 62, 57, 63, 53],
            'sex': ['Male', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male'],
            'cp': ['typical angina', 'asymptomatic', 'asymptomatic', 'non-anginal', 'atypical angina', 'non-anginal', 'asymptomatic', 'asymptomatic', 'asymptomatic', 'asymptomatic'],
            'trestbps': [145, 160, 120, 130, 130, 120, 140, 120, 130, 140],
            'chol': [233, 286, 229, 250, 204, 236, 268, 354, 254, 203],
            'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            'restecg': ['normal', 'lv hypertrophy', 'lv hypertrophy', 'normal', 'lv hypertrophy', 'normal', 'lv hypertrophy', 'normal', 'lv hypertrophy', 'normal'],
            'thalach': [150, 108, 129, 187, 172, 178, 160, 163, 147, 155],
            'exang': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
            'oldpeak': [2.3, 1.5, 2.6, 3.5, 1.4, 0.8, 3.6, 0.6, 1.4, 3.1],
            'slope': ['downsloping', 'flat', 'flat', 'downsloping', 'upsloping', 'upsloping', 'flat', 'upsloping', 'flat', 'downsloping'],
            'ca': [0, 3, 2, 0, 0, 0, 2, 0, 1, 0],
            'thal': ['fixed defect', 'normal', 'reversable defect', 'normal', 'normal', 'normal', 'normal', 'normal', 'reversable defect', 'reversable defect'],
            'num': [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        }
        df_data = pd.DataFrame(dummy_data)
    return df_data

df = load_training_data()

# ==========================================================
# FUNGSI PREPROCESSING & TRAINING
# ==========================================================
def prepare_training_data(data):
    """Membersihkan data, melakukan encoding, dan scaling."""
    data = data.copy().drop_duplicates()

    drop_cols = ["id", "dataset", "dataset_id", "Patient_ID"]
    drop_cols = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=drop_cols)

    target_col = None
    for col in ['num', 'target', 'heart_disease']:
        if col in data.columns:
            target_col = col
            break
            
    if target_col is None:
        target_col = data.columns[-1]

    if data[target_col].dtype != 'object':
        data['target_binary'] = data[target_col].apply(lambda x: "Sakit Jantung" if x > 0 else "Sehat")
    else:
        data['target_binary'] = data[target_col].astype(str)

    X = data.drop(columns=[target_col, 'target_binary'], errors='ignore')
    y = data['target_binary']

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    feature_encoders = {}
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))
        feature_encoders[col] = encoder

    X = X.fillna(X.mean(numeric_only=True))

    feature_columns = list(X.columns)
    scaler = StandardScaler()
    
    return X, y_encoded, target_encoder, feature_encoders, scaler, feature_columns

# ==========================================================
# ROUTING MEMBACA PILIHAN MENU SIDEBAR
# ==========================================================

# --- HOME ---
if menu == "🏠 Home":
    st.subheader("💡 Ringkasan Proyek Machine Learning")
    
    col_img, col_txt = st.columns([1, 1.2])
    with col_img:
        st.image(
            "https://cdn.kibrispdr.org/data/374/gambar-jantung-sederhana-10.png",
            caption="Penerapan Algoritma ML untuk Prediksi Penyakit Jantung",
            use_container_width=True
        )
    with col_txt:
        st.markdown(
            """
            ### Prediksi Risiko Penyakit Jantung
            Aplikasi ini memanfaatkan berbagai algoritma **Machine Learning** teruji untuk memprediksi risiko penyakit jantung secara cepat berdasarkan indikator medis non-invasif.

            #### 🛠️ Algoritma Klasifikasi Terintegrasi:
            * 🔹 Logistic Regression
            * 🔹 Decision Tree & Random Forest
            * 🔹 K-Nearest Neighbors (KNN)
            * 🔹 Naive Bayes
            * 🔹 Support Vector Machine (SVM)
            * 🔹 XGBoost *(opsional)*
            """
        )

    st.markdown("---")
    st.subheader("🚀 Fitur Utama Sistem")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📊 **Dataset & EDA**\nEksplorasi struktur data klinis, visualisasi sebaran target, serta analisis korelasi variabel secara visual.")
    with c2:
        st.success("🤖 **Multi-Model Training**\nPelatihan komparatif multi-algoritma secara real-time lengkap dengan metrik evaluasi (Accuracy, F1-Score, dll).")
    with c3:
        st.warning("🔍 **Prediksi Interaktif**\nForm input individual yang praktis serta fitur prediksi massal via upload file CSV.")

# --- DATASET ---
elif menu == "📊 Dataset Overview":
    st.header("📊 Dataset Training")
    st.info("Dataset yang digunakan: **Heart Disease UCI Database**")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Baris Data", f"{df.shape[0]} Sampel")
    m2.metric("Total Kolom Fitur", f"{df.shape[1]} Fitur")
    m3.metric("Total Missing Value", f"{df.isnull().sum().sum()} Data")

    st.markdown("<h4 class='sub-title'>Tabel Dataset Raw</h4>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<h4 class='sub-title'>Statistik Deskriptif</h4>", unsafe_allow_html=True)
        st.write(df.describe(include="all"))
    with c_right:
        st.markdown("<h4 class='sub-title'>Struktur & Tipe Data</h4>", unsafe_allow_html=True)
        st.write(df.dtypes.astype(str))

# --- EDA ---
elif menu == "📈 Exploratory Data Analysis (EDA)":
    st.header("📈 Exploratory Data Analysis (EDA)")
    
    st.markdown("<h4 class='sub-title'>Preview 5 Sampel Data Pertama</h4>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("<h4 class='sub-title'>Ringkasan Kualitas Data</h4>", unsafe_allow_html=True)
    info = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(info, use_container_width=True)

    target_col = 'num' if 'num' in df.columns else df.columns[-1]
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown(f"<h4 class='sub-title'>Distribusi Target ({target_col})</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df[target_col].value_counts()
        bars = ax.bar(counts.index.astype(str), counts.values, color=['#4caf50', '#e53935', '#ff9800', '#2196f3', '#9c27b0'])
        ax.set_xlabel("Kelas Target (0 = Sehat, >0 = Sakit)")
        ax.set_ylabel("Jumlah Pasien")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col_chart2:
        numeric = df.select_dtypes(include=np.number)
        if not numeric.empty:
            st.markdown("<h4 class='sub-title'>Matriks Korelasi (Fitur Numerik)</h4>", unsafe_allow_html=True)
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
            ax.set_yticks(range(len(corr.columns)))
            ax.set_yticklabels(corr.columns, fontsize=8)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            st.pyplot(fig)

    if not numeric.empty:
        st.markdown("<h4 class='sub-title'>Distribusi Fitur Numerik</h4>", unsafe_allow_html=True)
        num_cols = list(numeric.columns)
        cols_per_row = 3
        for i in range(0, len(num_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(num_cols[i:i+cols_per_row]):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    ax.hist(numeric[col_name].dropna(), bins=15, color='#42A5F5', edgecolor='white')
                    ax.set_title(f"{col_name}", fontsize=10, fontweight='bold')
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    st.pyplot(fig)

# --- PREPROCESSING ---
elif menu == "⚙️ Data Preprocessing":
    st.header("⚙️ Preprocessing Data")
    
    st.markdown(f"""
    <div class='custom-card'>
        <b>Status Dataset Awal:</b> <code>{df.shape[0]} baris</code> dan <code>{df.shape[1]} kolom</code>
    </div>
    """, unsafe_allow_html=True)

    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.success("✅ Preprocessing & Data Splitting Berhasil Dijalankan!")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("Jumlah Data Training (80%)", f"{X_train.shape[0]} Sampel")
        st.write("**Fitur Ter-encode:**", feature_columns)
    with col_p2:
        st.metric("Jumlah Data Testing (20%)", f"{X_test.shape[0]} Sampel")
        st.write("**Label Target Classes:**", list(target_encoder.classes_))

# --- TRAINING ---
elif menu == "🤖 Model Training & Comparison":
    st.header("🤖 Training Multi-Model Klasifikasi")
    
    X, y, target_encoder, feature_encoders, scaler, feature_columns = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(probability=True)
    }
    if xgb:
        models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", random_state=42)

    hasil = []
    trained_models = {}
    progress = st.progress(0)
    total = len(models)

    for i, (nama, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, pred)
        pre = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
        
        hasil.append([nama, acc, pre, rec, f1])
        trained_models[nama] = model
        progress.progress((i + 1) / total)

    hasil_df = pd.DataFrame(hasil, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    hasil_df = hasil_df.sort_values(by="Accuracy", ascending=False)

    st.success("🎉 Pelatihan Seluruh Model Selesai!")
    
    st.markdown("<h4 class='sub-title'>Tabel Komparasi Performa Model</h4>", unsafe_allow_html=True)
    st.dataframe(hasil_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    best_model_name = hasil_df.iloc[0]["Model"]
    best_acc = hasil_df.iloc[0]["Accuracy"]
    
    st.markdown(f"""
    <div class='badge-success' style='margin-bottom: 20px;'>
        🏆 Model Terbaik: <b>{best_model_name}</b> dengan Akurasi <b>{best_acc:.2%}</b>
    </div>
    """, unsafe_allow_html=True)

    # Simpan Artefak Model
    joblib.dump(trained_models, "Semua_Model.pkl")
    joblib.dump(scaler, "Scaler.pkl")
    joblib.dump(target_encoder, "Target_Encoder.pkl")
    joblib.dump(feature_encoders, "Feature_Encoders.pkl")
    joblib.dump(feature_columns, "Feature_Columns.pkl")
    st.info("📦 Semua komponen artefak model berhasil diekspor (.pkl).")

    # Visualisasi Komparasi Model
    st.markdown("<h4 class='sub-title'>Grafik Perbandingan Akurasi</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(hasil_df["Model"], hasil_df["Accuracy"], color='#ef5350')
    ax.set_ylabel("Accuracy Score")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=15, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    st.pyplot(fig)

# --- PREDIKSI MANUAL ---
elif menu == "🔍 Prediksi Manual Pasien":
    st.header("🔍 Prediksi Manual Pasien")
    
    required_files = [
        "Semua_Model.pkl",
        "Scaler.pkl",
        "Target_Encoder.pkl",
        "Feature_Encoders.pkl",
        "Feature_Columns.pkl"
    ]
    
    if not all(os.path.exists(file) for file in required_files):
        st.error("⚠️ Silakan jalankan menu **Model Training & Comparison** terlebih dahulu agar model siap digunakan.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    # Ambil model yang sudah dipilih dari Sidebar
    model_name = selected_model_name if selected_model_name else list(models.keys())[0]
    model = models[model_name]

    st.markdown(f"**Model Aktif:** `<span style='color: #e53935; font-size: 1.1rem;'>{model_name}</span>`", unsafe_allow_html=True)
    st.write("---")

    tab_form, tab_upload = st.tabs(["📝 Form Input Medis", "📁 Quick Upload CSV"])

    # ------------------------------------------------------
    # TAB 1: INPUT FORM MANDIRI
    # ------------------------------------------------------
    with tab_form:
        st.subheader("Isi Indikator Medis Pasien")
        input_data = {}
        
        label_mapping = {
            "age": "Usia / Umur Pasien",
            "sex": "Jenis Kelamin (sex)",
            "cp": "Tipe Nyeri Dada / Chest Pain (cp)",
            "trestbps": "Tekanan Darah Istirahat (trestbps) [mmHg]",
            "chol": "Kolesterol Serum (chol) [mg/dl]",
            "fbs": "Gula Darah Puasa > 120 mg/dl (fbs)",
            "restecg": "Hasil Elektrokardiogram Istirahat (restecg)",
            "thalach": "Detak Jantung Maksimum (thalach)",
            "exang": "Angina Induksi Olahraga (exang)",
            "oldpeak": "Depresi ST Induksi Olahraga (oldpeak)",
            "slope": "Kemiringan Segmen ST Peak (slope)",
            "ca": "Jumlah Pembuluh Darah Utama Terwarnai (ca)",
            "thal": "Status Thalium Stress Test (thal)"
        }

        option_mapping = {
            "Male": "Laki-laki (Male)",
            "Female": "Perempuan (Female)",
            "typical angina": "Typical Angina",
            "atypical angina": "Atypical Angina",
            "non-anginal": "Non-anginal Pain",
            "asymptomatic": "Asymptomatic",
            "normal": "Normal",
            "lv hypertrophy": "LV Hypertrophy",
            "st-t abnormality": "ST-T Wave Abnormality",
            "Yes": "Ya (1)",
            "No": "Tidak (0)",
            "upsloping": "Upsloping",
            "flat": "Flat",
            "downsloping": "Downsloping",
            "fixed defect": "Fixed Defect",
            "reversable defect": "Reversable Defect"
        }

        reverse_option_mapping = {v: k for k, v in option_mapping.items()}
        
        with st.form("form_prediksi_manual"):
            col1, col2 = st.columns(2)
            ui_inputs = {}
            
            for idx, col_name in enumerate(feature_columns):
                form_col = col1 if idx % 2 == 0 else col2
                display_label = label_mapping.get(col_name, f"Masukkan {col_name}")
                
                if col_name in feature_encoders:
                    labels_kategori = list(feature_encoders[col_name].classes_)
                    translated_options = [option_mapping.get(opt, opt) for opt in labels_kategori]
                    
                    ui_inputs[col_name] = form_col.selectbox(
                        display_label, 
                        options=translated_options,
                        key=f"ui_{col_name}"
                    )
                else:
                    min_val = float(df[col_name].min()) if col_name in df.columns else 0.0
                    max_val = float(df[col_name].max()) if col_name in df.columns else 500.0
                    mean_val = float(df[col_name].mean()) if col_name in df.columns else 10.0
                    
                    ui_inputs[col_name] = form_col.number_input(
                        display_label, 
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"ui_{col_name}"
                    )
            
            submitted = st.form_submit_button("🔮 Lakukan Prediksi Pasien", use_container_width=True)

        if submitted:
            for col_name in feature_columns:
                val = ui_inputs[col_name]
                if col_name in feature_encoders:
                    input_data[col_name] = reverse_option_mapping.get(val, val)
                else:
                    input_data[col_name] = val

            input_df = pd.DataFrame([input_data])
            
            for col, encoder in feature_encoders.items():
                try:
                    val_str = str(input_df.at[0, col]).strip()
                    known_classes = list(encoder.classes_)
                    matched_class = None
                    
                    for c in known_classes:
                        if str(c).lower() == val_str.lower():
                            matched_class = c
                            break
                    
                    if matched_class is not None:
                        input_df[col] = encoder.transform([matched_class])
                    else:
                        input_df[col] = encoder.transform([known_classes[0]])
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat encoding fitur '{col}': {e}")
                    st.stop()
            
            input_df = input_df[feature_columns]
            input_scaled = scaler.transform(input_df)
            
            prediksi_angka = model.predict(input_scaled)
            hasil_label = target_encoder.inverse_transform(prediksi_angka)[0]
            
            st.markdown("<h4 class='sub-title'>Hasil Prediksi Diagnosa</h4>", unsafe_allow_html=True)
            
            if "Sakit" in hasil_label or hasil_label == "1":
                st.markdown(f"""
                <div class='badge-danger'>
                    ⚠️ Hasil Prediksi ({model_name}): Pasien Terindikasi <b>{hasil_label.upper()}</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='badge-success'>
                    ✅ Hasil Prediksi ({model_name}): Kondisi Pasien <b>{hasil_label.upper()}</b>
                </div>
                """, unsafe_allow_html=True)
            
            if hasattr(model, "predict_proba"):
                probabilitas = model.predict_proba(input_scaled)[0]
                st.write("")
                st.write("**Probabilitas Keyakinan Model:**")
                prob_df = pd.DataFrame({
                    "Status Diagnosa": target_encoder.classes_,
                    "Keyakinan (Persentase)": [f"{p*100:.2f}%" for p in probabilitas]
                })
                st.dataframe(prob_df, use_container_width=True)

    # ------------------------------------------------------
    # TAB 2: UPLOAD FILE CSV
    # ------------------------------------------------------
    with tab_upload:
        st.subheader("Predict via CSV Upload")
        st.info("Sistem akan otomatis mendeteksi dan menyesuaikan kolom dataset pasien yang Anda unggah.")
        
        uploaded_file = st.file_uploader("Upload Dataset Pasien (CSV)", type=["csv"], key="manual_upload_csv")
        
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            st.markdown("<h4 class='sub-title'>Preview Data Upload</h4>", unsafe_allow_html=True)
            st.dataframe(user_data.head(), use_container_width=True)
            
            aligned_data = pd.DataFrame(index=user_data.index)

            for col in feature_columns:
                if col in user_data.columns:
                    aligned_data[col] = user_data[col].copy()
                else:
                    matched_col = None
                    simplified_target = col.lower().replace("_", "").replace(" ", "")
                    for user_col in user_data.columns:
                        if user_col.lower().replace("_", "").replace(" ", "") == simplified_target:
                            matched_col = user_col
                            break
                    
                    if matched_col:
                        aligned_data[col] = user_data[matched_col].copy()
                    else:
                        if col in feature_encoders:
                            aligned_data[col] = feature_encoders[col].classes_[0]
                        else:
                            aligned_data[col] = float(df[col].mean()) if col in df.columns else 0.0

            for col in feature_columns:
                if aligned_data[col].isnull().any():
                    if col in feature_encoders:
                        aligned_data[col] = aligned_data[col].fillna(feature_encoders[col].classes_[0])
                    else:
                        aligned_data[col] = aligned_data[col].fillna(float(df[col].mean()) if col in df.columns else 0.0)

            for col, encoder in feature_encoders.items():
                known_classes = set(encoder.classes_)
                default_class = encoder.classes_[0]
                
                aligned_data[col] = aligned_data[col].astype(str).apply(
                    lambda x: x if x in known_classes else default_class
                )
                aligned_data[col] = encoder.transform(aligned_data[col])

            aligned_data = aligned_data[feature_columns]
            aligned_scaled = scaler.transform(aligned_data)
            csv_preds = model.predict(aligned_scaled)
            csv_labels = target_encoder.inverse_transform(csv_preds)

            final_result = user_data.copy()
            final_result[f"Hasil Prediksi ({model_name})"] = csv_labels

            st.markdown("<h4 class='sub-title'>Hasil Prediksi Dataset Upload</h4>", unsafe_allow_html=True)
            st.dataframe(final_result, use_container_width=True)

            csv_output = final_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"⬇️ Download Hasil Prediksi {model_name} (CSV)",
                data=csv_output,
                file_name=f"hasil_prediksi_jantung_{model_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

# --- PREDIKSI DATASET UPLOAD ---
elif menu == "📁 Prediksi Massal (CSV Upload)":
    st.header("📁 Upload Dataset Pasien Baru & Analisis Model")
    st.info("💡 **Fitur Deteksi Kolom Otomatis:** Unggah berkas CSV pasien. Sistem akan otomatis menyesuaikan struktur kolom dengan dataset awal agar terhindar dari error.")

    uploaded_prediction = st.file_uploader("Upload Dataset Baru (CSV)", type=["csv"], key="dataset_prediction")
    if uploaded_prediction is None:
        st.warning("Silakan unggah berkas CSV terlebih dahulu.")
        st.stop()

    new_data = pd.read_csv(uploaded_prediction)
    st.markdown("<h4 class='sub-title'>Dataset Asli yang Di-upload</h4>", unsafe_allow_html=True)
    st.dataframe(new_data, use_container_width=True)

    required_files = ["Semua_Model.pkl", "Scaler.pkl", "Target_Encoder.pkl", "Feature_Encoders.pkl", "Feature_Columns.pkl"]
    if not all(os.path.exists(file) for file in required_files):
        st.error("⚠️ Model belum tersedia. Jalankan menu Model Training terlebih dahulu.")
        st.stop()

    models = joblib.load("Semua_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    target_encoder = joblib.load("Target_Encoder.pkl")
    feature_encoders = joblib.load("Feature_Encoders.pkl")
    feature_columns = joblib.load("Feature_Columns.pkl")

    has_target = False
    actual_labels = None
    target_candidates = ["num", "target", "heart_disease", "Status", "Diagnosa"]
    
    for candidate in target_candidates:
        if candidate in new_data.columns:
            has_target = True
            actual_labels = new_data[candidate].copy()
            break

    prediction_data = pd.DataFrame(index=new_data.index)

    for col in feature_columns:
        if col in new_data.columns:
            prediction_data[col] = new_data[col].copy()
        else:
            matched_col = None
            simplified_target = col.lower().replace("_", "").replace(" ", "")
            for user_col in new_data.columns:
                if user_col.lower().replace("_", "").replace(" ", "") == simplified_target:
                    matched_col = user_col
                    break
            
            if matched_col:
                prediction_data[col] = new_data[matched_col].copy()
            else:
                if col in feature_encoders:
                    default_cat = feature_encoders[col].classes_[0]
                    prediction_data[col] = default_cat
                else:
                    prediction_data[col] = 0.0

    for col in feature_columns:
        if prediction_data[col].isnull().any():
            if col in feature_encoders:
                prediction_data[col] = prediction_data[col].fillna(feature_encoders[col].classes_[0])
            else:
                prediction_data[col] = prediction_data[col].fillna(0.0)

    for col, encoder in feature_encoders.items():
        known_classes = set(encoder.classes_)
        default_class = encoder.classes_[0]
        
        prediction_data[col] = prediction_data[col].astype(str).apply(
            lambda x: x if x in known_classes else default_class
        )
        prediction_data[col] = encoder.transform(prediction_data[col])

    prediction_data = prediction_data[feature_columns]
    prediction_scaled = scaler.transform(prediction_data)
    
    st.write("---")
    perbandingan_list = []
    result_data_all = new_data.copy()

    for name, model_obj in models.items():
        preds = model_obj.predict(prediction_scaled)
        pred_labels = target_encoder.inverse_transform(preds)
        result_data_all[f"Prediksi ({name})"] = pred_labels
        
        if has_target:
            if actual_labels.dtype != 'object':
                act_str = actual_labels.apply(lambda x: "Sakit Jantung" if x > 0 else "Sehat").astype(str)
            else:
                act_str = actual_labels.astype(str)

            acc = accuracy_score(act_str, pred_labels.astype(str))
            pre = precision_score(act_str, pred_labels.astype(str), average="weighted", zero_division=0)
            rec = recall_score(act_str, pred_labels.astype(str), average="weighted", zero_division=0)
            f1 = f1_score(act_str, pred_labels.astype(str), average="weighted", zero_division=0)
            perbandingan_list.append([name, acc, pre, rec, f1])

    if has_target:
        st.markdown("<h4 class='sub-title'>📈 Perbandingan Performa Semua Model</h4>", unsafe_allow_html=True)
        df_compare = pd.DataFrame(
            perbandingan_list, 
            columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
        ).sort_values(by="Accuracy", ascending=False)

        st.dataframe(df_compare.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
        }), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(df_compare["Model"], df_compare["Accuracy"], color='#d32f2f')
        ax.set_ylabel("Accuracy Score")
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=15, fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height*100:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
        st.pyplot(fig)

    st.markdown("<h4 class='sub-title'>📋 Hasil Prediksi Lengkap</h4>", unsafe_allow_html=True)
    st.dataframe(result_data_all, use_container_width=True)

    csv_all = result_data_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Semua Hasil Prediksi (CSV)",
        data=csv_all,
        file_name="hasil_prediksi_semua_model.csv",
        mime="text/csv"
    )