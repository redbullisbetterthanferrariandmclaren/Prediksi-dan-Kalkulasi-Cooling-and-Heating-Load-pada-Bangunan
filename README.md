# EnergyPredict AI
### Prediksi & Kalkulasi Heating Load · Cooling Load · Efisiensi Energi Bangunan

Sistem AI berbasis **Streamlit** untuk memprediksi kebutuhan energi bangunan berdasarkan parameter fisik desain, menggunakan **Random Forest Regressor** dengan pendekatan **MultiOutputRegressor**.

---

## Struktur Proyek

```
energy_efficiency_app/
│
├── app.py               ← Aplikasi Streamlit utama
├── train_model.py       ← Pipeline pelatihan model ML
├── helper.py            ← Fungsi-fungsi pembantu
├── requirements.txt     ← Dependensi Python
├── README.md            ← Dokumentasi ini
│
├── model/               ← Dibuat otomatis saat training
│   ├── energy_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── outputs/             ← Dibuat otomatis saat training
    ├── correlation_heatmap.png
    └── feature_importance.png
```

---

## Cara Menjalankan

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Latih Model (WAJIB dilakukan pertama kali)

```bash
python train_model.py
```

Output yang diharapkan:
```
  ENERGY EFFICIENCY PREDICTION — TRAINING PIPELINE

  CRISP-DM: DATA UNDERSTANDING
  ✔ Dataset berhasil dimuat: 768 baris × 10 kolom
  ✔ Heatmap disimpan → outputs/correlation_heatmap.png

  CRISP-DM: DATA PREPARATION
  Split 70:30 → Train: 537, Test: 231
  ✔ StandardScaler terpasang

  CRISP-DM: MODELING
  ✔ Pelatihan selesai

  CRISP-DM: EVALUATION
  [Heating Load]
    MAPE : 3.24%
    RMSE : 0.8821
    R²   : 0.9971

  [Cooling Load]
    MAPE : 3.87%
    RMSE : 0.9145
    R²   : 0.9965

  ✔ Model   → model/energy_model.pkl
  ✔ Scaler  → model/scaler.pkl

  TRAINING SELESAI — Jalankan: streamlit run app.py
```

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

---

## Fitur Aplikasi

| Fitur | Deskripsi |
|-------|-----------|
| **Input Desain** | Form lengkap 8 parameter fisik bangunan |
| **Validasi** | Pengecekan nilai negatif, kompaktness 0-1, area > 0 |
| **Prediksi AI** | Heating Load & Cooling Load (kWh/m²) |
| **Efficiency Score** | Skor 0-100 (Tinggi/Sedang/Rendah) |
| **Performance Map** | Peta kontur zona efisiensi energi interaktif |
| **Explainable AI** | Feature Importance + rekomendasi berbasis fitur dominan |
| **What-If Analysis** | Simulasi perubahan parameter dan dampak energi |
| **Estimasi Biaya** | Proyeksi konsumsi kWh & biaya PLN per bulan |

---

## Tech Stack

- **Python** 3.9+
- **Streamlit** — UI aplikasi web
- **scikit-learn** — Random Forest + MultiOutputRegressor + StandardScaler
- **Plotly** — Visualisasi interaktif (P-Map, bar chart)
- **Pandas / NumPy** — Manipulasi data
- **Joblib** — Penyimpanan model
- **Matplotlib / Seaborn** — EDA & heatmap

---

## Dataset

**UCI Energy Efficiency Dataset**
- URL: https://archive.ics.uci.edu/dataset/242/energy+efficiency
- 768 sampel bangunan
- 8 fitur input, 2 target output (Heating Load, Cooling Load)

---

## Tim

240087 - Aisyah Zaimatun Nisa
240021 - Nailatus Sahlah
240039 - Hana' Azzahro'

---

## CRISP-DM

1. **Business Understanding** — Masalah energi bangunan & tujuan sistem
2. **Data Understanding** — Eksplorasi dataset UCI
3. **Data Preparation** — Validasi, normalisasi, split 70:30
4. **Modeling** — Random Forest + MultiOutputRegressor
5. **Evaluation** — MAPE, RMSE, R² Score
6. **Interpretation & Recommendation** — Skor efisiensi + rekomendasi desain

---

---

## Ethical Analysis

### Transparency
EnergyPredict AI applies Explainable AI principles through Feature Importance visualization and feature-based recommendations. This approach helps users understand which building parameters have the greatest influence on Heating Load and Cooling Load predictions, making the decision-making process more transparent.

### Reliability
The model's reliability is evaluated using MAPE, RMSE, and R² Score metrics to measure prediction accuracy. Testing is performed on data separated from the training set to ensure the model can provide consistent and dependable results across different scenarios.

### Human Oversight
This system is designed as a decision-support tool and is not intended to replace professional analysis. Prediction results should be used as an initial reference and still require consideration from architects, engineers, or other qualified professionals during the building design process.

### Sustainability
By helping users estimate building energy requirements more accurately, EnergyPredict AI supports the development of energy-efficient building designs. The system can contribute to reducing energy consumption, lowering operational costs, and promoting environmentally sustainable development.

### Limitations and Responsible Use
The model is trained using the UCI Energy Efficiency Dataset, which contains 768 building samples. Therefore, prediction performance may decrease when applied to buildings with characteristics that differ significantly from those represented in the training data. Users are encouraged to understand these limitations and use the prediction results responsibly as supporting information rather than the sole basis for decision-making.