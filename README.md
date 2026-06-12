# Geo Sentiment Padang Insight

Dashboard analisis geo-sentimen untuk ulasan warung makan Padang di Kabupaten Banyumas.

Proyek ini memadukan pemrosesan teks, klasifikasi sentimen, dan visualisasi spasial
untuk membaca persepsi pelanggan secara lebih terstruktur.

## Ringkasan

- Sumber data: ulasan Google Maps
- Fokus analisis: warung makan Padang di Kabupaten Banyumas
- Pendekatan: preprocessing teks, prediksi sentimen, dan pemetaan lokasi
- Antarmuka utama: dashboard interaktif berbasis Streamlit

## Fitur Utama

- Filter data berdasarkan tanggal, minggu, atau bulan
- Peta interaktif sentimen per lokasi
- Ringkasan metrik ulasan positif, negatif, dan total ulasan
- Uji ulasan baru untuk melihat hasil prediksi model

## Teknologi

- Python
- Streamlit
- Pandas
- Scikit-learn
- Sastrawi
- Folium
- streamlit-folium

## Instalasi

1. Buat virtual environment jika diperlukan.
2. Install dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

Pastikan struktur data dan model tersedia sesuai kebutuhan aplikasi, lalu jalankan:

```bash
streamlit run app.py
```

## Catatan

- `data/GeoSentiment_Cleaned_Labeled.csv` diperlukan untuk pembacaan data.
- `models/best_sentiment_model.pkl` dan `models/tfidf_vectorizer.pkl` diperlukan untuk inferensi.

## License

Proyek ini menggunakan lisensi MIT. Lihat berkas [LICENSE](/mnt/c/EngineeringProjects/GeoSentimenPadang/GeoSentimentPadangInsight/LICENSE) untuk detail lengkap.
