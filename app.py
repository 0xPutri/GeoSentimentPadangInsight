import pandas as pd
import streamlit as st
import re
import joblib
import folium
from streamlit_folium import st_folium
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(
    page_title="Dashboard Geo-Sentimen Banyumas",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%);
        border-right: 1px solid rgba(15, 23, 42, 0.08);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        padding-top: 0.25rem;
    }
    section[data-testid="stSidebar"] .sidebar-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.15rem;
    }
    section[data-testid="stSidebar"] .sidebar-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    section[data-testid="stSidebar"] .sidebar-note {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.6);
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-weight: 700;
        color: #0f172a;
    }
    section[data-testid="stSidebar"] [data-baseweb="radio"] {
        padding-top: 0.15rem;
        padding-bottom: 0.2rem;
    }
    section[data-testid="stSidebar"] [data-testid="stDateInput"],
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] {
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_ml_components():
    """
    Memuat komponen model yang dibutuhkan aplikasi.

    Fungsi ini mengambil model, vektor TF-IDF, dan stemmer agar proses
    prediksi dapat berjalan dengan stabil.

    Returns:
        tuple: Berisi model, vectorizer, dan stemmer.
    """
    try:
        model = joblib.load("models/best_sentiment_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        stemmer = StemmerFactory().create_stemmer()
        
        return model, vectorizer, stemmer
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file .pkl berada di folder 'models/'. Error: {e}")
        return None, None, None

@st.cache_data
def load_and_process_data(_model, _vectorizer):
    """
    Memuat dan menyiapkan data ulasan untuk analisis.

    Fungsi ini membersihkan data, memproses waktu, lalu menambahkan hasil
    prediksi model ke dalam dataframe.

    Args:
        _model: Model klasifikasi yang dipakai untuk prediksi.
        _vectorizer: Vectorizer TF-IDF untuk mengubah teks menjadi fitur.

    Returns:
        pd.DataFrame: Data yang sudah diproses dan diberi prediksi.
    """
    try:
        df = pd.read_csv("data/GeoSentiment_Cleaned_Labeled.csv")
        df = df.dropna(subset=["ulasan_bersih"])

        df["waktu_dt"] = pd.to_datetime(df["waktu"], format="%d/%m/%Y, %H.%M.%S", errors="coerce")
        df = df.dropna(subset=["waktu_dt"])
        df["tanggal"] = df["waktu_dt"].dt.date
        df["bulan_tahun"] = df["waktu_dt"].dt.strftime("%B %Y")
        df["minggu_mulai"] = df["waktu_dt"].dt.to_period("W-SUN").apply(lambda p: p.start_time.date())
        df["minggu_selesai"] = df["waktu_dt"].dt.to_period("W-SUN").apply(lambda p: p.end_time.date())

        X_tfidf = _vectorizer.transform(df["ulasan_bersih"])
        df["prediksi_model"] = _model.predict(X_tfidf)

        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset. Error: {e}")
        return pd.DataFrame()
    
best_model, tfidf_vectorizer, sastrawi_stemmer = load_ml_components()
if best_model is None:
    st.stop()

df_main = load_and_process_data(best_model, tfidf_vectorizer)
if df_main.empty:
    st.error("Dataset tidak memiliki baris valid setelah parsing waktu.")
    st.stop()

st.sidebar.markdown('<div class="sidebar-kicker">Kontrol Dashboard</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-title">Filter data sentimen</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<div class="sidebar-note">Gunakan filter waktu untuk membaca tren sentimen.</div>',
    unsafe_allow_html=True
)

with st.sidebar.expander("Rentang Waktu", expanded=True):
    pilihan_waktu = st.radio(
        "Pilih mode filter",
        ("Semua Waktu", "Tanggal Tertentu", "Minggu Tertentu", "Bulan Tertentu"),
        label_visibility="collapsed"
    )

    if pilihan_waktu == "Tanggal Tertentu":
        tanggal_min = df_main["tanggal"].min()
        tanggal_max = df_main["tanggal"].max()
        tanggal_dipilih = st.date_input(
            "Pilih tanggal",
            value=tanggal_max,
            min_value=tanggal_min,
            max_value=tanggal_max
        )
        df_filtered = df_main[df_main["tanggal"] == tanggal_dipilih]
    elif pilihan_waktu == "Minggu Tertentu":
        daftar_minggu = (
            df_main[["minggu_mulai", "minggu_selesai"]]
            .drop_duplicates()
            .sort_values(["minggu_mulai", "minggu_selesai"])
            .reset_index(drop=True)
        )
        opsi_minggu = [
            f"{row.minggu_mulai.strftime('%d %b %Y')} - {row.minggu_selesai.strftime('%d %b %Y')}"
            for row in daftar_minggu.itertuples(index=False)
        ]
        minggu_dipilih = st.selectbox("Pilih minggu", opsi_minggu)
        minggu_index = opsi_minggu.index(minggu_dipilih)
        minggu_row = daftar_minggu.iloc[minggu_index]
        df_filtered = df_main[
            (df_main["minggu_mulai"] == minggu_row["minggu_mulai"]) &
            (df_main["minggu_selesai"] == minggu_row["minggu_selesai"])
        ]
    elif pilihan_waktu == "Bulan Tertentu":
        daftar_bulan = df_main["bulan_tahun"].dropna().unique().tolist()
        bulan_dipilih = st.selectbox("Pilih bulan", daftar_bulan)
        df_filtered = df_main[df_main["bulan_tahun"] == bulan_dipilih]
    else:
        df_filtered = df_main.copy()

st.title("Analisis Geo-Sentimen Warung Padang (Banyumas)")
st.markdown("Dashboard ini menggunakan model **Random Forest** untuk memprediksi sentimen ulasan secara spasial.")

tab1, tab2 = st.tabs(["Peta Geo-Sentimen (Spasial)", "Uji Ulasan Baru (Inferensi)"])

with tab1:
    if pilihan_waktu == "Semua Waktu":
        judul_filter = "Semua Waktu"
    elif pilihan_waktu == "Tanggal Tertentu":
        judul_filter = f"Tanggal {tanggal_dipilih.strftime('%d %b %Y')}"
    elif pilihan_waktu == "Minggu Tertentu":
        judul_filter = minggu_dipilih
    else:
        judul_filter = bulan_dipilih

    st.subheader(f"Peta Sentimen: {judul_filter}")
    
    if df_filtered.empty:
        st.warning("Tidak ada ulasan yang ditemukan pada rentang waktu yang dipilih.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ulasan", len(df_filtered))
        col2.metric("Dominan Positif", len(df_filtered[df_filtered["prediksi_model"] == "Positif"]))
        col3.metric("Dominan Negatif", len(df_filtered[df_filtered["prediksi_model"] == "Negatif"]))

        agg_df = df_filtered.groupby(["nama_tempat", "latitude", "longitude"]).agg(
            total_ulasan=("prediksi_model", "count"),
            positif=("prediksi_model", lambda x: (x == "Positif").sum()),
            negatif=("prediksi_model", lambda x: (x == "Negatif").sum()),
            netral=("prediksi_model", lambda x: (x == "Netral").sum())
        ).reset_index()

        def get_dominant(row):
            if row["positif"] > row["negatif"] and row["positif"] > row["netral"]:
                return "Positif"
            elif row["negatif"] > row["positif"] and row["negatif"] > row["netral"]:
                return "Negatif"
            return "Netral"
            
        agg_df["sentimen_dominan"] = agg_df.apply(get_dominant, axis=1)

        center_lat = agg_df["latitude"].mean()
        center_lon = agg_df["longitude"].mean()
        peta_sentimen = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

        for _, row in agg_df.iterrows():
            if row["sentimen_dominan"] == "Positif":
                warna = "green"
            elif row["sentimen_dominan"] == "Negatif":
                warna = "red"
            else:
                warna = "gray"
                
            popup_html = f"""
            <div style="width: 220px; font-family: sans-serif;">
                <h4 style="margin-top:0;">{row['nama_tempat']}</h4>
                <p style="margin:2px;"><b>Total Ulasan:</b> {row['total_ulasan']}</p>
                <p style="margin:2px; color: green;"><b>Positif:</b> {row['positif']}</p>
                <p style="margin:2px; color: red;"><b>Negatif:</b> {row['negatif']}</p>
                <hr style="margin: 5px 0;">
                <p style="margin:2px;"><b>Sentimen Dominan:</b> {row['sentimen_dominan']}</p>
            </div>
            """
            
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row["nama_tempat"],
                icon=folium.Icon(color=warna, icon="info-sign")
            ).add_to(peta_sentimen)

        st_folium(peta_sentimen, width=900, height=500)

with tab2:
    st.subheader("Simulasi Inferensi Model")
    st.markdown("Masukkan ulasan baru untuk melihat bagaimana model memprediksi sentimen secara otomatis.")
    
    teks_input = st.text_area("Teks Ulasan Baru:", height=100, placeholder="Contoh: Tempatnya kurang bersih dan pelayanannya sangat lama.")
    
    if st.button("Prediksi Sentimen", type="primary"):
        if teks_input.strip() == "":
            st.warning("Harap masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses teks dan melakukan prediksi..."):
                teks_bersih = str(teks_input).lower()
                teks_bersih = re.sub(r'[^a-z\s]', ' ', teks_bersih)
                teks_bersih = re.sub(r'(.)\1{2,}', r'\1', teks_bersih)
                
                teks_stem = sastrawi_stemmer.stem(teks_bersih)
                
                vektor_teks = tfidf_vectorizer.transform([teks_stem])
                
                hasil_prediksi = best_model.predict(vektor_teks)[0]
                
                st.success("Proses inferensi selesai!")
                st.markdown("### Hasil Analisis")
                
                if hasil_prediksi == "Positif":
                    warna_teks = "green"
                    deskripsi = "Pelanggan menunjukkan kepuasan. Poin ini dapat menjadi nilai jual warung."
                elif hasil_prediksi == "Negatif":
                    warna_teks = "red"
                    deskripsi = "Terdapat indikasi keluhan. Ulasan ini perlu dievaluasi oleh manajemen warung."
                else:
                    warna_teks = "gray"
                    deskripsi = "Ulasan bersifat netral atau tidak menunjukkan emosi yang kuat."
                    
                st.markdown(f"**Prediksi Sentimen:** <span style='color:{warna_teks}; font-size:18px; font-weight:bold;'>{hasil_prediksi.upper()}</span>", unsafe_allow_html=True)
                st.markdown(f"**Saran Tindakan:** {deskripsi}")
                
                with st.expander("Lihat Detail Pemrosesan Teks (Explainability)"):
                    st.write("**Teks Asli:**", teks_input)
                    st.write("**Teks Setelah Preprocessing (Stemming):**", teks_stem)
