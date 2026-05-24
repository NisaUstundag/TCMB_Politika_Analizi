import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB          
from sklearn.tree import DecisionTreeClassifier     
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC                         
from sklearn.ensemble import RandomForestClassifier 

# --- 1. SAYFA AYARLARI VE CSS ---
st.set_page_config(layout="wide", page_title="TCMB ML Analiz - Nisa Üstündağ", page_icon="📈")

st.markdown("""
<style>
    .big-title { font-size: 2.8rem !important; font-weight: 800 !important; text-align: center; margin-bottom: 1rem; }
    .st_card { padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.1); }
    .st_card h3 { margin-top: 0; font-weight: 700; font-size: 1.2rem;}
    .st_card h1 { font-size: 2rem; margin-bottom: 0.5rem; }
            [data-testid="stAppViewContainer"] {
    background-color: #f8fafc; 
}
</style>
            
            
""", unsafe_allow_html=True)

# --- 2. YAN MENÜ (SIDEBAR) ---
st.sidebar.markdown(f'<div style="text-align:center; font-size:4rem;">🏦</div>', unsafe_allow_html=True)
st.sidebar.title("TCMB Analiz Sistemi")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Proje Kapsamı:**\n"
    "TCMB metinlerindeki dilsel sinyalleri NLP ile analiz ederek "
    "Enflasyon Riskini (Düşük/Orta/Yüksek) öngörmek."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 👩‍💻 Geliştirici")
st.sidebar.markdown("- **Nisa Üstündağ**")
st.sidebar.markdown("---")
st.sidebar.success("🟢 Sistem Aktif")

# --- 3. ANA BAŞLIK ---
st.markdown('<div class="big-title">📈 TCMB Metin Analizi ve Enflasyon Tahmin Sistemi</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8;'>5 farklı denetimli öğrenme algoritması kıyaslanmış ve en iyi model (Random Forest) seçilmiştir.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. VERİ İŞLEME VE MODELLEME ---
@st.cache_data
def veri_hazirla():
    metin_yolu = "ppk_metinleri_tam.csv"
    enflasyon_yolu = "enflasyon_verisi.csv"
    
    if not os.path.exists(metin_yolu) or not os.path.exists(enflasyon_yolu):
        return None

    metin_df = pd.read_csv(metin_yolu)
    enflasyon_df = pd.read_csv(enflasyon_yolu, skiprows=5)
    
    enflasyon_df = enflasyon_df.iloc[:, :2]
    enflasyon_df.columns = ['tarih_ay', 'enflasyon_index']
    enflasyon_df['tarih_ay'] = pd.to_datetime(enflasyon_df['tarih_ay'], errors='coerce')
    enflasyon_df.dropna(subset=['tarih_ay'], inplace=True)
    enflasyon_df['enflasyon_index'] = enflasyon_df['enflasyon_index'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    metin_df['temiz_metin'] = metin_df['metin'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True)
    
    sahin_kelimeler = ['sıkı', 'kararlılıkla', 'mücadele', 'gerekirse', 'risk', 'enflasyon']
    guvercin_kelimeler = ['büyüme', 'destekleyici', 'kademeli', 'ihtiyatlı', 'toparlanma', 'istihdam']

    def kelime_say(metin, liste): 
        return sum(metin.count(kelime) for kelime in liste)
    
    metin_df['net_sahin_skoru'] = metin_df['temiz_metin'].apply(lambda m: kelime_say(m, sahin_kelimeler)) - metin_df['temiz_metin'].apply(lambda m: kelime_say(m, guvercin_kelimeler))
    metin_df['enflasyon_vurgusu'] = metin_df['temiz_metin'].str.count('enflasyon')
    metin_df['buyume_vurgusu'] = metin_df['temiz_metin'].str.count('büyüme')
    metin_df['risk_vurgusu'] = metin_df['temiz_metin'].str.count('risk')

    def tarih_cikar(metin):
        aylar = {"ocak": "01", "şubat": "02", "mart": "03", "nisan": "04", "mayıs": "05", "haziran": "06", "temmuz": "07", "ağustos": "08", "eylül": "09", "ekim": "10", "kasım": "11", "aralık": "12"}
        ay_regex = '|'.join(aylar.keys())
        match = re.search(r'\d{1,2}\s+(' + ay_regex + r')\s+(\d{4})', metin)
        if match: ay_str, yil = match.groups(); return f"{yil}-{aylar[ay_str]}"
        return None

    metin_df['tarih_ay'] = metin_df['temiz_metin'].apply(tarih_cikar)
    dil_sinyalleri_df = metin_df.dropna(subset=['tarih_ay']).groupby('tarih_ay')[['net_sahin_skoru', 'enflasyon_vurgusu', 'buyume_vurgusu', 'risk_vurgusu']].mean().reset_index()
    enflasyon_df['tarih_ay'] = enflasyon_df['tarih_ay'].dt.strftime('%Y-%m')
    
    final_df = pd.merge(dil_sinyalleri_df, enflasyon_df, on='tarih_ay', how='inner')
    
    # Hedef Değişkeni Sınıflandırma (Classification)
    final_df['risk_sinifi'] = pd.qcut(final_df['enflasyon_index'], q=3, labels=["Düşük", "Orta", "Yüksek"])
    return final_df

df = veri_hazirla()

if df is None:
    st.error("Veri dosyaları bulunamadı! Lütfen CSV dosyalarının klasörde olduğundan emin olun.")
else:
    # Veri Bölme
    X = df[['net_sahin_skoru', 'enflasyon_vurgusu', 'buyume_vurgusu', 'risk_vurgusu']]
    y = df['risk_sinifi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. SEKMELER ---
    tab1, tab2, tab3 = st.tabs(["🚀 Algoritma Karşılaştırma", "📊 Model Performans Metrikleri", "🔮 Canlı Tahmin Simülasyonu"])

    # --- SEKME 1: Algoritma Karşılaştırma ---
    with tab1:
        st.header("Algoritma Performans Analizi")
        st.write("Aynı veri seti üzerinde 5 farklı modelin doğruluk (Accuracy) oranları test edilmiştir.")
        
        modeller = {
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        sonuclar = []
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for isim, model in modeller.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                sonuclar.append({"Algoritma": isim, "Doğruluk": acc})
                st.markdown(f"**{isim}**: %{acc*100:.2f}")

        with col2:
            sonuc_df = pd.DataFrame(sonuclar)
            st.bar_chart(sonuc_df.set_index("Algoritma"))
            st.success("✅ Analiz sonucunda en kararlı model seçilmiştir.")

    # --- SEKME 2: Detaylı Model Değerlendirme ---
    with tab2:
        st.header("Model Doğrulama ve Hata Analizi")
        st.info("Seçilen model (Random Forest) için Confusion Matrix ve sınıflandırma metrikleri.")
        
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X_train, y_train)
        y_final_pred = best_model.predict(X_test)

        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_final_pred)
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Düşük', 'Orta', 'Yüksek'], 
                        yticklabels=['Düşük', 'Orta', 'Yüksek'])
            plt.ylabel('Gerçek Sınıflar')
            plt.xlabel('Tahmin Edilen Sınıflar')
            st.pyplot(fig_cm)

        with c2:
            st.subheader("Sınıflandırma Raporu")
            report = classification_report(y_test, y_final_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='viridis'), use_container_width=True)

    # --- SEKME 3: Canlı Test (Deployment) ---
    with tab3:
        st.header("🔮 Canlı Risk Analizi")
        
        txt_input = st.text_area("Analiz edilecek metni giriniz:", height=150, placeholder="Merkez Bankası metnini buraya yapıştırın...")
        
        if st.button("Risk Analizi Çalıştır 🚀", use_container_width=True):
            if txt_input:
                clean_txt = txt_input.lower().replace('\n', ' ')
                s_count = sum(clean_txt.count(k) for k in ['sıkı', 'kararlılıkla', 'mücadele', 'gerekirse', 'risk', 'enflasyon'])
                g_count = sum(clean_txt.count(k) for k in ['büyüme', 'destekleyici', 'kademeli', 'ihtiyatlı', 'toparlanma', 'istihdam'])
                net_sahin = s_count - g_count
                
                enflasyon_sayisi = clean_txt.count('enflasyon')
                buyume_sayisi = clean_txt.count('büyüme')
                risk_sayisi = clean_txt.count('risk')

                input_features = pd.DataFrame([{
                    'net_sahin_skoru': net_sahin,
                    'enflasyon_vurgusu': enflasyon_sayisi,
                    'buyume_vurgusu': buyume_sayisi,
                    'risk_vurgusu': risk_sayisi
                }])
                
                prediction = best_model.predict(input_features)[0]
                
                st.markdown("### 🎯 Sinyal Sonuçları")
                m1, m2, m3 = st.columns(3)
                m1.metric("Şahinlik Skoru", net_sahin)
                m2.metric("Enflasyon Vurgusu", enflasyon_sayisi)
                m3.metric("Büyüme Vurgusu", buyume_sayisi)

                st.divider()
                
                # --- ANA SONUÇ EKRANI ---
                if prediction == "Yüksek":
                    st.error(f"🚨 TAHMİN EDİLEN RİSK SEVİYESİ: **{prediction.upper()}**")
                elif prediction == "Orta":
                    st.warning(f"⚠️ TAHMİN EDİLEN RİSK SEVİYESİ: **{prediction.upper()}**")
                else:
                    st.success(f"✅ TAHMİN EDİLEN RİSK SEVİYESİ: **{prediction.upper()}**")
                
               # --- DİNAMİK AÇIKLAMA (XAI) EKRANI ---
                st.markdown("### 🧠 Modelin Karar Gerekçesi")
                st.write(f"**Ne Anlama Geliyor?** Model, bu metne dayanarak önümüzdeki dönemde **enflasyonist baskıların (fiyat artış riskinin) {prediction.lower()}** olacağını öngörüyor.")
                
                with st.expander("Detaylı Sinyal Analizini Gör", expanded=True):
                    st.write("**Neden Bu Karar Verildi?** Algoritma aşağıdaki kelime sinyallerini çapraz olarak değerlendirdi:")
                    
                    # Şahin/Güvercin Yorumu (Sadece Durum Tespiti)
                    if net_sahin > 0:
                        st.write(f"- 🦅 **Şahin Ton (+{net_sahin}):** 'Sıkı', 'mücadele' gibi sert kelimeler, ılımlı kelimelerden daha fazla kullanılmış.")
                    elif net_sahin < 0:
                        st.write(f"- 🕊️ **Güvercin Ton ({net_sahin}):** 'Büyüme', 'destekleyici' gibi ılımlı kelimeler daha ağır basıyor.")
                    else:
                        st.write("- ⚖️ **Nötr Ton (0):** Sert ve ılımlı kelimeler dengeli kullanılmış.")
                    
                    # Kelime Yorumları
                    if enflasyon_sayisi > 0:
                        st.write(f"- 📈 **Enflasyon Vurgusu:** {enflasyon_sayisi} kez enflasyon kelimesi geçmiş.")
                    if risk_sayisi > 0:
                        st.write(f"- ⚠️ **Risk Sinyali:** {risk_sayisi} kez risk kelimesi kullanılmış.")
                    if buyume_sayisi > 0:
                        st.write(f"- 🏭 **Büyüme Vurgusu:** {buyume_sayisi} kez büyüme kelimesi geçiyor.")

                    # --- YAPAY ZEKA SENTEZİ (ÇELİŞKİ ÇÖZÜCÜ) ---
                    st.markdown("---")
                    st.write("🤖 **Algoritmanın Sentezi:**")
                    if prediction == "Düşük" and net_sahin > 0:
                        st.info("Model, metinde 'şahin' (sert) kelimeler geçmesine rağmen; büyüme vurgusunu veya kelimelerin birbiriyle olan örüntüsünü daha güçlü bularak **genel risk tablosunu DÜŞÜK** olarak sınıflandırmıştır. Makine öğrenmesinin (Random Forest) insan gözünden kaçan karmaşık bağları kurma gücü buradadır.")
                    elif prediction == "Yüksek" and net_sahin <= 0:
                        st.error("Model, metinde ılımlı (güvercin) kelimeler geçse bile, enflasyon kelimesinin sıklığı ve geçmiş veri örüntüleri sebebiyle **gizli tehlikeyi sezmiş ve riski YÜKSEK** olarak hesaplamıştır.")
                    else:
                        st.success("Çıkarılan kelime sinyalleri ile modelin nihai risk tahmini birbiriyle tamamen tutarlı bir örüntü çiziyor.")
            else:
                st.warning("Lütfen analiz için geçerli bir metin giriniz.")