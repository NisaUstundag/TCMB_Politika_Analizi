# 📈 NLP Tabanlı Finansal Erken Uyarı ve Karar Destek Sistemi

Bu proje, Türkiye Cumhuriyet Merkez Bankası (TCMB) tarafından yayınlanan Para Politikası Kurulu (PPK) duyuru metinlerini **Doğal Dil İşleme (NLP)** ve **Makine Öğrenmesi (ML)** yöntemleriyle analiz ederek, enflasyon riskini (Düşük/Orta/Yüksek) öngörmeyi amaçlayan bir FinTech çözümüdür.

---

##  Projenin Amacı ve Temel Problemi
Ekonomi dünyasında Merkez Bankası'nın kullandığı dil, piyasa beklentilerini şekillendiren en kritik unsurdur. Ancak bu metinlerin manuel analizi;
1. **Sübjektiftir:** Kişiden kişiye değişen yorumlara açıktır, yanılma payı yüksektir.
2. **Zaman Alıcıdır:** Karar anında hızlı ve algoritmik aksiyon almayı zorlaştırır.

**Bu sistem**, metinlerdeki "Şahin" (sert/mücadeleci) ve "Güvercin" (ılımlı/destekleyici) tonu matematiksel bir skora dökerek, veriye dayalı, objektif ve saniyeler içinde çalışan bir **Erken Uyarı Sistemi** sunmaktadır.

---

##  Teknik Metodoloji
Proje, ham veriden nihai risk öngörüsüne kadar uçtan uca bir makine öğrenmesi boru hattı (pipeline) üzerine kurgulanmıştır.

### 1. Veri Kaynağı ve Güvenilirlik
* **Resmi Kaynak:** 2006-2025 yıllarını kapsayan PPK metinleri ve enflasyon verileri doğrudan **TCMB EVDS** (Elektronik Veri Dağıtım Sistemi) üzerinden çekilmiştir.
* **Veri İşleme:** Model, tamamen gerçek ve resmi devlet verileriyle eğitilerek doğruluğu maksimize edilmiştir.

### 2. Algoritma Seçimi (Model Yarıştırma)
Proje kapsamında 5 farklı denetimli öğrenme algoritması aynı veri seti üzerinde performans testine tabi tutulmuştur:
* **Naive Bayes**
* **Decision Tree**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machines (SVM)**
* **Random Forest**

Yapılan analizler ve çapraz doğrulama sonuçlarına göre en kararlı performansı sergileyen **Random Forest**, ana tahmin motoru olarak seçilmiştir.

### 3. Açıklanabilir Yapay Zeka (XAI)
Modelimiz bir "Kara Kutu" (Black Box) değildir. Ürettiği risk tahmininin gerekçesini (kelime ağırlıkları, tonlama skoru ve geçmiş veri örüntüleri) şeffaf bir şekilde raporlayarak **Karar Destek Sistemi** görevini üstlenir.

---

##  Öne Çıkan Özellikler
* **Canlı Çıkarım (Inference):** Yeni yayınlanan metinleri anlık olarak analiz edebilme yeteneği.
* **Dinamik Sentez:** Kelime sinyalleri ile makine öğrenmesi tahminini çapraz kontrol ederek tutarlı raporlama sunma.
* **İnteraktif Dashboard:** Streamlit tabanlı, kurumsal standartlarda kullanıcı dostu arayüz.

---

## 👩‍💻 Geliştirici
**Nisa Üstündağ** *Yazılım Mühendisliği Bölümü | Malatya Turgut Özal Üniversitesi*

---

##  Kurulum ve Çalıştırma
Sistem Python tabanlı olup Streamlit kütüphanesi üzerinden çalışmaktadır. Projeyi yerel ortamınızda ayağa kaldırmak için aşağıdaki adımları takip edebilirsiniz:

```bash
# Gerekli kütüphanelerin yüklenmesi
pip install streamlit pandas scikit-learn seaborn matplotlib

# Uygulamanın başlatılması
streamlit run main.py
