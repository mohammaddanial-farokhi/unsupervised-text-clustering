# 📌 JEOPARDY Question Clustering with Embeddings + UMAP + HDBSCAN  
### A Full Pipeline for Text Embedding, Dimensionality Reduction & Unsupervised Topic Discovery

این پروژه یک سیستم کامل برای **استخراج بردارهای معنایی از سوالات Jeopardy**، کاهش ابعاد با **UMAP** و خوشه‌بندی **HDBSCAN** ارائه می‌دهد.  
نتیجهٔ نهایی شامل **برچسب خوشه‌ها، امتیاز سیلوئت، و استخراج کلمات کلیدی هر خوشه** است.

---

## 🏷️ Badges
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.19-blue?logo=python" />
  <img src="https://img.shields.io/badge/SentenceTransformers-all--mpnet--base--v2-purple?logo=pytorch" />
  <img src="https://img.shields.io/badge/UMAP-0.5.5-orange" />
  <img src="https://img.shields.io/badge/HDBSCAN-0.8.33-green" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
</p>

---

## 🧩 Pipeline Sections

### 🔹 **Embedding Generator**
- پاک‌سازی متن  
- تولید Embedding با مدل: **all-mpnet-base-v2**  
- نرمال‌سازی بردارها  
- ذخیره Embedding در فایل `embeddings.npy`

---

### 🔹 **Dimensionality Reduction (UMAP)**
- کاهش ابعاد به 10 مؤلفه  
- metric = cosine  
- ذخیره مدل UMAP در `umap_model.pkl`

---

### 🔹 **Clustering (HDBSCAN)**
- خوشه‌بندی بدون نیاز به تعیین تعداد خوشه  
- حذف نویز با label = -1  
- ذخیره مدل HDBSCAN در `hdbscan_model.pkl`  
- ذخیره برچسب‌ها در `labels.npy`

---

### 🔹 **Evaluation**
- محاسبه Silhouette Score  
- بررسی تعداد خوشه‌های معتبر  

---

### 🔹 **Keyword Extraction**
- استخراج 20 کلمه پرتکرار برای هر خوشه  
- ذخیره خروجی در: `cluster_top_words.csv`

---

## 📁 Dataset

لینک دیتاست اصلی : 

🔗  
https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions/data

فایل دیتاست باید با نام JEOPARDY_CSV.csv در پوشه اصلی پروژه قرار گیرد


دیتاست نهایی شامل تمام خوشه بندی ها به همراه 20 کلمه پرتکرار هر خوشه همراه فایل اصلی کد اپلود شده است 
