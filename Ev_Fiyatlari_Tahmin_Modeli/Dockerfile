# 1. Python yüklü, hafif bir işletim sistemi seçiyoruz
FROM python:3.11-slim

# 2. Konteyner içinde çalışacağımız klasörü belirliyoruz
WORKDIR /app

# 3. Önce pip'i güncelliyoruz ki indirme işlemlerini daha iyi yönetsin
RUN pip install --upgrade pip

# 4. PyTorch'un sadece CPU versiyonunu (çok daha küçük boyutlu) ve diğer kütüphaneleri kuruyoruz.
# Eklediğimiz --default-timeout=1000 parametresi bağlantının hemen kopmasını engeller.
RUN pip install --no-cache-dir --default-timeout=1000 \
    pandas numpy scikit-learn matplotlib seaborn && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 5. Kodları ve veri setini kopyalıyoruz
COPY house_prices.py .
COPY house_prices.csv .

# 6. Uygulamayı çalıştırıyoruz
CMD ["python3", "house_prices.py"]