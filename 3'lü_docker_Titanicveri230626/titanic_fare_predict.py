import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_fare_prediction():
    print("--- Titanic Bilet Fiyatı Tahmin Modeli Başlatılıyor ---")
    
    # 1. Veri Yükleme
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("[+] Veri seti internet üzerinden indiriliyor...")
    df = pd.read_csv(url)
    
    # 2. Veri Temizleme ve Ön İşleme
    # Bilet fiyatını (Fare) etkileyen özellikleri seçelim
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    df = df[features + ['Fare']].dropna()
    
    # Kategorik verileri sayısal değerlere dönüştürme
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    X = df[features]
    y = df['Fare']
    
    # Verileri standartlaştırma (Ölçeklendirme)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Eğitim ve test seti ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 3. Model Eğitimi
    print("[+] Linear Regression (Doğrusal Regresyon) algoritması eğitiliyor...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Tahmin ve Değerlendirme
    print("[+] Test verileri üzerinde bilet fiyatı (Fare) tahmin ediliyor...")
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"[+] Model Hata Oranı (MSE): {mse:.2f}")
    print(f"[+] Model Başarı Skoru (R2): {r2:.2f}")
    
    # 5. Sonuçları Kaydetme
    os.makedirs("/app/output", exist_ok=True)
    output_path = "/app/output/titanic_fare_report.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Titanic Bilet Fiyatı Tahmini Raporu (Linear Regression) ===\n\n")
        f.write(f"Ortalama Kare Hatası (MSE): {mse:.2f}\n")
        f.write(f"R-Kare Skoru (R2): {r2:.2f} (1'e ne kadar yakınsa o kadar iyi)\n\n")
        f.write("Özelliklerin (Feature) Ağırlıkları:\n")
        for i, col in enumerate(features):
            f.write(f"- {col}: {model.coef_[i]:.2f}\n")
            
    print(f"--- İŞLEM BİTTİ ---")
    print(f"Detaylı rapor {output_path} konumuna kaydedildi.")

if __name__ == "__main__":
    run_fare_prediction()
