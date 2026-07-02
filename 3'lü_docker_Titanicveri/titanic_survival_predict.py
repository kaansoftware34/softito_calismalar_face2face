import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def run_survival_prediction():
    print("--- Titanic Survival Prediction Model Başlatılıyor ---")
    
    # 1. Veri Yükleme
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("[+] Veri seti internet üzerinden indiriliyor...")
    df = pd.read_csv(url)
    print(f"[+] Veri başarıyla yüklendi: {len(df)} kayıt bulundu.")
    
    # 2. Veri Temizleme ve Ön İşleme
    # Sadece anlamlı özellikleri (feature) seçelim
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[features + ['Survived']].dropna()
    
    # Kategorik verileri sayısal değerlere dönüştürme (Label Encoding)
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    X = df[features]
    y = df['Survived']
    
    # Eğitim ve test seti ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Eğitimi
    print("[+] Random Forest Classifier algoritması eğitiliyor...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Tahmin ve Değerlendirme
    print("[+] Test verileri üzerinde hayatta kalma durumu tahmin ediliyor...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print(f"[+] Model Genel Doğruluk Oranı (Accuracy): %{accuracy * 100:.2f}")
    
    # 5. Sonuçları Kaydetme
    os.makedirs("/app/output", exist_ok=True)
    output_path = "/app/output/titanic_survival_report.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Titanic Hayatta Kalma Tahmini Raporu (Random Forest) ===\n\n")
        f.write(f"Doğruluk Oranı (Accuracy): %{accuracy * 100:.2f}\n\n")
        f.write("Sınıflandırma Detayları:\n")
        f.write(report)
        f.write("\nKullanılan Özellikler:\n")
        for i, col in enumerate(features):
            f.write(f"- {col} (Önem Derecesi: %{model.feature_importances_[i] * 100:.2f})\n")
            
    print(f"--- İŞLEM BİTTİ ---")
    print(f"Detaylı rapor {output_path} konumuna kaydedildi.")

if __name__ == "__main__":
    run_survival_prediction()
