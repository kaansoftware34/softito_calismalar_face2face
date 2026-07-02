import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_passenger_clustering():
    print("--- Titanic Yolcu Kümeleme Analizi Başlatılıyor ---")
    
    # 1. Veri Yükleme
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("[+] Veri seti internet üzerinden indiriliyor...")
    df = pd.read_csv(url)
    
    # 2. Veri Temizleme ve Ön İşleme
    # Yolcuları sadece Yaş (Age) ve Ücret (Fare) metriklerine göre gruplayacağız
    features = ['Age', 'Fare']
    df_clean = df[features].dropna()
    
    # K-Means mesafeye dayalı çalıştığı için verileri standartlaştırmalıyız
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # 3. Model Eğitimi (K-Means)
    print("[+] K-Means modeli 3 farklı yolcu profili (küme) için eğitiliyor...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Kümeleri orijinal veriye ekleyelim
    df_clean['Cluster'] = clusters
    
    # Modelin kümeleme kalitesini ölçelim
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"[+] Model Silhouette Skoru: {sil_score:.3f} (1'e ne kadar yakınsa o kadar iyi)")
    
    # 4. Sonuçları Kaydetme
    os.makedirs("/app/output", exist_ok=True)
    
    # Kümelerin detaylı dökümü (CSV)
    csv_output_path = "/app/output/titanic_detailed_clusters.csv"
    df_clean.to_csv(csv_output_path, index=False)
    
    # Kümelerin özet raporu (TXT)
    txt_output_path = "/app/output/titanic_cluster_summary.txt"
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("=== Titanic Yolcu Kümeleme Özeti ===\n\n")
        f.write(f"Kullanılan Özellikler: {', '.join(features)}\n")
        f.write(f"Oluşturulan Küme Sayısı: 3\n")
        f.write(f"Silhouette Skoru: {sil_score:.3f}\n\n")
        f.write("Kümelerin Ortalama Değerleri:\n")
        
        # Her bir kümenin yaş ve bilet fiyatı ortalamasını hesaplayıp yazalım
        cluster_means = df_clean.groupby('Cluster').mean()
        for cluster_id, row in cluster_means.iterrows():
            f.write(f"Küme {cluster_id}:\n")
            f.write(f" - Ortalama Yaş: {row['Age']:.1f}\n")
            f.write(f" - Ortalama Bilet Ücreti: ${row['Fare']:.2f}\n")
            
    print(f"--- İŞLEM BİTTİ ---")
    print(f"Detaylı veriler {csv_output_path} konumuna kaydedildi.")
    print(f"Özet rapor {txt_output_path} konumuna kaydedildi.")

if __name__ == "__main__":
    run_passenger_clustering()
