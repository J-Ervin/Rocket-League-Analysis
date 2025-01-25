import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('match_data_with_wins.csv')

#Remove unnecessary columns
df = df.drop(columns=['color', 'team name', 'match_id', 'result'])

#Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

#Determine Optimal Number of Clusters
inertia = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

#Mathematical Code to Find Optimal Cluster Count
inertia_diff = np.diff(inertia)  
inertia_diff2 = np.diff(inertia_diff)  

elbow_index = np.argmax(inertia_diff2) + 2

print(f"The optimal number of clusters is: {elbow_index}")

#Fit K-means Model
kmeans = KMeans(n_clusters=elbow_index, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

#Add cluster labels
df['Cluster'] = df['Cluster'].astype('category')

print(df.head())

#Save Data
df.to_csv('match_data_with_clusters.csv', index=False)

#Calculate the mean for each cluster
cluster_profiles = df.groupby('Cluster').mean()

print(cluster_profiles)

#PCA Plot
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Scatter plot of PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Cluster'], cmap='viridis', s=100, edgecolors='black')
plt.title('PCA Projection of Clusters (Reduced to 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

features = ['shots', 'assists', 'time on ground', 'bpm', 'saves', 'shooting percentage', 'avg boost amount', 'amount stolen', '0 boost time', 'total distance', 'time slow speed', 'time low in air', 'time high in air', 'time boost speed', 'time behind ball', 'time in front of ball', 'time defensive half', 'time offensive half', 'time defensive third', 'time neutral third', 'time offensive third', 'time ball possession', 'time ball in own side', 'demos inflicted', 'demos taken']

#subplot setup
plt.figure(figsize=(15, 8))

#boxplot creation
for i, feature in enumerate(features, 1):
    plt.subplot(5, 6, i)  # 2 rows, 3 columns, feature i
    sns.boxplot(x='Cluster', y=feature, data=df, palette='Set2')
    plt.title(f'{feature} Distribution')

plt.tight_layout()
plt.show()
