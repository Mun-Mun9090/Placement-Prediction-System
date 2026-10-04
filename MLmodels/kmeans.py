import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent / 'data' / 'placement_data.csv'

df = pd.read_csv(file_path)

features = [
    "CGPA",
    "AttendancePercent",
    "Projects",
    "CodingTestScore"
]
X = df[features].dropna().copy()

print("Selected Features:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

X['cluster'] = clusters

print("\nCluster Assignment:")
print(X.head(10))

centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)

centers_df = pd.DataFrame(
    centers,
    columns=features
)

print("\nCluster Centers:")
print(centers_df)
print("\nStudents in Each Cluster:")
print(X["cluster"].value_counts().sort_index())

plt.figure(figsize=(8, 6))

plt.scatter(
    X["CGPA"],
    X["CodingTestScore"],
    c=X["cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("K-Means Clustering of Students")

plt.show()
