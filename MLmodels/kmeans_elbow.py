import sys
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data,
    split_X_data
)


def find_optimal_k(X):
    wcss = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        model.fit(X)
        wcss.append(model.inertia_)

    print("\nWCSS Values:")
    for k, value in zip(range(1, 11), wcss):
        print(f"K = {k}, WCSS = {value:.4f}")

    # Elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS Value")
    plt.title("Elbow Method for Optimal K")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show(block=False)
    return wcss


def create_model(k):
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
    return model


def train_model(model, X):
    labels = model.fit_predict(X)
    print("\nK-Means trained successfully!")
    return model, labels


def evaluate_model(model, X, labels):
    print("\nInertia (WCSS Value):")
    print(model.inertia_)
    print("\nIterations to Converge:")
    print(model.n_iter_)
    sil_score = silhouette_score(X, labels)
    print("\nSilhouette Score:")
    print(sil_score)
    return sil_score


def display_clusters(X, labels, model):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        cmap="viridis",
        s=30
    )
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=200,
        color="red",
        label="Centroids"
    )
    plt.title("K-Means Clustering")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.legend()
    plt.show(block=False)


def main():
    df = load_data()
    print("Original Dataset Shape:")
    print(df.shape)

    X = split_X_data(
        df,
        drop_columns=[
            "StudentID",
            "PlacementStatus",
            "Salary Package",
            "IsAnomaly",
            "IsAnomally"
        ]
    )
    print("\nK-Means Dataset Shape:")
    print(X.shape)

    numerical_features, categorical_features = identify_features(X)
    print("\nNumerical Features:")
    print(numerical_features)
    print("\nCategorical Features:")
    print(categorical_features)

    one_hot_features = [f for f in ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"] if f in X.columns]
    ordinal_features = [f for f in ["CollegeTier", "CGPA_Tier"] if f in X.columns]

    if numerical_features:
        X, _, _ = handle_missing_values(X, X, numerical_features)
        print("\nMissing Value Handling Completed.")

        X, _, _ = standardize_data(X, X, numerical_features)
        print("\nStandardization Completed.")

    if one_hot_features:
        X, _, _ = one_hot_encode_data(X, X, one_hot_features)
        print("\nOne-Hot Encoding Completed.")

    if ordinal_features:
        X, _, _ = ordinal_encode_data(X, X, ordinal_features)
        print("\nOrdinal Encoding Completed.")

    print("\nFinding Optimal K using Elbow Method...")
    wcss = find_optimal_k(X)

    optimal_k = 3
    print(f"\nBuilding final K-Means model with K = {optimal_k}...")
    model = create_model(optimal_k)
    model, labels = train_model(model, X)

    evaluate_model(model, X, labels)
    display_clusters(X, labels, model)


if __name__ == "__main__":
    main()