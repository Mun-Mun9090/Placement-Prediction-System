from src.data.load_data import load_data
from src.data.preprocess import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def create_model(k):
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
