

import seaborn as sns

import pandas as pd

from src.data.load_data import load_data
import matplotlib.pyplot as plt
import os
import shutil
results_dir = r"E:\PythonProject\Placement_Prediction_System\results"
charts_dir = r"E:\PythonProject\Placement_Prediction_System\app\static\charts"
# Ensure charts directory exists
os.makedirs(charts_dir, exist_ok=True)
# Move any existing images from old results directory into charts
if os.path.isdir(results_dir):
    for fname in os.listdir(results_dir):
        src = os.path.join(results_dir, fname)
        dst = os.path.join(charts_dir, fname)
        try:
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        except Exception:
            pass

def basic_eda(df):
    print("First five rows")
    print(df.head())
    print("Last five rows")
    print(df.tail())
    print("25 to 35 records/rows")
    print(df.iloc[25:36])
    print("Sample of 10 records")
    print(df.sample(10))
    print("Column names")
    print(df.columns)
    print("Gender column")
    print(df[df.columns[1]])
    print("Complete information of the dataset: ")
    print(df.info())
    print("Description of the dataset: ")
    print(df.describe())

    print("Missing values: ")
    print(df.isnull().sum())

    print("Duplicate values: ")
    print(df.duplicated().sum())

    print("Target variable status: ")
    print(df["PlacementStatus"].value_counts())
    count=df["PlacementStatus"].value_counts()
    plt.figure(figsize=(6,5))
    plt.bar(count.index,count.values)
    plt.title("Distribution of Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\placement_status.png")
    plt.show()

def univariate(df):
    plt.figure(figsize=(6,5))
    plt.hist(df["CGPA"],bins=10,color="yellow")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\cgpa_stats.png")
    plt.show()
    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6,5))
    plt.pie(gendercount,labels=gendercount.index,autopct="%1.1f%%",startangle=90)
    plt.title("Distribution of Gender")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\gender_count.png")
    plt.show()

def bivariate(df):

    #Scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(df["CGPA"],df["AptitudeTestScore"],color="orange")
    plt.title("CGPA vs Aptitude Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Score")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\cgpa_aptitude_scores.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(6,5))
    placed=df[df["PlacementStatus"]==1]["CGPA"]
    not_placed=df[df["PlacementStatus"]==0]["CGPA"]
    plt.boxplot([placed,not_placed],labels=["placed","not placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\cgpa_placement_status.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(6,5))
    count=pd.crosstab(df["Gender"],df["PlacementStatus"])
    count.plot(kind="bar",stacked=True,rot=0)
    plt.title("Gender vs Placement Status")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\gender_placement_status.png")
    plt.show()
    plt.close()

def multivariate(df):
    data=df[["CGPA","AptitudeTestScore","PlacementStatus"]]
    correlation = data.corr(numeric_only=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f")
    plt.title("Correlation HeatMap")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\app\static\charts\CGPA_correlation_heatmap.png")
    plt.show()


if __name__=="__main__":
    df=load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)