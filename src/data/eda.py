from src.data.load_data import load_data
import matplotlib.pyplot as plt
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
    print(df.isnull().sum()>0)

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
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\results\placement_status.png")
    plt.show()

def univariate(df):
    plt.figure(figsize=(6,5))
    plt.hist(df["CGPA"],bins=10,color="yellow")
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\results\cgpa_stats.png")
    plt.show()
    gendercount = df["Gender"].value_counts()
    plt.figure(figsize=(6,5))
    plt.pie(gendercount,labels=gendercount.index,autopct="%1.1f%%",startangle=90)
    plt.title("Distribution of Gender")
    plt.xlabel("Gender")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\results\gender_count.png")
    plt.show()

def bivariate(df):

    #Scatter plot
    plt.figure(figsize=(6,5))
    plt.scatter(df["CGPA"],df["AptitudeTestScore"],color="orange")
    plt.title("CGPA vs Aptitude Score")
    plt.xlabel("CGPA")
    plt.ylabel("Aptitude Score")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\results\cgpa aptitude scores.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(6,5))
    placed=df[df["PlacementStatus"]==1]["CGPA"]
    not_placed=df[df["PlacementStatus"]==0]["CGPA"]
    plt.boxplot([placed,not_placed],label=["placed","not placed"])
    plt.title("CGPA vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("CGPA")
    plt.savefig(r"E:\PythonProject\Placement_Prediction_System\results\cgpa placement status.png")
    plt.show()

if __name__=="__main__":
    df=load_data()
    basic_eda(df)
    univariate(df)
    bivariate(df)