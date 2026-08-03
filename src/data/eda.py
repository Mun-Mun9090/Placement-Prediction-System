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




if __name__=="__main__":
    df=load_data()
    basic_eda(df)