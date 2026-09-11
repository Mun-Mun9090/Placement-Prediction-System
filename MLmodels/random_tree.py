from pyexpat import features
from shlex import split

from src.data.load_data import load_data

from sklearn.ensemble import RandomForestClassifier

from src.data.preprocess import split_data, identify_features, X_train, one_hot_encode_data, handle_missing_values


def create_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        oob_score=True
    )
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nDecision tree trained sucessfully")
    return model

def evaluate_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    print("\nAccuracy: ", accuracy)
    print("\nOOB Score: ", model.oob_score_)
    return accuracy

def main():
    data = load_data()
    print("Original Dataset Shape:")
    print(data.shape)
    x_train, y_train, x_test, y_test = split_data(
        data,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "ISAnomally"
        ]
    )

    numerical_features, categorical_features = (
        identify_features(X_train)
    )

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "specialization",
        "Hostel",
        "HistoryOfBacklogs"

    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    X_train,X_test,imputer=handle_missing_values(
        X_train,
        X_test,
        numerical_features
    )

    print