import sys
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    ordinal_encode_data
)


# ==================================================
# 1. CREATE GRADIENT BOOSTING MODEL
# ==================================================

def create_model():

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    return model


# ==================================================
# 2. TRAIN MODEL
# ==================================================

def train_model(model, x_train, y_train):

    model.fit(x_train, y_train)

    print("\n========================================")
    print("Gradient Boosting Model Trained Successfully")
    print("========================================\n")

    return model


# ==================================================
# 3. EVALUATE MODEL
# ==================================================

def evaluate_model(model, x_test, y_test):

    # Predict test data
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n========================================")
    print("MODEL EVALUATION")
    print("========================================")

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return y_pred


# ==================================================
# 4. DISPLAY FEATURE IMPORTANCE
# ==================================================

def display_feature_importance(model, feature_names):

    # Get feature importance values
    importances = model.feature_importances_

    # Sort features from highest importance
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(12, 8))

    plt.barh(
        range(len(importances)),
        importances[indices]
    )

    plt.yticks(
        range(len(importances)),
        [feature_names[i] for i in indices]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Features")

    plt.title(
        "Gradient Boosting - Feature Importance"
    )

    # Most important feature at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show(block=False)


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    print("\nLoading Dataset...")

    df = load_data()

    print("\nDataset Loaded Successfully.")

    print("\nDataset Shape:")
    print(df.shape)


    # ==================================================
    # 2. SPLIT DATA
    # ==================================================

    print("\nSplitting Dataset...")

    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nData Split Completed.")

    print("\nTraining Data Shape:")
    print(x_train.shape)

    print("\nTesting Data Shape:")
    print(x_test.shape)


    # ==================================================
    # 3. IDENTIFY FEATURES
    # ==================================================

    numerical_features, categorical_features = identify_features(
        x_train
    )

    print("\n========================================")
    print("FEATURE IDENTIFICATION")
    print("========================================")

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)


    # ==================================================
    # 4. DEFINE ENCODING FEATURES
    # ==================================================

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]


    # ==================================================
    # 5. HANDLE MISSING VALUES
    # ==================================================

    print("\nHandling Missing Values...")

    x_train, x_test, imputer = handle_missing_values(
        x_train,
        x_test,
        numerical_features
    )

    print("Missing Value Handling Completed.")


    # ==================================================
    # 6. STANDARDIZE NUMERICAL FEATURES
    # ==================================================

    print("\nStandardizing Numerical Features...")

    x_train, x_test, scaler = standardize_data(
        x_train,
        x_test,
        numerical_features
    )

    print("Standardization Completed.")


    # ==================================================
    # 7. ONE-HOT ENCODING
    # ==================================================

    print("\nPerforming One-Hot Encoding...")

    x_train, x_test, one_hot_encoder = one_hot_encode_data(
        x_train,
        x_test,
        one_hot_features
    )

    print("One-Hot Encoding Completed.")


    # ==================================================
    # 8. ORDINAL ENCODING
    # ==================================================

    print("\nPerforming Ordinal Encoding...")

    x_train, x_test, ordinal_encoder = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )

    print("Ordinal Encoding Completed.")


    # ==================================================
    # 9. DISPLAY FINAL DATA INFORMATION
    # ==================================================

    print("\n========================================")
    print("FINAL PROCESSED DATA")
    print("========================================")

    print("\nFinal Training Shape:")
    print(x_train.shape)

    print("\nFinal Testing Shape:")
    print(x_test.shape)

    print("\nFinal Features:")
    print(list(x_train.columns))


    # ==================================================
    # 10. CREATE GRADIENT BOOSTING MODEL
    # ==================================================

    print("\nCreating Gradient Boosting Model...")

    model = create_model()


    # ==================================================
    # 11. TRAIN MODEL
    # ==================================================

    model = train_model(
        model,
        x_train,
        y_train
    )


    # ==================================================
    # 12. EVALUATE MODEL
    # ==================================================

    y_pred = evaluate_model(
        model,
        x_test,
        y_test
    )


    # ==================================================
    # 13. DISPLAY FEATURE IMPORTANCE
    # ==================================================

    print("\nDisplaying Feature Importance...")

    display_feature_importance(
        model,
        x_train.columns
    )


# ==================================================
# RUN PROGRAM
# ==================================================

if __name__ == "__main__":
    main()
