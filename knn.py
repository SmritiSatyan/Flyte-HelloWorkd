# Import the dependencies
from typing import List, NamedTuple

import pandas as pd
from flytekit import task, workflow
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Use NamedTuple to name the output.
split_data = NamedTuple(
    "split_data",
    train_features=pd.DataFrame,
    test_features=pd.DataFrame,
    train_labels=pd.DataFrame,
    test_labels=pd.DataFrame,
)


# Define a task that processes the wine dataset after loading it into the environment.
@task
def data_processing() -> split_data:
    # load wine dataset
    wine = load_wine()

    # convert features and target (numpy arrays) into Modin DataFrames
    wine_features = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    wine_target = pd.DataFrame(data=wine.target, columns=["species"])

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        wine_features, wine_target, test_size=0.4, random_state=101
    )
    print("Sample data:")
    print(X_train.head(5))
    return split_data(
        train_features=X_train,
        test_features=X_test,
        train_labels=y_train,
        test_labels=y_test,
    )


# Define a task that:
#
# 1. trains a KNeighborsClassifier model,
# 2. fits the model to the data, and
# 3. predicts the output for the test dataset.
@task
def fit_and_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
) -> List[int]:
    lr = KNeighborsClassifier()  # create a KNeighborsClassifier model
    lr.fit(X_train, y_train)  # fit the model to the data
    predicted_vals = lr.predict(X_test)  # predict values for test data
    return predicted_vals.tolist()


# Define a task to compute the accuracy of the model.
@task
def calc_accuracy(y_test: pd.DataFrame, predicted_vals_list: List[int]) -> float:
    return accuracy_score(y_test, predicted_vals_list)


# Define a workflow that shows order of execution of tasks.
@workflow
def pipeline() -> float:
    split_data_vals = data_processing()
    predicted_vals_output = fit_and_predict(
        X_train=split_data_vals.train_features,
        X_test=split_data_vals.test_features,
        y_train=split_data_vals.train_labels,
    )
    return calc_accuracy(
        y_test=split_data_vals.test_labels, predicted_vals_list=predicted_vals_output
    )

# You can run the code locally.
if __name__ == "__main__":
    print(f"Accuracy of the model is {pipeline()}%")
