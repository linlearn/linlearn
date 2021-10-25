from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import pandas as pd
import pickle
import numpy as np
import gzip

data_folder = "offline_datasets/"

def load_simulated_regression(n_samples, n_features, test_size=0.3, random_state=42):

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=n_samples, n_features=n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test

def load_used_car(test_size=0.3, random_state=42):

    csv = pd.read_csv(data_folder + "used_car/vw.csv")
    csv["year"] = 2020 - csv["year"]
    csv = csv.drop("model", axis=1)
    categoricals = ["transmission", "fuelType"]  # , "model"]
    label = "price"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv[cat], prefix=cat)
        csv = csv.drop(cat, axis=1)
        csv = csv.join(one_hot)
    csv = pd.DataFrame(StandardScaler().fit_transform(csv), columns=csv.columns)
    df_train, df_test = train_test_split(
        csv,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)

    return (
        df_train.to_numpy(),
        df_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )


def load_data_boston(test_size=0.3, random_state=42):
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def load_california_housing(test_size=0.3, random_state=42):
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test


def fetch_diabetes(test_size=0.3, random_state=42):
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test



def load_mnist(test_size=0.3, random_state=42):################
    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), "B", offset=16)
        return pixels.reshape(-1, 784).astype("float64") / 255


    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), "B", offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype="uint8")
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)


    mnist_train_images_file = data_folder + "mnist_data/train-images-idx3-ubyte.gz"
    mnist_train_labels_file = data_folder + "mnist_data/train-labels-idx1-ubyte.gz"

    mnist_test_images_file = data_folder + "mnist_data/t10k-images-idx3-ubyte.gz"
    mnist_test_labels_file = data_folder + "mnist_data/t10k-labels-idx1-ubyte.gz"

    X_train = _images(mnist_train_images_file)
    y_train = _labels(mnist_train_labels_file)

    X_test = _images(mnist_test_images_file)
    y_test = _labels(mnist_test_labels_file)
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def load__iris(test_size=0.3, random_state=42):############

    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def load_simulated(n_samples, n_features, n_classes, test_size=0.3, random_state=42):

    from sklearn.datasets import make_multilabel_classification

    X, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                          allow_unlabeled=False)
    y = np.argmax(y, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def load_bank(test_size=0.3, random_state=42):
    with open(data_folder + "pickled_bank.pickle", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    f.close()
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def load_adult(test_size=0.3, random_state=42):
    with open(data_folder + "pickled_adult.pickle", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    f.close()
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def load_heart(test_size=0.3, random_state=42):
    csv_heart = pd.read_csv(data_folder + "heart/heart.csv")
    categoricals = ["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"]
    label = "output"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_heart[cat], prefix=cat)
        csv_heart = csv_heart.drop(cat, axis=1)
        csv_heart = csv_heart.join(one_hot)
    df_train, df_test = train_test_split(
        csv_heart,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_heart[label],
    )

    y_train = df_train.pop(label)
    y_test = df_test.pop(label)

    return (
        df_train.to_numpy(),
        df_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )

def load_htru2(test_size=0.3, random_state=42):
    csv_htru2 = pd.read_csv(data_folder + "HTRU2/HTRU_2.csv", header=None)

    label = 8

    df_train, df_test = train_test_split(
        csv_htru2,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_htru2[label],
    )

    y_train = df_train.pop(label)
    y_test = df_test.pop(label)

    return (
        df_train.to_numpy(),
        df_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )



def load_stroke(test_size=0.3, random_state=42):
    csv_stroke = (
        pd.read_csv(data_folder + "stroke/healthcare-dataset-stroke-data.csv")
        .drop("id", axis=1)
        .dropna()
    )

    categoricals = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    label = "stroke"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_stroke[cat], prefix=cat)
        csv_stroke = csv_stroke.drop(cat, axis=1)
        csv_stroke = csv_stroke.join(one_hot)
    df_train, df_test = train_test_split(
        csv_stroke,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_stroke[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return (
        df_train.to_numpy(),
        df_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )


def load_aus(test_size=0.3, random_state=42):
    # drop columns with too many NA values and Location because it has too many categories
    csv_aus = pd.read_csv(data_folder + "weatherAUS/weatherAUS.csv").drop(
        ["Sunshine", "Evaporation", "Cloud9am", "Cloud3pm", "Location"], axis=1
    )
    # convert date to yearday
    csv_aus = csv_aus.rename(columns={"Date": "yearday"})
    csv_aus["yearday"] = csv_aus.yearday.apply(
        lambda x: 30 * (int(x[5:7]) - 1) + int(x[8:])
    ).astype(int)
    categoricals = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    label = "RainTomorrow"
    for cat in categoricals:
        one_hot = pd.get_dummies(csv_aus[cat], prefix=cat)
        csv_aus = csv_aus.drop(cat, axis=1)
        csv_aus = csv_aus.join(one_hot)

    csv_aus["RainToday"] = LabelBinarizer().fit_transform(
        csv_aus["RainToday"].astype("str")
    )
    csv_aus["RainTomorrow"] = LabelBinarizer().fit_transform(
        csv_aus["RainTomorrow"].astype("str")
    )
    csv_aus = csv_aus.dropna()
    df_train, df_test = train_test_split(
        csv_aus,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
        stratify=csv_aus[label],
    )
    y_train = df_train.pop(label)
    y_test = df_test.pop(label)
    return (
        df_train.to_numpy(),
        df_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )
