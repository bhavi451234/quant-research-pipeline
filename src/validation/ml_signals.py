import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def make_next_day_direction_labels(df: pd.DataFrame) -> pd.Series:
    """
    Label for each day: 1 if the NEXT day's close is higher than today's,
    0 otherwise. The last row has no "next day" within this data, so it's
    NaN and must be dropped before fitting.
    """
    next_close = df["Close"].shift(-1)
    labels = (next_close > df["Close"]).astype(float)
    labels[next_close.isna()] = np.nan
    return labels


def _prepare_training_set(train_featured: pd.DataFrame, feature_columns: list):
    """
    Shared by both fit_and_predict_* functions: builds labels, then keeps
    only rows where both the label is known (not the last row) and every
    feature is fully computed (not in a window's warm-up zone).
    """
    labels = make_next_day_direction_labels(train_featured)
    features_ready = train_featured[feature_columns].notna().all(axis=1)
    valid = labels.notna() & features_ready
    X_train = train_featured.loc[valid, feature_columns]
    y_train = labels.loc[valid]
    return X_train, y_train


def fit_and_predict_logistic(train_featured: pd.DataFrame, test_featured: pd.DataFrame,
                             feature_columns: list) -> tuple:
    """
    Fits a logistic regression on train_featured's features and labels,
    then predicts probabilities of "up" for every row of test_featured.
    """
    X_train, y_train = _prepare_training_set(train_featured, feature_columns)

    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        constant_probability = float(unique_classes[0])
        return pd.Series(constant_probability, index=test_featured.index), None

    model = LogisticRegression(C=0.5, max_iter=1000)
    model.fit(X_train, y_train)

    X_test = test_featured[feature_columns]
    probabilities = model.predict_proba(X_test)[:, 1]

    return pd.Series(probabilities, index=test_featured.index), model


def fit_and_predict_gradient_boosting(train_featured: pd.DataFrame, test_featured: pd.DataFrame,
                                      feature_columns: list) -> tuple:
    """
    Same job as fit_and_predict_logistic, using a gradient boosted tree
    ensemble instead. Kept conservative on purpose (shallow trees, modest
    tree count, low learning rate) for the same reason C=0.5 was chosen for
    logistic regression: ~500-row windows are easy to overfit with a
    flexible model.
    """
    X_train, y_train = _prepare_training_set(train_featured, feature_columns)

    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        constant_probability = float(unique_classes[0])
        return pd.Series(constant_probability, index=test_featured.index), None

    model = GradientBoostingClassifier(max_depth=2, n_estimators=75, learning_rate=0.05, random_state=0)
    model.fit(X_train, y_train)

    X_test = test_featured[feature_columns]
    probabilities = model.predict_proba(X_test)[:, 1]

    return pd.Series(probabilities, index=test_featured.index), model