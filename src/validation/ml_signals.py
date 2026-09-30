import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def make_next_day_direction_labels(df: pd.DataFrame) -> pd.Series:
    """
    Label for each day: 1 if the NEXT day's close is higher than today's,
    0 otherwise. The last row has no "next day" within this data, so it's
    NaN and must be dropped before fitting (rows with a NaN label can't be
    learned from).
    """
    next_close = df["Close"].shift(-1)
    labels = (next_close > df["Close"]).astype(float)
    labels[next_close.isna()] = np.nan
    return labels

def fit_and_predict_logistic(train_featured: pd.DataFrame, test_featured: pd.DataFrame,
                             feature_columns: list) -> pd.Series:
    """
    Fits a logistic regression on train_featured's features and labels,
    then predicts probabilities of "up" for every row of test_featured.

    Both DataFrames must already have the feature columns computed. Rows
    in train_featured with a NaN label (the last row, with no known next
    day) are dropped before fitting.
    """
    labels = make_next_day_direction_labels(train_featured)
    valid = labels.notna()

    X_train = train_featured.loc[valid, feature_columns]
    y_train = labels.loc[valid]

    model = LogisticRegression(C=0.5, max_iter=1000)
    model.fit(X_train, y_train)

    X_test = test_featured[feature_columns]
    probabilities = model.predict_proba(X_test)[:, 1]

    return pd.Series(probabilities, index=test_featured.index)