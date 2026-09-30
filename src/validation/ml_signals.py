import numpy as np
import pandas as pd
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
    """
    labels = make_next_day_direction_labels(train_featured)

    features_ready = train_featured[feature_columns].notna().all(axis=1)
    valid = labels.notna() & features_ready

    X_train = train_featured.loc[valid, feature_columns]
    y_train = labels.loc[valid]

    unique_classes = y_train.unique()
    if len(unique_classes) < 2:
        # Every labeled day in this window went the same direction (e.g. a
        # strong, near-monotonic trend). There is no boundary to learn - the
        # model has literally never seen the other outcome happen. The
        # rational prediction is that one class, with full confidence,
        # rather than trying to fit a classifier with nothing to separate.
        constant_probability = float(unique_classes[0])
        return pd.Series(constant_probability, index=test_featured.index)

    model = LogisticRegression(C=0.5, max_iter=1000)
    model.fit(X_train, y_train)

    X_test = test_featured[feature_columns]
    probabilities = model.predict_proba(X_test)[:, 1]

    return pd.Series(probabilities, index=test_featured.index)