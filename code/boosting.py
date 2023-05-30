import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def calculate_feature_importances(df, response):
    """
    Calculates feature importances using an XGBoost classifier and returns a 
    dataframe listing the features and their corresponding importance.

    Parameters
    ----------
    df : pandas DataFrame 
        input df containing all features and target.
    target_col : str
        The target column name in the dataframe.

    Returns
    -------
    feature_importance_df : pandas DataFrame 
        A dataframe containing 'Feature' (feature names) and 'Importance' 
        (calculated importance) as columns, sorted in descending order 
    """

    # Splitting the data into features and target
    X = df.drop(response, axis=1)
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating the pipeline
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    pipeline = Pipeline(steps=[('model', model)])
    pipeline.fit(X_train, y_train)

    # Get Feature Importance from the classifier
    feature_importance = pipeline.named_steps['model'].feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    return feature_importance_df.sort_values('Importance', ascending=False)


def boosted_decision_tree(X, y, X_train, X_test, y_train, y_test, categorical_vars):
    """
    Performs Boosted Decision Tree Algorithm on drug data

    Parameters
    ----------
    X : pandas DataFrame 
        input df containing all features and target.
    y : pandas Series or numpy 1-d array
        The response column name in the dataframe.

    Returns
    -------
    None
    """

    # create XGBClassifier
    xg_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    xg_clf.fit(X_train,y_train)

    # Predict the labels of the test set
    y_pred = xg_clf.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    # Compute feature importances
    feature_importances = xg_clf.feature_importances_
    for feature, importance in zip(categorical_vars, feature_importances):
        print(f"Feature: {feature}, Importance: {importance}")

    # Compute cross-validation scores
    cv_scores = cross_val_score(xg_clf, X, y, cv=5)
    print("\nCross-validation scores:")
    print(cv_scores)
    print("\nMean cross-validation score:")
    print(cv_scores.mean())

    pickle.dump(xg_clf, open("../models/xgboost_model.pkl", "wb"))