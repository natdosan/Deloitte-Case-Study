import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
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



def train_and_evaluate_model(model, X, y, model_name, features):
    """
    Trains and evaluates a model using provided features and target.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator or xgboost.XGBClassifier
        The machine learning model to train and evaluate.
    X : pandas DataFrame
        DataFrame containing the features for training and evaluating the model.
    y : pandas Series or numpy 1-d array
        The target for training and evaluating the model.
    model_name : str
        The name of the model, used for saving the model to a pickle file.
    features : list
        list of the most significant features

    Returns
    -------
    pandas DataFrame : feature importances
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    # Precision, Recall, F1 score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)

    # Cross-validation Scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\nCross-validation scores:")
    print(cv_scores)
    print("\nMean cross-validation score:")
    print(cv_scores.mean())

    # Print feature importances for tree-based models
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        for feature, importance in zip(features, feature_importances):
            print(f"Feature: {feature}, Importance: {importance}")

        result = pd.DataFrame({'Feature' : features,
                        'Importance' : feature_importances,
                        }, columns=['Feature','Importance']).sort_values(by = 'Importance', ascending=False)

    # otherwise print coefficients
    else:
        coeffs = model.coef_
        for feature, coeff in zip(features, coeffs):
            print(f"Feature: {feature}, Coefficient: {coeff}")

        #result = pd.DataFrame({'Feature' : features,
        #                'Coefficient' : coeffs,
        #                }, columns=['Feature','Coefficient']).sort_values(by = 'Coefficient', ascending=False)
        result = coeffs

    # Save model to a pickle file
    pickle.dump(model, open(f"../models/{model_name}.pkl", "wb"))
    
    return result
