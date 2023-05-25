import pandas as pd
from sklearn.model_selection import train_test_split
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
