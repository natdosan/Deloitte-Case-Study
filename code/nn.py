from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

def create_model(categorical_vars, df):
    """
    Create a Neural Network to predict the "uses drugs" column

    Parameters
    ----------
    categorical_vars : list
        List of names of the categorical variables.
    df : pandas DataFrame 
        df containing the categorical variables.

    Returns
    -------
    model : keras.models.Model()
        A compiled Keras model for our problem
    """
    inputs = []
    embeddings = []

    for categorical_var in categorical_vars:
        no_of_unique_cat  = df[categorical_var].nunique()

        # Calculate the size of the embedding layer. The size is half the number of unique values (capped at 50)
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
        embedding_size = int(embedding_size)

        # The vocabulary size is one more than the number of unique categories
        vocab  = no_of_unique_cat+1

        # create input, embedding layers
        input_cat = Input(shape=(1,))
        embedding = Embedding(vocab ,embedding_size, input_length = 1)(input_cat)
        embedding = Flatten()(embedding)
        inputs.append(input_cat)
        embeddings.append(embedding)

    # Concatenate all embedding layers together
    x = Concatenate()(embeddings)

    # Add a dense layer (fully-connected layer) using the relu activation function
    x = Dense(256, activation='relu')(x)

    # output layer (note we use sigmoid as the activation since our output is binary)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_and_evaluate(categorical_vars, X, y, epochs=10, batch_size=32, test_size=0.2, random_state=42):
    """
    Creates a Keras model, trains it and evaluates it on a test set.

    Parameters
    ----------
    categorical_vars : list
        List of names of the categorical variables.
    X : pandas DataFrame
        df containing the feature data
    y : pandas Series c
        ontaining the target data.
    epochs : int
        Number of epochs for training the model. Defaults to 10.
    batch_size : int
        Batch size for training the model. Defaults to 32.
    test_size : float
        Proportion of the dataset to include in the test split. Defaults to 0.2.
    random_state :
        The seed used by the random number generator. Defaults to 42.

    Returns
    -------
    model : keras.models.Model()
        A trained Keras model for our problem
    loss : float
        Loss of the model on the test set.
    accuracy : float
        Accuracy of the model on the test set.
    """


    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X[categorical_vars], 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    model = create_model(categorical_vars, X)

    # Train and fit the model
    X_train_list = [X_train[feature].values for feature in categorical_vars]
    model.fit(X_train_list, 
              y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_split=0.2)

    # Evaluate the model
    X_test_list = [X_test[feature].values for feature in categorical_vars]
    loss, accuracy = model.evaluate(X_test_list, y_test)

    return model, loss, accuracy
