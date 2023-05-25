from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.optimizers import Adam
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
