import json

import keras

from individual import Individual
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
from keras.models import Sequential
from keras.layers import Dense
import tensorflow

# HYPERPARAMETERS
importdata = True
randomrelabel = False
generations = 10
num_experiments = 1
population_size = 22
num_parents = 4
mutation_rate = 0.5

start_time = time.time()
# shouldn't be used other than for testing
# input_dim = 20


# AIDAN DEGOOYER - JAN 28, 2024
#
# Works Cited:
# DATASET: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
#
# https://keras.io/guides/training_with_built_in_methods/
# https://pandas.pydata.org/docs/reference/frame.html#dataframe
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe
# https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html
# https://www.javatpoint.com/how-to-add-two-lists-in-python
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#
# Debugging: https://datascience.stackexchange.com/questions/67047/loss-being-outputed-as-nan-in-keras-rnn
#            https://stackoverflow.com/questions/49135929/keras-binary-classification-sigmoid-activation-function
#


# data parsing begins here======================================================================================
# Load data from csv
if importdata:
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # Remove id and number
    train_data = train_data.iloc[:, 2:]
    test_data = test_data.iloc[:, 2:]

    # Input missing values with the mean
    train_data.fillna(train_data.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)

    # set number of data points from each class
    b1 = 100
    b2 = 300

    class_0_train = train_data[train_data['label'] == 0]
    class_1_train = train_data[train_data['label'] == 1]


    def resample_data():
        global train_class_0
        global train_class_1
        global train_data
        train_class_0 = resample(class_0_train, n_samples=b1, random_state=42)
        train_class_1 = resample(class_1_train, n_samples=b2, random_state=42)
        train_data = pd.concat([train_class_0, train_class_1])
        global X_train
        global y_train
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values


    resample_data()

    # Sample b1 data points from one class and b2 from the other
    if randomrelabel:
        # Randomly relabel 5% of the training data from each class
        num_samples_class_0_relabel = int(0.05 * len(train_class_0))
        num_samples_class_1_relabel = int(0.05 * len(train_class_1))

        selected_indices_class_0_relabel = np.random.choice(train_class_0.index, size=num_samples_class_0_relabel,
                                                            replace=False)
        selected_indices_class_1_relabel = np.random.choice(train_class_1.index, size=num_samples_class_1_relabel,
                                                            replace=False)

        train_data.loc[selected_indices_class_0_relabel, 'label'] = 1
        train_data.loc[selected_indices_class_1_relabel, 'label'] = 0

        # Concatenate the relabeled data with the original data
        train_data = pd.concat([train_class_0, train_class_1])

    # Separate testing data into two classes based on labels
    class_0_test = test_data[test_data['label'] == 0]
    class_1_test = test_data[test_data['label'] == 1]

    # Sample 100 data points from each class for testing
    test_class_0 = class_0_test.sample(n=100, random_state=42)
    test_class_1 = class_1_test.sample(n=100, random_state=42)
    test_data = pd.concat([test_class_0, test_class_1])

    # Split features and labels for training and testing sets
    X_test = test_data.drop(columns=['label']).values
    y_test = test_data['label'].values

    # standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]


# END data parsing ================================================================================================




def evaluate_model_conf_matrix(x_test, y_test, model):
    # not really useful except when needing confusion matricies for a report
    current_model = Sequential()
    current_model.add(Dense(units=input_dim, activation='sigmoid', input_dim=input_dim))
    current_model.add(Dense(units=input_dim * 2, activation='sigmoid'))
    current_model.add(Dense(units=1, activation='sigmoid'))
    current_model.set_weights(model.get_weights())
    predictions = current_model.predict(x_test)
    rounded_predictions = np.rint(predictions)
    conf_matrix = confusion_matrix(y_test, rounded_predictions)  # needs help
    return conf_matrix

def create_model():
    model = Sequential()
    model.add(Dense(units=input_dim * 2, activation='sigmoid', input_dim=input_dim))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the fitness function (evaluates the performance of the neural network)
def fitness_function(weights):
    model = create_model()
    model.set_weights(weights)
    model.fit(X_train, y_train, epochs=10, verbose=0)  # Train the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Make predictions
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    print(accuracy)
    return -accuracy  # Minimize (negative of accuracy)

optimized_model_list = []

for i in range(num_experiments):
    # Initialize the bounds for differential evolution (assuming weights have the same shape as in the model)
    bounds = [(-1, 1)] * sum(w.size for w in create_model().get_weights())

    # Perform differential evolution to optimize the weights
    result = differential_evolution(fitness_function, bounds, maxiter=generations)

    # Get the optimized weights
    optimized_weights = result.x
    optimized_model = create_model()
    optimized_model.set_weights(optimized_weights)

    current_individual = Individual()
    current_individual.set_weights(optimized_weights)
    current_individual.calculate_q(X_train, y_train, b1, b2)

    # Evaluate the optimized model
    y_pred_optimized = (optimized_model.predict(X_test) > 0.5).astype(int)
    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)

    optimized_model_list.append(current_individual)

# Compare results
print("Accuracy (optimized model):", accuracy_optimized)

print("--- %s seconds ---" % (time.time() - start_time))

file = open("output.txt", "w")


for i, model in enumerate(optimized_model_list):
    file.write((f"Experiment #{i+1} - Train"))
    conf_matrix = evaluate_model_conf_matrix(X_train, y_train, model)
    file.write(f"\n{conf_matrix}\n")

    file.write((f"Experiment #{i + 1} - Test"))
    conf_matrix = evaluate_model_conf_matrix(X_test, y_test, model)
    file.write(f"\n{conf_matrix}\n")


file.write("---------------------\nbest q's in trials\n")

for item in optimized_model_list:
    file.write(f"\n{item.q}\n")

file.close()