from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.metrics import confusion_matrix


class Individual:
    def __init__(self, input_dim):
        self.q = None
        self.input_dim = input_dim
        self.model = Sequential()
        self.model.add(Dense(units=self.input_dim * 2, activation='sigmoid'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.weights = self.model.get_weights()

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights
        return

    def calculate_q(self, x_train, y_train, b1, b2):
        self.model.set_weights(self.weights)
        predictions = self.model.predict(x_train, verbose=0)
        rounded_predictions = np.rint(predictions)
        conf_matrix = confusion_matrix(y_train, rounded_predictions)  # needs help
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]

        # Calculate q value
        self.q = (FP + FN) / (b1 + b2) + 10 * (max(0, FP / b1 - 0.1) + max(0, FN / b2 - 0.1))
        return

    def mutate(self, mutation_rate):
        mutated_weights = []
        mutation_function = lambda x, mutation_rate: x + np.random.normal(loc=0.0, scale=mutation_rate, size=x.shape)
        for layer_weights in self.weights:
            mutated_layer_weights = [mutation_function(w, mutation_rate) for w in layer_weights]
            mutated_layer_weights = np.array(mutated_layer_weights)
            mutated_weights.append(mutated_layer_weights)
        self.weights = mutated_weights
        return
