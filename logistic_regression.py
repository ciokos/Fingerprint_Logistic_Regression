import numpy as np


class LogisticRegressionClassifier:

    def __init__(self, alpha, lmbda, maxiter):
        # learning rate for gradient ascent
        self.alpha = float(alpha)
        # regularization constant
        self.lmbda = float(lmbda)
        # the maximum number of iterations through the data before stopping
        self.maxiter = int(maxiter)
        # convergence measure
        self.epsilon = 0.00001
        # the class prediction threshold
        self.threshold = 0.5

    def fit(self, X, y):

        # save data shape
        n = X.shape[1]  # the number of features
        m = X.shape[0]  # the number of instances

        # stores the model theta
        self.theta = np.zeros(n)

        # iterate through the data at most maxiter times, updating the theta for each feature
        # also stop iterating if error is less than epsilon (convergence tolerance constant)
        for iteration in range(self.maxiter):
            # calc probabilities
            probabilities = self.predict_proba(X)

            # calculate the gradient and update theta
            gw = (1.0 / m) * np.dot(X.T, (probabilities - y))
            # regularize using the lmbda term
            gw += ((self.lmbda * self.theta) / m)
            # update parameters
            self.theta -= self.alpha * gw

            # calculate the magnitude of the gradient and check for convergence
            loss = np.linalg.norm(gw)
            if self.epsilon > loss:
                break

            # print loss for each iteration
            print iteration + 1, ":", loss

    def predict_proba(self, X):
        return 1.0 / (1 + np.exp(-np.dot(X, self.theta)))

    def predict(self, X):
        y_pred = [probability > self.threshold for probability in self.predict_proba(X)]
        return np.array(y_pred)


def compute_accuracy(y_test, y_pred):
    correct = 0
    for i, prediction in enumerate(y_pred):
        if int(prediction) == y_test[i]:
            correct += 1
    return float(correct) / y_test.size
