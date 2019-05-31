from image_preprocessing import *
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegressionClassifier, compute_accuracy

X = np.load("new_features.npy")
y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

alpha = 0.01
lmbda = 0
maxiter = 10000

# create the logistic regression classifier using the training data
LRC = LogisticRegressionClassifier(alpha, lmbda, maxiter)

# fit the model to the loaded training data
print "Fitting the training data...\n"
LRC.fit(X_train, y_train)

# predict the results for the test data
print "\nGenerating probability prediction for the test data...\n"
y_pred = LRC.predict(X_test)

# print simple precision metric to the console
print('Accuracy:  {0:.2f}%'.format(compute_accuracy(y_test, y_pred) * 100))
