from NN.NeuralNetwork import NeuralNetwork
import numpy as np

# First test Start
#nn = NeuralNetwork([2, 2, 1], 'tanh')
#temp = [[0, 0], [0, 1], [1, 0], [1, 1]]
#X = np.array(temp)
#y = np.array([0, 1, 1, 0])
#nn.fit(X, y)
#for i in temp:
#    print(i, nn.predict(i))
# First test End

from sklearn.datasets import load_digits
# Second test Start
#import pylab as pl
#digits = load_digits()
#print(digits.data.shape)
#pl.gray()
#pl.matshow(digits.images[0])
#pl.show()
# Second test End

# Digit identify
digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64, 20, 100, 10], 'logistic')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import LabelBinarizer
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print("Start fitting!")

nn.fit(X_train, labels_train, epochs=3000)
print("Finish training. Start predicting!")
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

