from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from KnnAlgo import getNeighbors,getAccuracy,getResponse

x = genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", delimiter=',',
               dtype='unicode')
#
# print(my_data)
y = []
for i in range(0, len(x)):
    if x[i][-1] == 'Iris-setosa':
        y.append(0)
    if x[i][-1] == 'Iris-versicolor':
        y.append(1)
    if x[i][-1] == 'Iris-virginica':
        y.append(2)
x1 = x[:, 0:4]

X_train,X_test,y_train,y_test=train_test_split(x1,y,test_size=0.3,stratify=y)
X_train= X_train.astype('float64')
X_test = X_test.astype('float64')


def main():
    predictions = []
    k = 3
    for x in range(len(X_test)):
        neighbors = getNeighbors(X_train, X_test[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(X_test[x][-1]))
    accuracy = getAccuracy(X_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == "__main__":
    main()

