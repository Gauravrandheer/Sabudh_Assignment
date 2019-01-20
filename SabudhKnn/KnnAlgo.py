import math
import operator


def euclideanDistance(Xtrain, Xtest, length):
    distance = 0
    for x in range(length):
        distance += pow((Xtrain[x] - Xtest[x]), 2)
    return math.sqrt(distance)


def getNeighbors(Xtrain, Xtest, k):# k =3
    distances = []
    length = len(Xtest) - 1
    for x in range(len(Xtrain)):
        dist = euclideanDistance(Xtest, Xtrain[x], length)
        distances.append((Xtrain[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    rep = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in rep:
            rep[response] += 1
        else:
            rep[response] = 1
    sortedVotes = sorted(rep.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(Xtest, predictions):
    correct = 0
    for x in range(len(Xtest)):
        if Xtest[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(Xtest))) * 100.0




