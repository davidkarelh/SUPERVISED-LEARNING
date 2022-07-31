from numpy import array

class KNN:
    k : int
    features : array
    target : array

    def __init__(self, k : int = 5) -> None:
        self.k = k    
        print("KNN algorithm used")
    
    def train(self, features : array, target : array):
        if (len(features) != len(target)):
            raise Exception("Lengths of target and features are different")

        self.features = features
        self.target = target

    def test(self, features : array, target : array):
        if (len(features) != len(target)):
            raise Exception("Lengths of target and features are different")

        correct = 0
        for i in range(len(features)):
            if (self.__calculateKNearestNeighbor(features[i]) == target[i]):
                correct += 1
        
        print(f"Correct/Total Test = {correct}/{len(features)}")
        
    def __calculateKNearestNeighbor(self, feature : array):
        localArray = []

        for i in range(len(self.features)):
            localArray.append((self.__calculateDistance(self.features[i], feature), self.target[i]))
        
        localArray.sort(key = lambda x: x[0])
        localDict = {}

        for i in range(self.k):
            if (localArray[i][1] in localDict):
                localDict[localArray[i][1]] += 1
            else:
                localDict[localArray[i][1]] = 1
        
        maxAppearance = -1
        ret = ""

        for key, value in localDict.items():
            if (value > maxAppearance):
                maxAppearance = value
                ret = key
            
        return ret



    def __calculateDistance(self, feature1: array, feature2 : array):
        if (len(feature1) != len(feature2)):
            raise Exception("Lengths of feature1 and feature2 are different")

        ret = 0
        for i in range(len(feature1)):
            ret += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i])
        return ret


