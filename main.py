import sys, getopt, os
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from knn import *
from id3 import *

def main():
    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = ""
    
    # Long options
    long_options = ["model=", "k_nearest=", "lr=", "epochs=", "data=", "test_size="]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        argsDict = {}
        for currentArgument, currentValue in arguments:
            if (currentArgument in [("--" + element)[:-1] for element in long_options]):
                argsDict[currentArgument[2:]] = currentValue
            else:
                raise Exception("Argument error")

        if (argsDict["model"] == "knn"):
            k = 5

            if ("k_nearest" in argsDict):
                k = int(argsDict["k_nearest"])

            classifier = KNN(k)

            data = pd.read_csv(argsDict["data"])

            features = data.iloc[:, :-1].values
            target = data.iloc[:, -1].values

            features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=float(argsDict["test_size"]) if ("test_size" in argsDict) else 0.20)

            classifier.train(features_train, target_train)

            classifier.test(features_test, target_test)

        elif (argsDict["model"] == "id3"):
            classifier = ID3()


            
    except Exception as e:
        print(e)

    except:
        if (getopt.error):
            print(str(getopt.error))
        
        print("Error(s) occured")

if __name__ == "__main__":
    main()