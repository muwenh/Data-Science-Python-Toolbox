#* Del_3_Test.py #*
#* ANLY 555 Fall 2022
#* Project Deliverable 3
#*
#* Due on: 10/31/2022
#* Author(s): Muwen Huang
#*
#*
#* In accordance with the class policies and Georgetown's
#* Honor Code, I certify that, with the exception of the
#* class resources and those items noted below, I have neither 
#* given nor received any assistance on this project other than 
#* the TAs, professor, textbook and teammates.
#*

#=====================================================================
# Testing script for Deliverable 3: Source Code Classifier, Experiment
#=====================================================================

from Del_3_SourceCode_ClassifierExperiment import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

#=====================================================================
# Testing Classifier Class 
#=====================================================================
from Del_3_SourceCode_ClassifierExperiment import (ClassifierAlgorithm,
                                        simpleKNNClassifier, kdTreeKNNClassifier)
                                        
def ClassifierAlgorithmTests():
    print("ClassifierAlgorithm Instantiation....")
    classifier = ClassifierAlgorithm()
    print("==============================================================")
    print("Check class member methods...\n")
    print("ClassifierAlgorithm.train():")
    classifier.train()
    print("ClassifierAlgorithm.test():")
    classifier.test()
    print("\n\n")

def simpleKNNClassifierTests():
    print("Check inheritence ...")
    classifier = simpleKNNClassifier()
    print("==============================================================")
    print("Check that all the member methods have been overriden...\n")
    print("simpleKNNClassifier.train():")
    classifier.train(trainingData=train, trueLabels=labels)
    print("simpleKNNClassifier.test():")
    classifier.test(testData=train)
    print("\n\n")

#=====================================================================
# Testing Experiment Class 
#=====================================================================
from Del_3_SourceCode_ClassifierExperiment import Experiment

def ExperimentTests():
    print("Experiment class instantiation ...")
    classifier1 = simpleKNNClassifier(k=8)
    classifier2 = simpleKNNClassifier(k=10)
    classifier3 = simpleKNNClassifier(k=12)
    experiment = Experiment(dataset=train, labels=labels, classifiers=[classifier1, classifier2, classifier3])
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.runCrossVal():")
    experiment.runCrossVal()
    print("Experiment.score():")
    experiment.score()
    print("Experiment.confusionMatrix():")
    experiment.confusionMatrix()
    print("\n\n")
    
def main():
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    ExperimentTests()


if __name__=="__main__":
    import numpy as np
    # prepare the training data
    iris = QuantDataSet("iris.csv")
    iris.readFromCSV()
    iris.data = np.delete(iris.data, -1, axis=1) # omit label column
    iris.data = np.delete(iris.data, 0, axis=0) # omit header row
    iris.clean()
    train = iris.data
    train = train.astype(np.float64)
    # prepare the training labels
    iris = QualDataSet("iris.csv")
    iris.readFromCSV()
    iris.data = np.delete(iris.data, [0,1,2,3], axis=1) # only keep label column
    iris.data = np.delete(iris.data, 0, axis=0) # omit header row 
    iris.clean(columns=[0], fill='mode')
    labels = np.reshape(iris.data, (1, len(iris.data)))[0]

    main()
