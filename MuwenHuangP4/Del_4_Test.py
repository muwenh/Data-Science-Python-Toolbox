#* Del_4_Test.py #*
#* ANLY 555 Fall 2022
#* Project Deliverable 4
#*
#* Due on: 11/19/2022
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
# Testing script for Deliverable 4: Source Code ROC 
#=====================================================================

from Del_4_SourceCode_ROC import (DataSet, QuantDataSet, QualDataSet)

#=====================================================================
# Testing Classifier Class 
#=====================================================================
from Del_4_SourceCode_ROC import (ClassifierAlgorithm, simpleKNNClassifier)
                                        
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
from Del_4_SourceCode_ROC import Experiment

def ExperimentTests():
    print("Experiment class instantiation ...")
    classifier = simpleKNNClassifier(k=10)
    experiment = Experiment(dataset=train, labels=labels, classifiers=[classifier])
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
    dataset = QuantDataSet("heart.csv")
    dataset.readFromCSV()
    dataset.data = np.delete(dataset.data, -1, axis=1) # omit label column
    dataset.data = np.delete(dataset.data, 0, axis=0) # omit header row
    dataset.clean()
    train = dataset.data
    train = train.astype(np.float64)
    # prepare the training labels
    dataset = QualDataSet("heart.csv")
    dataset.readFromCSV()
    dataset.data = np.delete(dataset.data, slice(13), axis=1) # only keep label column
    dataset.data = np.delete(dataset.data, 0, axis=0) # omit header row
    dataset.clean(columns=[0], fill='mode')
    labels = np.reshape(dataset.data, (1, len(dataset.data)))[0]

    main()
