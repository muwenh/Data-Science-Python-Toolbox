#* Del_5_Test.py #*
#* ANLY 555 Fall 2022
#* Project Deliverable 5
#*
#* Due on: 12/6/2022
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
# Testing script for Deliverable 5: Source Code HeterogeneousDataSet, LSH
#=====================================================================

#=====================================================================
# Testing DataSet Class
#=====================================================================
from Del_5_SourceCode_Heterogeneous_LSH import (DataSet, QuantDataSet, QualDataSet,
                               TextDataSet, TimeSeriesDataSet, HeterogenousDataSet)

def HeterogenousDataSetTests():
    print("Check inheritence ...")
    print("HeterogenousDataSet Instantiation invokes the load(), the clean(), the explore(), and the select() methods.")
    data = HeterogenousDataSet(["Sales_Transactions_Dataset_Weekly.csv", "mitbih_train.csv"])
    print("==============================================================")
    print("Check member attributes...")
    print("HeterogenousDataSet.datasets:", data.datasets)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Now call HeterogenousDataSet.load()...")
    data.load(["QuantDataSet", "TimeSeriesDataSet"])
    print("Now call HeterogenousDataSet.clean()...")
    data.clean()
    print("Now call HeterogenousDataSet.explore()...")
    data.explore()
    print("Now call HeterogenousDataSet.select()...")
    print(data.select(0))
    print("\n\n")

#=====================================================================
# Testing Classifier Class 
#=====================================================================
from Del_5_SourceCode_Heterogeneous_LSH import (ClassifierAlgorithm, lshKNNClassifier)
                                        
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

def lshKNNClassifierTests():
    print("Check inheritence ...")
    classifier = lshKNNClassifier(k=10, l=32)
    print("==============================================================")
    print("Check that all the member methods have been overriden...\n")
    print("lshKNNClassifier.train():")
    classifier.train(trainingData=train, trueLabels=labels)
    print("lshKNNClassifier.test():")
    classifier.test(testData=train)
    print("\n\n")

#=====================================================================
# Testing Experiment Class 
#=====================================================================
from Del_5_SourceCode_Heterogeneous_LSH import Experiment

def ExperimentTests():
    print("Experiment class instantiation ...")
    classifier = lshKNNClassifier(k=10, l=32)
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
    HeterogenousDataSetTests()
    ClassifierAlgorithmTests()
    lshKNNClassifierTests()
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
