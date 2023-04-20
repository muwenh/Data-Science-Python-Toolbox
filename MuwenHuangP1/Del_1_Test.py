#* Del_1_SourceCode_Test.py #*
#* ANLY 555 Fall 2022
#* Project Deliverable 1
#*
#* Due on: 9/30/2022
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
# Testing script for Deliverable 1: Source Code Framework
#=====================================================================

#=====================================================================
# Testing DataSet Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from Del_1_SourceCode_Framework import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

def DataSetTests():
    print("DataSet Instantiation invokes the __load(), the __readFromCSV(), \
the clean(), and the explore() methods.")
    data = DataSet("filename")
    print("==============================================================")
    print("Check member attributes...")
    print("DataSet._filename:", data._filename, "This is a file path.\n")
    print("==============================================================")
    print("Check class member methods...\n")
    print("Now call DataSet.__readFromCSV...")
    data._DataSet__readFromCSV("filename")
    print("Now call DataSet.__load...")
    data._DataSet__load("filename")
    print("Now call DataSet.clean()...")
    data.clean()
    print("Now call DataSet.explore()...")
    data.explore()
    print("\n\n")

def QuantDataSetTests():
    data = QuantDataSet("filename")
    print("Check inheritence ...")
    print("QuantDataSet._filename:",data._filename, "This is a file path.\n")
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call QuantDataSet.__readFromCSV...")
    data._QuantDataSet__readFromCSV("filename")
    print("Now call QuantDataSet.__load...")
    data._QuantDataSet__load("filename")
    print("QuantDataSet.clean():")
    data.clean()
    print("QuantDataSet.explore():")
    data.explore()
    print("\n\n")
    
def QualDataSetTests():
    data = QualDataSet("filename")
    print("Check inheritence ...")
    print("QualDataSet._filename:",data._filename,"This is a file path.\n")
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call QualDataSet.__readFromCSV...")
    data._QualDataSet__readFromCSV("filename")
    print("Now call QualDataSet.__load...")
    data._QualDataSet__load("filename")
    print("QualDataSet.clean():")
    data.clean()
    print("QualDataSet.explore():")
    data.explore()
    print("\n\n")
    
def TextDataSetTests():
    data = TextDataSet("filename")
    print("Check inheritence ...")
    print("TextDataSet._filename:",data._filename,"This is a file path.\n")
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call TextDataSet.__readFromCSV...")
    data._TextDataSet__readFromCSV("filename")
    print("Now call TextDataSet.__load...")
    data._TextDataSet__load("filename")
    print("TextDataSet.clean():")
    data.clean()
    print("TextDataSet.explore():")
    data.explore()
    print("\n\n")
    
def TimeSeriesDataSetTests():
    data = TimeSeriesDataSet("filename")
    print("Check inheritence ...")
    print("TimeSeriesDataSet._filename:",data._filename,"This is a file path.\n")
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call TimeSeriesDataSet.__readFromCSV...")
    data._TimeSeriesDataSet__readFromCSV("filename")
    print("Now call TimeSeriesDataSet.__load...")
    data._TimeSeriesDataSet__load("filename")
    print("TimeSeriesDataSet.clean():")
    data.clean()
    print("TimeSeriesDataSet.explore():")
    data.explore()
    print("\n\n")

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from Del_1_SourceCode_Framework import (ClassifierAlgorithm,
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
    classifier.train()
    print("simpleKNNClassifier.test():")
    classifier.test()
    print("\n\n")

def kdTreeKNNClassifierTests():
    print("Check inheritence ...")
    classifier = kdTreeKNNClassifier()
    print("==============================================================")
    print("Check that all the member methods have been overriden...\n")
    print("kdTreeKNNClassifier.train():")
    classifier.train()
    print("kdTreeKNNClassifier.test():")
    classifier.test()
    print("\n\n")

#=====================================================================
# Testing Classifier Class 
# (Not meant to be called, but will show instantiation, attributes,
# and member methods)
#=====================================================================
from Del_1_SourceCode_Framework import Experiment

def ExperimentTests():
    print("Experiment class instantiation ...")
    experiment = Experiment()
    print("==============================================================")
    print("Check class member methods...\n")
    print("Experiment.runCrossVal(numFolds):")
    experiment.runCrossVal("numFolds")
    print("Experiment.score():")
    experiment.score()
    print("Experiment.__confusionMatrix():")
    experiment._Experiment__confusionMatrix()
    print("\n\n")
    
    
def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    ClassifierAlgorithmTests()
    simpleKNNClassifierTests()
    kdTreeKNNClassifierTests()
    ExperimentTests()
    
if __name__=="__main__":
    main()
