#* Del_2_Test.py #*
#* ANLY 555 Fall 2022
#* Project Deliverable 2
#*
#* Due on: 10/09/2022
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
# Testing script for Deliverable 2: Source Code DataSets
#=====================================================================

#=====================================================================
# Testing DataSet Class
#=====================================================================
from Del_2_SourceCode_DataSets import (DataSet, QuantDataSet, QualDataSet,
                                        TextDataSet, TimeSeriesDataSet)

def DataSetTests():
    print("DataSet Instantiation invokes the load(), the readFromCSV(), \
the clean(), and the explore() methods.")
    data = DataSet("Sales_Transactions_Dataset_Weekly.csv")
    print("==============================================================")
    print("Check member attributes...")
    print("DataSet._filename:", data.filename)
    print("==============================================================")
    print("Check class member methods...\n")
    print("Now call DataSet.load()...")
    # data.load()
    print("Now call DataSet.readFromCSV()...")
    data.readFromCSV()
    print("Now call DataSet.clean()...")
    data.clean()
    print("Now call DataSet.explore()...")
    data.explore()
    print("\n\n")

def QuantDataSetTests():
    data = QuantDataSet("Sales_Transactions_Dataset_Weekly.csv")
    print("Check inheritence ...")
    print("QuantDataSet._filename:", data.filename)
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call QuantDataSet.load...")
    # data.load()
    print("Now call QuantDataSet.readFromCSV...")
    data.readFromCSV()
    print("QuantDataSet.clean():")
    data.clean()
    print("QuantDataSet.explore():")
    data.explore([1])
    print("\n\n")
    
def QualDataSetTests():
    data = QualDataSet("multiple_choice_responses.csv")
    print("Check inheritence ...")
    print("QualDataSet._filename:", data.filename)
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call QualDataSet.load...")
    # data.load()
    print("Now call QualDataSet.readFromCSV...")
    data.readFromCSV()
    print("QualDataSet.clean():")
    data.clean(columns=[1], fill='mode')
    data.clean(columns=[0], fill='median')
    print("QualDataSet.explore():")
    data.explore(columns=[1])
    print("\n\n")
    
def TextDataSetTests():
    data = TextDataSet("yelp.csv")
    print("Check inheritence ...")
    print("TextDataSet._filename:", data.filename)
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call TextDataSet.load...")
    # data.load()
    print("Now call TextDataSet.readFromCSV...")
    data.readFromCSV()
    print("TextDataSet.clean():")
    data.clean(columns=[4])
    print("TextDataSet.explore():")
    data.explore(columns=[4], top=15)
    print("\n\n")
    
def TimeSeriesDataSetTests():
    data = TimeSeriesDataSet("mitbih_train.csv")
    print("Check inheritence ...")
    print("TimeSeriesDataSet._filename:", data.filename)
    print("===========================================================")
    print("Check that all the member methods have been overriden...\n")
    print("Now call TimeSeriesDataSet.load...")
    # data.load()
    print("Now call TimeSeriesDataSet.readFromCSV...")
    data.readFromCSV()
    print("TimeSeriesDataSet.clean():")
    data.clean()
    print("TimeSeriesDataSet.explore():")
    data.explore(columns=[2])
    print("\n\n")
    
    
def main():
    DataSetTests()
    QuantDataSetTests()
    QualDataSetTests()
    TextDataSetTests()
    TimeSeriesDataSetTests()
    
if __name__=="__main__":
    main()
