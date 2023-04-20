#* Del_1_SourceCode_Framework.py #*
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

class DataSet():
    """Represent an input dataset."""

    def __init__(self, filename):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file 
        """
        self._filename = filename

    def __readFromCSV(self, filename):
        """Read data from a CSV file.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__readFromCSV' for DataSet was invoked.")

    def __load(self, filename):
        """Load data from other file types, such as .txt files.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__load' for DataSet was invoked.")

    def clean(self):
        """Clean the input data."""
        print("Method 'clean' for DataSet was invoked.")

    def explore(self):
        """Explore the input data."""
        print("Method 'explore' for DataSet was invoked.")


class TimeSeriesDataSet(DataSet):
    """This class inherits from the parent class DataSet.
    Represent a dataset that contains time-series data. 
    """

    def __init__(self, filename):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        super().__init__(filename)

    def __readFromCSV(self, filename):
        """Read data from a CSV file.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__readFromCSV' for TimeSeriesDataSet was invoked.")

    def __load(self, filename):
        """Load data from other file types, such as .txt files.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__load' for TimeSeriesDataSet was invoked.")

    def clean(self):
        """Clean the input data."""
        print("Method 'clean' for TimeSeriesDataSet was invoked.")

    def explore(self):
        """Explore the input data."""
        print("Method 'explore' for TimeSeriesDataSet was invoked.")


class TextDataSet(DataSet):
    """This class inherits from the parent class DataSet.
    Represent a dataset that contains text data.
    """

    def __init__(self, filename):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        super().__init__(filename)

    def __readFromCSV(self, filename):
        """Read data from a CSV file.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__readFromCSV' for TextDataSet was invoked.")

    def __load(self, filename):
        """Load data from other file types, such as .txt files.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__load' for TextDataSet was invoked.")

    def clean(self):
        """Clean the input data."""
        print("Method 'clean' for TextDataSet was invoked.")

    def explore(self):
        """Explore the input data."""
        print("Method 'explore' for TextDataSet was invoked.")


class QuantDataSet(DataSet):
    """This class inherits from the parent class DataSet.
    Represent a dataset that contains quantitative data.
    """

    def __init__(self, filename):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        super().__init__(filename)

    def __readFromCSV(self, filename):
        """Read data from a CSV file.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__readFromCSV' for QuantDataSet was invoked.")

    def __load(self, filename):
        """Load data from other file types, such as .txt files.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__load' for QuantDataSet was invoked.")

    def clean(self):
        """Clean the input data."""
        print("Method 'clean' for QuantDataSet was invoked.")

    def explore(self):
        """Explore the input data."""
        print("Method 'explore' for QuantDataSet was invoked.")


class QualDataSet(DataSet):
    """This class inherits from the parent class DataSet.
    Represent a dataset that contains qualitative data.
    """

    def __init__(self, filename):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        super().__init__(filename)

    def __readFromCSV(self, filename):
        """Read data from a CSV file.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__readFromCSV' for QualDataSet was invoked.")

    def __load(self, filename):
        """Load data from other file types, such as .txt files.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file
        """
        print("Method '__load' for QualDataSet was invoked.")

    def clean(self):
        """Clean the input data."""
        print("Method 'clean' for QualDataSet was invoked.")

    def explore(self):
        """Explore the input data."""
        print("Method 'explore' for QualDataSet was invoked.")


class ClassifierAlgorithm():
    """Represent a general classification algorithm."""

    def __init__(self):
        """Initialize member attributes for the class."""
        pass

    def train(self):
        """Train the model."""
        print("Method 'train' for ClassifierAlgorithm was invoked.")

    def test(self):
        """Test the model."""
        print("Method 'test' for ClassifierAlgorithm was invoked.")


class simpleKNNClassifier(ClassifierAlgorithm):
    """This class inherits from the parent class ClassifierAlgorithm.
    Represent a simple KNN classification algorithm.
    """

    def __init__(self):
        """Initialize member attributes for the class."""
        super().__init__()

    def train(self):
        """Train the model."""
        print("Method 'train' for simpleKNNClassifier was invoked.")

    def test(self):
        """Test the model."""
        print("Method 'test' for simpleKNNClassifier was invoked.")


class kdTreeKNNClassifier(ClassifierAlgorithm):
    """This class inherits from parent class ClassifierAlgorithm.
    Represent a KNN search on a KD Tree.
    """

    def __init__(self):
        """Initialize member attributes for the class."""
        super().__init__()

    def train(self):
        """Train the model."""
        print("Method 'train' for kdTreeKNNClassifier was invoked.")

    def test(self):
        """Test the model."""
        print("Method 'test' for kdTreeKNNClassifier was invoked.")


class Experiment():
    """Perform cross validation and evaluate model performance."""

    def __init__(self):
        """Initialize member attributes for the class."""
        pass

    def runCrossVal(self, k):
        """Perform a k-folds cross validation.
        
        Keyword arguments:
        k -- number of folds
        """
        print("Method 'runCrossVal' was invoked.")

    def score(self):
        """Compute the score for evaluating model performance."""
        print("Method 'score' was invoked.")

    def __confusionMatrix(self):
        """Compute the confusion matrix."""
        print("Method '__confusionMatrix' was invoked.")
