#* Del_2_SourceCode_DataSets.py #*
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

import csv
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import matplotlib.pyplot as plt 
import os
from wordcloud import WordCloud

class DataSet():
    """Represent an input dataset."""

    def __init__(self, filename):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        filename -- direct or relevant path of the input file 
        """
        self.filename = filename

    def readFromCSV(self):
        """Read data from a CSV file."""
        with open(self.filename, encoding='utf-8', mode='r') as f:
            csvreader = csv.reader(f, delimiter=',')
            self.data = np.array([l for l in csvreader])

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if len(self.data[i][j]) == 0:
                    self.data[i][j] = 'NaN'  # Assign missing value to NaN

    def load(self):
        """Get file path and file type from user input."""
        filename = input("Enter your dataset path: ")
        self.filename = filename
        filetype = input("Enter your dataset type: ")

    def clean(self):
        """Clean the input data."""
        print("Method 'clean' for DataSet was invoked.")

    def explore(self):
        """Explore the input data."""
        print("Method 'explore' for DataSet was invoked.")


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

    def readFromCSV(self):
        """Read data from a CSV file."""
        with open(self.filename, encoding='utf-8', mode='r') as f:
            csvreader = csv.reader(f, delimiter=',')
            self.data = np.array([l for l in csvreader])

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if len(self.data[i][j]) == 0:
                    self.data[i][j] = 'NaN'  # Assign missing value to NaN

    def load(self):
        """Get file path and file type from user input."""
        filename = input("Enter your dataset path: ")
        self.filename = filename
        filetype = input("Enter your dataset type: ")
        if filetype != "quant":
            raise TypeError('invalid dataset type')

    def clean(self):
        """Clean the input data."""
        mean = {}  # initialize a dictionary to store the mean of each column
        for c in range(1, len(self.data[0])):
            column = self.data[:, c]
            for r in range(1, len(column)):
                if column[r] == 'NaN':
                    continue
                try:
                    float(column[r])  # check if the data type is float
                except:
                    raise ValueError("not numeric value")
            column = column[1:]
            column = column[column != 'NaN']
            column = [float(i) for i in column]
            mean[c] = np.mean(column)  # compute mean
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] == 'NaN':
                    self.data[i][j] = mean[j]  # fill in missing values with the mean

    def explore(self, columns):
        """Explore the input data.
        
        Keyword arguments:
        columns -- a list of column indexes to be explored
        """
        for col in columns:
            column = self.data[:, col]
            column = column[1:]
            plt.hist(column, bins='auto')  # histogram
            plt.title('Histogram of Column '+str(col))
            plt.show()
            column = [float(i) for i in column]
            plt.boxplot(column, vert=False)  # boxplot
            plt.title('Boxplot of Column '+str(col))
            plt.show() 


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

    def readFromCSV(self):
        """Read data from a CSV file."""
        with open(self.filename, encoding='utf-8', mode='r') as f:
            csvreader = csv.reader(f, delimiter=',')
            self.data = np.array([l for l in csvreader])

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if len(self.data[i][j]) == 0:
                    self.data[i][j] = 'NaN'  # Assign missing value to NaN

    def load(self):
        """Get file path and file type from user input."""
        filename = input("Enter your dataset path: ")
        self.filename = filename
        filetype = input("Enter your dataset type: ")
        if filetype != "qual":
            raise TypeError('invalid dataset type')

    def clean(self, columns, fill):
        """Clean the input data.
        
        Keyword arguments:
        columns -- a list of column indexes to be cleaned
        fill -- method of filling in missing values, can be 'mode' or 'median'
        """
        def get_mode(array):
            """Get the mode of an array.
            
            Keyword arguments:
            array -- an array of qualitative elements
            """
            count = {}
            for i in array:
                count[i] = count.get(i, 0) + 1  # count each element
            return sorted(count.keys(), key=lambda x: -count[x])[0]  # get the element with the largest count

        if fill == 'mode':
            mode = {}
            for c in columns:
                column = self.data[:, c]
                column = column[column != 'NaN']
                mode[c] = get_mode(column)  # get the mode

            for i in range(len(self.data)):
                for j in columns:
                    if self.data[i][j] == 'NaN':
                        self.data[i][j] = mode[j]  # fill in missing values with the mode

        if fill == 'median':
            median = {}
            for c in columns:
                column = self.data[:, c]
                for r in range(2, len(column)):
                    if column[r] == 'NaN':
                        continue
                    try:
                        int(column[r])  # check if element can be converted to integer
                    except:
                        raise ValueError("can not convert to integer")
                column = column[2:]
                column = column[column != 'NaN']
                column = [int(item) for item in column]
                median[c] = int(np.median(column))  # get the median

            for i in range(len(self.data)):
                for j in columns:
                    if self.data[i][j] == 'NaN':
                        self.data[i][j] = median[j]  # fill in missing values with the median

    def explore(self, columns):
        """Explore the input data.
        
        Keyword arguments:
        columns -- a list of column indexes to be explored
        """
        for col in columns:
            column = self.data[:, col]
            column = column[2:]
            labels = []  # store the labels
            size = []  # store the sizes
            count = {} 
            for i in column:
                count[i] = count.get(i, 0) + 1
            for element in count:
                labels.append(element)  # get label
                size.append(count[element] / len(column) * 100)  # get size
            plt.hist(column, bins='auto')  # histogram
            plt.title('Bar Plot of Column '+str(col))
            plt.show()
            plt.pie(size, labels=labels, autopct='%1.1f%%', startangle=90)  # pie chart
            plt.title('Pie Chart of Column '+str(col))
            plt.show()


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

    def readFromCSV(self):
        """Read data from a CSV file."""
        with open(self.filename, encoding='utf-8', mode='r') as f:
            csvreader = csv.reader(f, delimiter=',')
            self.data = np.array([l for l in csvreader])

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if len(self.data[i][j]) == 0:
                    self.data[i][j] = 'NaN'  # Assign missing value to NaN

    def load(self):
        """Get file path and file type from user input."""
        filename = input("Enter your dataset path: ")
        self.filename = filename
        filetype = input("Enter your dataset type: ")
        if filetype != "text":
            raise TypeError('invalid dataset type')

    def clean(self, columns):
        """Clean the input data.
        
        Keyword arguments:
        columns -- a list of column indexes to be cleaned
        """
        def pipeline(text, stopwords, stemmer):
            """Define a pipeline for cleaning the documents.
            
            Keyword arguments:
            text -- a document to be cleaned
            stopwords -- a list of stop words to be removed
            stemmer -- a stemmer for stemming the words
            """
            tokens = []
            for word in nltk.word_tokenize(text):  # tokenize the document
                if word.lower() not in stopwords:  # remove stopwords
                    tokens.append(word.lower())
            clean_tokens = []
            for token in tokens:
                if token.encode('utf-8').isalpha():  # remove numbers and punctuations
                    clean_tokens.append(token)
            stems = [stemmer.stem(t) for t in clean_tokens]  # stem the words
            return stems

        # nltk.download('stopwords')
        # nltk.download('punkt')

        stopwords = nltk.corpus.stopwords.words('english')  # get stopwords
        stemmer = SnowballStemmer("english")  # instantiate the stemmer
        for c in columns:
            column = self.data[:, c]
            for r in range(1, len(column)):
                self.data[r][c] = ' '.join(pipeline(self.data[r][c], stopwords, stemmer))

    def explore(self, columns, top):
        """Explore the input data.
        
        Keyword arguments:
        columns -- a list of column indexes to be explored
        top -- a number of the words with the highest frequencies 
        """
        for col in columns:
            column = self.data[:, col]
            count = {}  # store word frequencies
            word_cloud = ''  # store the content for creating word cloud
            for doc in column:
                word_cloud += doc + ' '
                for word in doc.split(' '):
                    count[word] = count.get(word, 0) + 1  # get the counts of the words
            top_words = sorted(count.keys(), key=lambda x: -count[x])[:top]  # get the top words
            top_words_count = []
            for word in top_words:
                top_words_count.append(count[word])  # get the counts of the top words
            plt.barh(top_words, top_words_count)  # bar plot of the top words frequencies
            plt.title('Top '+str(top)+' Words in Column '+str(col))
            plt.show()
            wordcloud = WordCloud(background_color ='white').generate(word_cloud)  # word cloud
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.show()


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

    def readFromCSV(self):
        """Read data from a CSV file."""
        with open(self.filename, encoding='utf-8', mode='r') as f:
            csvreader = csv.reader(f, delimiter=',')
            self.data = np.array([l for l in csvreader])

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                if len(self.data[i][j]) == 0:
                    self.data[i][j] = 'NaN'  # Assign missing value to NaN

    def load(self):
        """Get file path and file type from user input."""
        filename = input("Enter your dataset path: ")
        self.filename = filename
        filetype = input("Enter your dataset type: ")
        if filetype != "time series":
            raise TypeError('invalid dataset type')

    def clean(self):
        """Clean the input data."""
        def median_filter(data, index, size=5):
            """Find the median of a list with size size.
            
            Keyword arguments:
            data -- a list of data elements
            index -- the index for finding the median
            size -- the size of the filter
            """
            left_s = (size + 1) // 2  # filter size on the left of the index
            right_s = size - left_size # filter size on the right of the index
            if index - left_s < 0:  # check if left size is out of range
                left_s = index - 0  # new possible left size 
                right_s = size - left_s  # new right size
            if index + right_s >= len(data):  # check if right size is out of range
                right_s = len(data) - 1 - index  # new possible right size
                left_s = size - right_s  # new left size
            window = []  # initialize a median filter window
            left = index - 1  # left pointer
            right = index + 1  # right pointer
            while left >= 0 and left_s > 0:
                if data[left] != 'NaN':
                    window.append(float(data[left]))
                    left -= 1
                    left_s -= 1
                else:
                    left -= 1
            # right_s += left_s
            while right < len(data) and right_s > 0:
                if data[right] != 'NaN':
                    window.append(float(data[right]))
                    right += 1
                    right_s -= 1
                else:
                    right += 1
            return np.median(sorted(window))

        for c in range(len(self.data[0])):
            column = self.data[:, c]
            for r in range(len(self.data)):
                if column[r] == 'NaN':
                    index = r
                    column[r] = median_filter(column, r, size=5)  # fill the missing value with median filter
            self.data[:, c] = column # refresh the column after filling missing values

    def explore(self, columns):
        """Explore the input data.
        
        Keyword arguments:
        columns -- a list of column indexes to be explored
        """
        for col in columns:
            column = self.data[:, col]
            column = [float(i) for i in column]
            plt.hist(column, bins='auto')  # histogram
            plt.title('Histogram of Column '+str(col))
            plt.show()
            plt.boxplot(column, vert=False)  # boxplot
            plt.title('Boxplot of Column '+str(col))
            plt.show() 


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

