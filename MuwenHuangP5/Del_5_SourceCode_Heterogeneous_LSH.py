#* Del_5_SourceCode_Heterogeneous_LSH.py #*
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

import csv
import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import matplotlib.pyplot as plt 
import os
from wordcloud import WordCloud
import heapq

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


class HeterogenousDataSet(DataSet):
    """Represent a list of datasets."""

    def __init__(self, list_of_datasets):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        list_of_datasets -- a list of dataset paths 
        """
        self.datasets = list_of_datasets

    def load(self, type_of_datasets):
        """Load the datasets.
        
        Keyword arguments:
        type_of_datasets -- a list of dataset types 
        """
        self.heterogenous = [] # store the datasets
        self.type = type_of_datasets # types of the datasets

        # For each dataset in the list
        for i in range(len(self.datasets)):

            if self.type[i] == 'TimeSeriesDataSet':
                data = TimeSeriesDataSet(self.datasets[i])
                data.readFromCSV()
                self.heterogenous.append(data)

            elif self.type[i] == 'TextDataSet':
                data = TextDataSet(self.datasets[i])
                data.readFromCSV()
                self.heterogenous.append(data)

            elif self.type[i] == 'QuantDataSet':
                data = QuantDataSet(self.datasets[i])
                data.readFromCSV()
                self.heterogenous.append(data)

            elif self.type[i] == 'QualDataSet':
                data = QualDataSet(self.datasets[i])
                data.readFromCSV()
                self.heterogenous.append(data)

            else:
                raise AttributeError("The dataset must be in one of the following types: TimeSeriesDataSet, TextDataSet, QuantDataSet, and QualDataSet.")

    def clean(self):
        """Clean each dataset in the list."""
        for j in range(len(self.heterogenous)):
            self.heterogenous[j].clean()

    def explore(self):
        """Explore each dataset in the list."""
        for j in range(len(self.heterogenous)):
            self.heterogenous[j].explore(columns=[1]) # columns can be more than one

    def select(self, index_datasets):
        """Select one of the constituent datasets.
        
        Keyword arguments:
        index_datasets -- the index of the selected data set
        """
        try:
            return (self.datasets[index_datasets])
        except:
            print('Please put in the index of the selected dataset.')


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

    def __init__(self, k=10):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        k -- the number of the closest training samples (default: 10)
        """
        super().__init__()
        self.k = k

    def train(self, trainingData, trueLabels):
        """Store the training data and labels member attributes.
        
        Keyword arguments:
        trainingData -- data to train the classifier
        trueLabels -- training labels
        """
        self.X_train = trainingData
        self.y_train = trueLabels

    def __get_mode(self, array):
            """Get the mode of an array.
            
            Keyword arguments:
            array -- an array of qualitative elements
            """
            item_count = {}
            for i in array:
                item_count[i] = item_count.get(i, 0) + 1  # count each element
            mode = sorted(item_count.keys(), key=lambda x: -item_count[x])[0]  # get the mode
            return mode, item_count

    def test(self, testData):
        """Test and return the predicted labels.
        
        Keyword arguments:
        testData -- data to test the classifier
        """                                                                             # step count       # space count
        k = self.k                                                                      # 1 op             # 1
        # store the prediction result
        self.prediction = []                                                            # 1 op             # n
        self.prediction_probs = []                                                      # 1 op             # n
        # iterate each row
        for row in testData:                                                            # 2n ops
            # compute euclidean distance between test data and every train data point 
            distance = [np.sqrt(np.sum((self.X_train - row) ** 2, axis=1))]             # 3m+1 ops         # n
            # extract top k indexes
            topk_index = np.argsort(distance)[0][:k]                                    # k*log(k) ops     # k
            # get top k labels by indexes
            topk_labels = [self.y_train[index] for index in topk_index]                 # 4k ops           # k
            # get mode of the labels and append to result
            mode_label, item_count = self.__get_mode(topk_labels)                       # k*log(k) ops
            self.prediction.append(mode_label)                                          # 1 op
            self.prediction_probs.append({k: v / total for total in (sum(item_count.values()),) for k, v in item_count.items()})
                                                                                        # 1 op
        # return predicted labels
        return self.prediction                                                          # 1 op

# Dimension of test data: n * m. 
# k is the number of the closest training samples.
 
# T(n) = 4 + 2n * (3m + 1 + 2k*log(k) + 4k + 2)
# Tight-fit upperbound: O(n^2 * log(n))

# S(n) = 2n + n + 2k + 1
# Tight-fit upperbound: O(n + k)


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


class lshKNNClassifier(ClassifierAlgorithm):
    """This class inherits from parent class ClassifierAlgorithm.
    Represent a KNN search with random projection based LSH.
    """

    def __init__(self, k, l):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        k -- the number of the closest training samples
        l -- length of binary code
        """
        super().__init__()
        self.k = k
        self.l = l

    def __random_projection_hash(self, dataset_without_label):
        """Perform random projection based LSH.
        
        Keyword arguments:
        dataset_without_label -- dataset without the label column
        """                                                                # step count
        # generate m*l random matrix A from standard normal distribution
        self.projections = np.random.randn(self.col, self.l)               # m * l ops
        # store n l-bits binary codes
        self.hash_table = list()                                           # 1 op
        # generate the binary codes for each row
        for row in dataset_without_label:                                  # n ops
            binary_code = (np.dot(row, self.projections) >= 0).astype(int) # m * l ops
            self.hash_table.append(''.join(binary_code.astype('str')))     # 3 ops
        return self.hash_table                                             # 1 op

    def train(self, trainingData, trueLabels):
        """Return the hash indexes of the training data.
        
        Keyword arguments:
        trainingData -- data to train the classifier
        trueLabels -- training labels
        """                                                             # step count          # space count
        self.trainDS = trainingData                                     # 1 op                # n 
        self.trainL = trueLabels                                        # 1 op                # n
        self.row, self.col = self.trainDS.shape                         # 2 ops               # n + m
        self.train_hash = self.__random_projection_hash(self.trainDS)   # n*m*l+3n+ml+2 ops   # n
        return self.train_hash                                          # 1 op

# Dimension of training data: n * m
# l: length of binary code

# T(n) = n*m*l + 3n + ml + 7
# Tight-fit upperbound: O(n * m * l)

# S(n) = 4n + m
# Tight-fit upperbound: O(n + m)

    def __distance(self, binary_code1, binary_code2):
        """Compute the Hamming distance between two binary codes.
        
        Keyword arguments:
        binary_code1 -- l-bits binary code of the test row
        binary_code2 -- l-bits binary code of the train row
        """                                                             # step count
        dist = 0                                                        # 1 op
        for n in range(len(binary_code1)):                              # l ops
            if binary_code1[n] != binary_code2[n]:                      # 3 ops
                dist += 1                                               # 1 op
        return dist                                                     # 1 op

    def __find_neighbors(self, test_row):
        """Find k nearest neighbors.
        
        Keyword arguments:
        test_row -- l-bits binary code of the test row
        """                                                             # step count
        distance = list()                                               # 1 op
        i = 0                                                           # 1 op
        for train_row in self.train_hash:                               # n ops
            dist = self.__distance(test_row, train_row)                 # 4l + 2 ops
            distance.append((self.trainL[i], dist))                     # 1 op
            i += 1                                                      # 1 op
        # sort the labels by distances
        distance.sort(key=lambda tup: tup[1])                           # n*log(n) ops
        neighbor = list()                                               # 1 op
        # extract the nearest k neighbors
        for j in range(self.k):                                         # k ops
            neighbor.append(distance[j][0])                             # 1 op
        return neighbor                                                 # 1 op

    def test(self, testData):
        """Return the prediction result and scores.
        
        Keyword arguments:
        testData -- data to test the classifier
        """                                                          # step count             # space count
        # store the prediction result
        self.prediction = []                                         # 1 op                   # n
        # transform the test data
        testData_hash = self.__random_projection_hash(testData)      # n*m*l+3n+ml+2 ops      # n
        # for each test row
        for row in testData_hash:                                    # n ops
            # get the nearest k neighbors
            neighbor = self.__find_neighbors(row)                    # nlogn+4nl+4n+k+4 ops   # k
            # count the labels
            lab, count = np.unique(neighbor, return_counts=True)     # klogk ops              # 2k
            # get the mode
            predicted_class = lab[np.argmax(count)]                  # k ops                  # k
            # append the predicted label to the result list
            self.prediction.append(predicted_class)                  # 1 op
        # return predicted labels 
        return self.prediction                                       # 1 op

# Dimension of test data: n * m
# l: length of binary code
# k: number of nearest neighbors

# T(n) = n^2 * log(n) + 4n^2 * l + 4n^2 + nml + ml + n*k*log(k) + 2nk + 8n + 4
# Tight-fit upperbound: O(n^2 * log(n))

# S(n) = 2n + 4k
# Tight-fit upperbound: O(n + k)

# For my implementation, the time complexity does not improve compared to the simple KNN algorithm. 
# Mainly due to the inefficient sorting method. In theory, the LSH KNN algorithm should have an 
# improved time complexity compared to the simple KNN algorithm.

# The accuracy of LSH KNN is less than the accuracy of simple KNN. This is acceptable since LSH
# algorithms often require lots of tuning. Perhaps the current parameters still need to be adjusted
# to get the optimal result.


class Experiment():
    """Perform cross validation and evaluate model performance."""

    def __init__(self, dataset, labels, classifiers):
        """Initialize member attributes for the class.
        
        Keyword arguments:
        dataset -- represent a training dataset
        labels -- training labels
        classifiers -- a list of classifiers
        """
        self.data = dataset  # store training data
        self.labels = labels  # store training labels
        self.classifiers = classifiers  # store classifiers

    def runCrossVal(self, k=5):
        """Perform a k-folds cross validation.
        
        Keyword arguments:
        k -- number of folds (default: 5)
        """
        self.predLabels = []  # store the predicted labels
        self.true = []  # store the true test labels
        size = len(self.data) // k  # the size of each test set
        # for each fold
        for i in range(1, k+1):
            preds = []  # store predicted labels for each fold
            start, end = size*(i-1), size*i
            X_test = self.data[start:end, :]  # get test set
            y_test = self.labels[start:end]  # get test labels
            # remove validation set from training set
            X_train = np.delete(self.data, [i for i in range(start, end)], axis=0)
            # remove validation labels from training labels
            y_train = np.delete(self.labels, [i for i in range(start, end)])

            # for all classifiers
            for classifier in self.classifiers:
                classifier.train(trainingData=X_train, trueLabels=y_train)
                pred = classifier.test(testData=X_test)
                preds.append(pred)
                # # produce ROC plot
                # print("Generating ROC curves")
                # self.ROC(classifier.prediction_probs, y_test)

            # store cv result
            self.predLabels.append(preds)
            self.true.append(y_test)

        return self.predLabels, self.true

    def score(self):
        """Compute the accuracy of each classifier for every cross validation step.""" # step count     # space count
        self.scores = []  # store accuracies                                           # 1 op           # m * (k + 1)
        for i in range(len(self.predLabels)):                                          # 2k ops
            accuracies = []  # store accuracy for the ith fold                         # 1 op           # m
            preds = self.predLabels[i]  # predicted labels                             # 2 ops          # m * n
            true = self.true[i]  # true labels                                         # 2 ops          # m * n
            for j in range(len(self.classifiers)):                                     # 2m ops
                pred_j = preds[j]  # predicted labels for jth classifier               # 2 ops          # n
                numCorrect = 0  # TF + TN                                              # 1 op           # 0 ~ n
                for n in range(len(pred_j)):                                           # 2n ops
                    if pred_j[n] == true[n]:                                           # 3 ops
                        numCorrect += 1                                                # 2 ops
                acc = numCorrect / len(pred_j)  # compute accuracy                     # 2 ops          # 1
                accuracies.append(acc)                                                 # 1 op
            self.scores.append(accuracies)                                             # 1 op
        self.scores = np.array(self.scores)  # convert to numpy array                  # 2 ops
        # average accuracy for each classifier
        avg_accuracy = np.sum(self.scores, axis=0) / len(self.predLabels)              # k * m ops      # m
        # append average accuracies
        self.scores = np.append(self.scores, [avg_accuracy], axis=0)                   # 2 ops
        self.__printScore()  # present as a table
        return self.scores                                                             # 1 op

# k: number of cross validation sets
# m: number of classifiers
# n : length of test data

# T(n) = 1 + 2k * (6 + 2m * (6 + 2n * 5)) + k * m + 5
# Tight-fit upperbound: O(k * (m + n))

# S(n) = m * (k+1) + 2m + 1 + (2m+2) * n
# The worst case is when the number of folds equals the length of test data. That k == n.
# Tight-fit upperbound: O(m * n)  

    def __printScore(self):
        """Present the accuracy result as a table."""
        classifier_name = []  # header row
        for i in range(len(self.classifiers)):
            classifier_name.append(f'Classifier {i+1}')
        header = '| Cross Validation Step ' + '| '
        for name in classifier_name:
            header += name + ' | '
        print(header)

        for i in range(len(self.scores)):
            if i >= len(self.predLabels):
                row = f'| Average Accuracy' + ' ' * (len('| Cross Validation Step |') - len('| Average Accuracy ')) + '|'
            else:
                row = f'| Cross Validation {i+1}' + ' ' * (len('| Cross Validation Step |') - len(f'| Cross Validation {i+1} ')) + '|'
            for j in range(len(self.scores[0])):
                row += ' ' + str(np.round(self.scores[i][j], 8)) + ' ' * (12 - len(str(np.round(self.scores[i][j], 8)))) + ' |'
            print(row) 

    def confusionMatrix(self):
        """Compute and display a confusion matrix for each classifier."""                      # step count      # space count
        unique_label = []                                                                      # 1 op            # y
        for label in set(self.labels):                                                         # 2y ops
            unique_label.append(label)                                                         # 1 op
        unique_label = np.array(unique_label)                                                  # 2 ops
        ConfusionMatrix = []                                                                   # 1 op            # m * y * y
        # for every classifier
        for i in range(len(self.classifiers)):                                                 # 2m ops
            matrix = [[0 for _ in range(len(unique_label))] for _ in range(len(unique_label))] # y * y  ops      # y * y
            for j in range(len(self.predLabels)):                                              # 2k ops
                pred = self.predLabels[j]                                                      # 2 ops           # m * n
                true = self.true[j]                                                            # 2 ops           # m * n
                pred_i = pred[i]                                                               # 2 ops           # n
                for row in range(len(true)):                                                   # 2n ops
                    # get column index
                    pred_loc = np.where(unique_label == pred_i[row])[0][0]                     # y ops           # 1
                    # get row index
                    true_loc = np.where(unique_label == true[row])[0][0]                       # y ops           # 1
                    matrix[true_loc][pred_loc] += 1                                            # 3 ops
            ConfusionMatrix.append(np.array(matrix))                                           # 2 ops
        
        # print a confusion matrix for each classifier
        max_len = max([len(label) for label in unique_label])                                  # y ops           # 1
        header = ' ' * (max_len + 1) + ' '.join(unique_label)                                  # y ops           # 1
        for i in range(len(ConfusionMatrix)):                                                  # 2m ops
            print(f'Confusion matrix for classifier {i+1}:')                                    # 1 op
            print(header)                                                                      # 1 op
            for j in range(len(unique_label)):                                                 # 2y ops          
                row = ''                                                                       # 1 op            # 1
                for num in ConfusionMatrix[i][j]:                                              # 2y ops
                    row += ' '  + str(num)                                                     # 4 ops
                print(unique_label[j] + ' ' * (max_len - len(unique_label[j])) + row)          # 1 op
        
        return ConfusionMatrix                                                                 # 1 op

# k: number of cross validation sets
# m: number of classifiers
# n : length of test data
# y : number of unique labels

# T(n) = 1 + 4y + 3 + 2m * (y^2 + 2k * (6 + 2n * (2y + 3)) + 2) + 2y + 2m * (2 + 2y * (1 + 6y + 1)) + 1
# Tight-fit upperbound: O(m * k * n * y)

# S(n) = y + m * y^2 + y^2 + 2m * n + n + 5
# Tight-fit upperbound: O(m * y^2)

    def sortReverse(self, alist):
        """Sort an array in decreasing order with heap sort.
        
        Keyword arguments:
        alist -- an array of values
        """                                                                             # step count       # space count
        h = []                                                                          # 1 op             # 1
        for idx, val in enumerate(alist):                                               # n ops            # 0
            heapq.heappush(h, (-val, idx))  # -val for reverse sort                     # log(n) ops       # n
        result = [heapq.heappop(h) for i in range(len(h))]                              # n*log(n) ops     # n
        vals_sorted = [t[0]*(-1) for t in result]  # convert to original value          # n ops            # n
        idx_sorted = [t[1] for t in result]                                             # n ops            # n
        return vals_sorted, idx_sorted                                                  # 1 op             # 0

# Let n be the size of the array. 
# Pushing a new node to the heap requires log(n) operations. 
# The same as popping the node with the highest priority from the heap. 
# T(n) = 2n*log(n) + 2n + 2
# Tight-fit upperbound for T(n): O(n * log(n))
# S(n) = 4n + 1
# Tight-fit upperbound for S(n): O(n)

    def ROC(self, prediction_probs, trueLabels):
        """Produce a ROC plot. 
        Implemented according to the paper (An introduction to ROC analysis) by Tom Fawcett.
        
        Keyword arguments:
        prediction_probs -- a list of dictionaries containing prediction probabilities
        trueLabels -- a list of true labels for the test set
        """                                                                                   # step count      # space count
        legends = []                                                                          # 1 op            # y
        labels_graph = np.unique(trueLabels)                                                  # n ops           # y

        for label_val in labels_graph:                                                        # y ops
            probs = [d[label_val] if label_val in d.keys() else 0 for d in prediction_probs]  # n ops           # n

            # sort the prediction probabilities in decreasing order
            probs_sorted, idx_sorted = self.sortReverse(probs)                                # n*log(n) ops    # 2n
            # sort the test labels according to sorted indices
            trueLabels = list(trueLabels)                                                     # 2 ops           # n
            labels_sorted = [trueLabels[i] for i in idx_sorted]                               # n ops           # n

            # initialize parameters
            TP = 0                                                                            # 1 op            # 1
            FP = 0                                                                            # 1 op            # 1
            R = []  # a list of ROC points                                                    # 1 op            # n
            P = len([i for i in trueLabels if i == label_val])                                # n ops           # 1
            N = len(trueLabels) - P                                                           # 3 ops           # 1
            f_prev = float('-inf')                                                            # 1 op            # 1
            i = 0                                                                             # 1 op            # 1

            # implement the algorithm
            while i < len(labels_sorted):                                                     # n ops
                f_i = probs_sorted[i]                                                         # 2 ops           # 1
                label = labels_sorted[i]                                                      # 2 ops           # 1
                if f_i != f_prev:                                                             # 1 op
                    R.append((FP/N, TP/P)) # push point onto R                                # 1 op
                    f_prev = f_i                                                              # 1 op
                if label == label_val:                                                        # 1 op
                    TP += 1  # true positive                                                  # 1 op
                else:                                                                         # 1 op
                    FP += 1  # false positive                                                 # 1 op
                i += 1                                                                        # 1 op

            val_1 = (FP/N, TP/P)                                                              # 1 op            # 1
            R.append(val_1) # push (1, 1) onto R                                              # 1 op
            legends.append(f'Class {label_val}')                                              # 1 op
            plt.plot(*zip(*R))                                                                # 1 op

        plt.plot([0, 1], [0, 1], 'k--')                                                       # 1 op
        plt.xlabel('False Positive Rate')                                                     # 1 op
        plt.ylabel('True Positive Rate')                                                      # 1 op
        plt.title('ROC')                                                                      # 1 op
        plt.legend(legends)                                                                   # 1 op
        plt.show()                                                                            # 1 op

# n: length of test data
# y: number of unique labels
# y is much less than n. Thus, y can be regarded as a constant when computing time complexity.

# T(n) = 1 + n + y * (nlogn + 3n + 12n + 14) + 6
# Tight-fit upperbound for T(n): O(n * log(n))

# S(n) = 6n + 2y + 9
# Tight-fit upperbound for S(n): O(n) 
