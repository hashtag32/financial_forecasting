'''
This script perfoms the basic process for forecasting stock values with 
the help of ML algorithms.

The program is divided into 4 steps:
   1. Readin the provided dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)

Thanks to Luis Torgo for the source data.
License: CC0 (No Rights Reserved)
For more information about the dataset: http://mldata.org/repository/data/viewslug/stockvalues/
'''


#inputdata; names of the stocks are unimportant. (aerospace companies)
datafilename='input/stockvalues.csv'

#switches
switch_plot_input=0

from pandas import read_table
from sklearn import preprocessing, cross_validation
import numpy as np
import matplotlib.pyplot as plt
import time

try:
    #optimizing the plotting (if package seaborn is installed)
    import seaborn
except ImportError:
    pass

# =====================================================================


def readin_data():
    '''
    Readin the data from the prior downloaded csv file the data into a pandas DataFrame.
    '''
    #collect all data into one pandas frame
    frame = read_table(
        datafilename,
        encoding='latin-1',
        sep=',',                # comma separated values
        skipinitialspace=True,  # Ignore spaces after the separator
        index_col=None,         #each row get its own number
        header=None,            #each column get its own number
    )
    dates=np.asarray(frame[0])

    dataDict={}

    #iterate over the number of columns, but start from 1 (just the data, not the dates)
    j=0
    dataList=np.array(frame[1:-1],dtype=np.float)
    print dataList
    time.sleep(50)
    #print dataList
    #for i in range(1,frame.shape[1]):
    #    dataList=np.asarray(frame[i]) #data from one company over the whole dataset
    #    dataDict[j]=dataList 
    #    j+=1
    #    if(switch_plot_input):
    #        plt.plot(dataList)
    #        plt.show()

    return dates,dataDict


# =====================================================================


def get_features_and_labels(dataDict):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Convert values to floats and into a numpy array (is required by sklearn)
    #arr = np.array(frame, dtype=np.float)
    for dataList in dataDict:
        # Normalize the entire data set (is required for some regression ml methods)
        #from sklearn.preprocessing import StandardScaler, MinMaxScaler
        arr = preprocessing.MinMaxScaler().fit_transform(arr)
    
        # Use 80% of the data for training and test with the rest
        #X, y = arr[:, :-1], arr[:, -1]
        X_train, _, y_train, _ = cross_validation.train_test_split(X, y, test_size=0.8)
        #last column is the target
        X_test, y_test = X,y
 
    # Imputation=if missing values are in the dataset (in this set are no invalid elements)
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    
    scaler = preprocessing.StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test


# =====================================================================


def evaluate_learner(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different algorithms to evaluate 
    the performance relatively to each other 

    Returns a sequence of tuples containing:
        (title, expected values, actual values)
    for each learner.
    '''

    # Use a support vector machine for regression
    from sklearn.svm import SVR

    clfList=[SVR(kernel='rbf', gamma=0.1), SVR(kernel='linear'), SVR(kernel='poly', degree=2)]

    for clf in clfList:

        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_test)
        r_2 = svr.score(X_test, y_test)
        yield 'RBF Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    # Train using a linear kernel
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Linear Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred

    # Train using a polynomial kernel
    svr = SVR(kernel='poly', degree=2)
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Polynomial Model ($R^2={:.3f}$)'.format(r_2), y_test, y_pred


# =====================================================================


def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, expected values, actual values)
    
    All the elements in results will be plotted.
    '''

    # Using subplots to display the results on the same X axis
    fig, plts = plt.subplots(nrows=len(results), figsize=(8, 8))
    fig.canvas.set_window_title('Predicting data from ' + datafilename)

    # Show each element in the plots returned from plt.subplots()
    for subplot, (title, y, y_pred) in zip(plts, results):
        # Configure each subplot to have no tick marks
        # (these are meaningless for the sample dataset)
        subplot.set_xticklabels(())
        subplot.set_yticklabels(())

        # Label the vertical axis
        subplot.set_ylabel('stock price')

        # Set the title for the subplot
        subplot.set_title(title)

        # Plot the actual data and the prediction
        subplot.plot(y, 'b', label='actual')
        subplot.plot(y_pred, 'r', label='predicted')
        
        # Shade the area between the predicted and the actual values
        subplot.fill_between(
            # Generate X values [0, 1, 2, ..., len(y)-2, len(y)-1]
            np.arange(0, len(y), 1),
            y,
            y_pred,
            color='r',
            alpha=0.2
        )

        # Mark the extent of the training data
        subplot.axvline(len(y) // 2, linestyle='--', color='0', alpha=0.2)

        # Include a legend in each subplot
        subplot.legend()

    # Let matplotlib handle the subplot layout
    fig.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================
#MAIN

if __name__ == '__main__':
    # Readin the dataset
    print("Readin the dataset {}".format(datafilename))
    dates,dataDict = readin_data()

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(dataDict)

    # Evaluate multiple regression learners on the data
    print("Evaluating regression learners")
    results = list(evaluate_learner(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    plot(results)
