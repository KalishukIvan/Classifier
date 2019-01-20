import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

punctuation = set(string.punctuation)  # Set of all punctuation symbols
LogisticRegression(solver='lbfgs')  # I have 1 FutureWarning, so I tried to improve it


def _clean(row):
    # To del all unnecessary symbols
    return ''.join(i for i in row if i not in punctuation)


def _mark(rating, n, mode):
    '''
    Changing rating to +1, -1, NaN
    :param rating: rating of our review (can be 1,2,3,4,5)
    :param n: expected 3
    :param mode: mode will decide what to do if rating == n
    mode:
        - positive -> +1
        - negative -> -1
        - NaN -> NaT
    :return: it will be our mark of review
    '''

    if int(rating) > n:
        return '+1'
    elif int(rating) < n:
        return '-1'
    elif mode == 'positive' and int(rating) == n:
        return '+1'
    elif mode == 'negative' and int(rating) == n:
        return '-1'
    elif mode == 'NaN' and int(rating) == n:
        return pd.NaT


def classifier(filename):
    '''
    It will contain 2 parts: work with data and working with classifier
    :param filename: expecting 'amazon_baby.csv'
    :return:
    '''
    # ----------------------------- Part 1 -----------------------------

    data = pd.read_csv(filename, sep=',', names=['name', 'review', 'rating'])
    data = pd.DataFrame(data)  # transforming into pandas DataFrame
    data = data.dropna()  # deleting all empty reviews
    data = data.drop(columns='name')  # deleting unnecessary column of names
    data = data.drop([0])  # deleting row: 0 - review - rating
    # print(data.head())
    print('Reading from file - done')

    for index, row in data.iterrows():  # for each review and rating we make transformation
        row[0] = _clean(row[0])
        row[1] = _mark(row[1], 3, 'NaN')
    data = data.dropna()  # deleting all reviews with NaT
    # print(data.head())
    print('Working with data - done.')

    # ----------------------------- Part 2 -----------------------------
    X = data.review.values
    Y = data.rating.values
    # splitting on training and validation sets
    x_training, x_test, y_training, y_test = train_test_split(X, Y, test_size=0.22)
    print('Splitting on sets - done')

    # Preparing trainig set for LogisticRegression
    text_clf = Pipeline([
        ('vect', CountVectorizer(token_pattern=r'\b\w+\b')),
        ('clf', LogisticRegression())])

    # Training
    text_clf.fit(x_training, y_training)
    print('Training classifier - done')

    # Predicting our validation set with trained classifier
    y_predicted = text_clf.predict(x_test)
    print('Prediction - done')

    # Results
    print(metrics.classification_report(y_test, y_predicted))

    # Matrix of view:
    # TP    FN
    # FP    TN
    res = metrics.confusion_matrix(y_test, y_predicted)
    print(res)


if __name__ == '__main__':
    import time

    start = time.time()
    classifier("amazon_baby.csv")
    print('Time elapsed - ', time.time() - start)
