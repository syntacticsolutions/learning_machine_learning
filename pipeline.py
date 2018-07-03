from sklearn import datasets

# load the data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split the data into test and training sets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .1)

# Create the classifier
from sklearn.neighbors import KNeighborsClassifier

my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

#predict the values of the test data

predictions = my_classifier.predict(X_test)

#output the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))