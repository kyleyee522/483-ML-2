from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import datasets, metrics, model_selection, svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# TODO: Implement randomization of df's rows
# TODO: Implement K-Fold for training and testing



TRAIN_PERCENT = 0.75

df = pd.read_csv("emails.csv")

# Shuffle the dataframe's rows to improve results
df = df.sample(frac = 1, random_state = 4).reset_index()

df_train = df.iloc[:int(TRAIN_PERCENT * len(df)), :]
df_test = df.iloc[int(TRAIN_PERCENT * len(df)):, :]

# Get the x column
df_train_x = df_train["text"]
df_test_x = df_test["text"]

# Get the y column
y_train = df_train["spam"]
y_test = df_test["spam"]

df_test.reset_index(drop=True, inplace=True)

# df_train_numpy = df_train.to_numpy()
# df_test_numpy = df_test.to_numpy()


# This is what we're gonna use to convert the emails to a list of numbers counting how many of each word
# every email uses
vectorizer = CountVectorizer()

# Create the vocabulary of words
vectorizer.fit(df_train_x)

# Create the 2D list of numbers for training and testing
x_train = vectorizer.transform(df_train_x)
x_test = vectorizer.transform(df_test_x)

# print(x_train)
# print(df_train_y.to_numpy())

# print(vectorizer.get_feature_names_out())
# print(len(vectorizer.get_feature_names_out()))
# print(word_count_train.toarray())
# print(len(word_count_train.toarray()))

# Create the object used to perform KNN
neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(x_train, y_train)

test_predict = neigh.predict(x_test)

# Plot the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predict)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="KNN")
display.plot()
plt.show()

# Plot the Confusion Matrix
confuse_matrix = confusion_matrix(y_test, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=confuse_matrix, display_labels=neigh.classes_)
disp.plot()
plt.show()

# Out RMSE score for test data
# rmse_test = np.sqrt(mean_squared_error(y_test, test_predict))

# Our R^2 Score for test data
#r2_test = r2_score(y_test, test_predict)
# r2_test = r2_score(y_test, test_predict)

# print("Test RMSE: " + str(rmse_test))
# print("Test R^2 Score: " + str(r2_test))

# train_predict = neigh.predict(x_train)

# Out RMSE score for train data
# rmse_train = np.sqrt(mean_squared_error(y_train, train_predict))

# r2_train = r2_score(y_train, train_predict)

# print("Train RMSE: " + str(rmse_train))
# print("Train R^2 Score: " + str(r2_train))

test_accuracy = accuracy_score(y_test, test_predict)

class_names = ["Not Spam", "Spam"]
test_classification_report = classification_report(y_test, test_predict, target_names = class_names)

confuse_matrix = confusion_matrix(y_test, test_predict)

#decision_function = decision_function(x_test)

y_scores = neigh.predict_proba(x_test)



fpr,tpr,thresholds = roc_curve(y_test, y_scores[:, 1])

total = confuse_matrix[0,0] + confuse_matrix[0,1] + confuse_matrix[1,0] + confuse_matrix[1,1]
accuracy = (confuse_matrix[0,0] + confuse_matrix[1,1]) / total
sensitivity = confuse_matrix[0,0] / (confuse_matrix[0,0] + confuse_matrix[1,0])
specificity = confuse_matrix[1,1] / (confuse_matrix[1,1] + confuse_matrix[0,1])
precision = confuse_matrix[0,0] / (confuse_matrix[0,0] + confuse_matrix[0,1])
confidence_lower = (1 - accuracy) - (1.96 * np.sqrt((accuracy * (1 - accuracy)) / total))
confidence_upper = (1 - accuracy) + (1.96 * np.sqrt((accuracy * (1 - accuracy)) / total))

print("Confusion Matrix: \n" + str(confuse_matrix))
print("Accuracy: " + str(accuracy))
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))
print("Precision: " + str(precision))
print("Confidence Inderval (95%): [" + str(confidence_lower) + ", " + str(confidence_upper) + "]")

# print("Accuracy Score: " + str(test_accuracy))

#print("Classification Report: \n" + str(test_classification_report))

# print("Confusion Matrix: \n" + str(confuse_matrix))

# print("False Positive Rates: " + str(fpr))

# print("True Positive Rates: " + str(tpr))

# print("Thresholds: " + str(thresholds))
# print(y_test)
# print(test_predict)

# print(np.count_nonzero(test_results == 0))
# print(np.count_nonzero(test_results == 1))