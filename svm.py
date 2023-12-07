from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics, model_selection, svm
import matplotlib.pyplot as plt
from nltk.corpus import words
import pandas as pd
import numpy as np


TRAIN_PERCENT = 0.75

df = pd.read_csv("emails.csv")

# Shuffle the dataframe's rows to improve results
df = df.sample(frac = 1, random_state = 1).reset_index()

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
# print(len(x_train[0]))
# print(type(x_train))

# print(x_test)
# print(len(x_test[0]))
# print(type(x_test))

# print(vectorizer.get_feature_names_out())
# print(len(vectorizer.get_feature_names_out()))

# This is for removing words
# col_names = vectorizer.get_feature_names_out()
# word_bank = words.words()

# print(len(word_bank))
# print(np.setdiff1d(col_names, word_bank))

# i = 0

# while i < len(col_names):

#     print("i = " + str(i))

#     if col_names[i] in word_bank:

#         pass

#     else:

#         # Delete column i on axis 1 (axis 1 = column)
#         np.delete(x_train, i, 1)
#         np.delete(x_test, i, 1)

#     i += 1

# print("New Training X Col Length: " + str(len(x_train)))
# print("New Testing X Col Length: " + str(len(x_test)))

#exit(1)

# print("starting SVC")

# Create the SVC object
#svc = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='linear'))

svc = SVC(kernel='linear')

# print("In SVC")

# Create the model with training
svc.fit(x_train, y_train)

#metrics.plot_roc_curve(svc, x_test, y_test)

# print("I just fitted the distribution")

# Get the array
test_predict = svc.predict(x_test)

# Plot the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predict)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="SVM")
display.plot()
plt.show()

# print("got train prediction")

# Out RMSE score for test data
# rmse_test = np.sqrt(mean_squared_error(y_test, test_predict))

# Our R^2 Score for test data
# r2_test = r2_score(y_test, test_predict)
# r2_test = r2_score(y_test, test_predict)

# print("Test RMSE: " + str(rmse_test))
# print("Test R^2 Score: " + str(r2_test))

train_predict = svc.predict(x_train)

# Out RMSE score for train data
# rmse_train = np.sqrt(mean_squared_error(y_train, train_predict))

# r2_train = r2_score(y_train, train_predict)

# print("Train RMSE: " + str(rmse_train))
# print("Train R^2 Score: " + str(r2_train))

test_accuracy = accuracy_score(y_test, test_predict)

class_names = ["Not Spam", "Spam"]
test_classification_report = classification_report(y_test, test_predict, target_names = class_names)

# Plot the Confusion Matrix
confuse_matrix = confusion_matrix(y_test, test_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=confuse_matrix, display_labels=svc.classes_)
disp.plot()
plt.show()

total = confuse_matrix[0,0] + confuse_matrix[0,1] + confuse_matrix[1,0] + confuse_matrix[1,1]
accuracy = (confuse_matrix[0,0] + confuse_matrix[1,1]) / total
sensitivity = confuse_matrix[0,0] / (confuse_matrix[0,0] + confuse_matrix[1,0])
specificity = confuse_matrix[1,1] / (confuse_matrix[1,1] + confuse_matrix[0,1])
precision = confuse_matrix[0,0] / (confuse_matrix[0,0] + confuse_matrix[0,1])
confidence_lower = (1 - accuracy) - (1.96 * np.sqrt((accuracy * (1 - accuracy)) / total))
confidence_upper = (1 - accuracy) + (1.96 * np.sqrt((accuracy * (1 - accuracy)) / total))

#roc_curve = roc_curve()

# print("Accuracy Score: " + str(test_accuracy))

# print("Classification Report: \n" + str(test_classification_report))

print("Confusion Matrix: \n" + str(confuse_matrix))
print("Accuracy: " + str(accuracy))
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))
print("Precision: " + str(precision))
print("Confidence Inderval (95%): [" + str(confidence_lower) + ", " + str(confidence_upper) + "]")