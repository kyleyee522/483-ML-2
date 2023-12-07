import pandas as pd
import numpy as np
import sklearn.naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm


TRAIN_PERCENT = 0.75

df = pd.read_csv("emails.csv")

# df_train = df.sample(frac=0.75, replace=True)
# df_test = df.sample(frac=0.25, replace=True)

# Shuffle the dataframe's rows to improve results
df = df.sample(frac = 1, random_state = 2).reset_index()

df_train = df.iloc[:int(TRAIN_PERCENT * len(df)), :]
df_test = df.iloc[int(TRAIN_PERCENT * len(df)):, :]

df_train_x = df_train["text"]
df_test_x = df_test["text"]

y_train = df_train["spam"]
y_test = df_test["spam"]

vectorizer = CountVectorizer()
# Create the vocabulary of words
vectorizer.fit(df_train_x)
# Create the 2D list of numbers
x_train = vectorizer.transform(df_train_x)
x_test = vectorizer.transform(df_test_x)

x_train = x_train.toarray()
x_test = x_test.toarray()

nb = sklearn.naive_bayes.MultinomialNB()

nb.fit(x_train, y_train)

test_predict = nb.predict(x_test)

# rmse_test = np.sqrt(mean_squared_error(df_test_y, test_predict))
# r2_test = r2_score(df_test_y, test_predict)
# test_accuracy = accuracy_score(y_test, test_predict)
confuse_matrix = confusion_matrix(y_test, test_predict)
# total = sum(sum(confuse_matrix))
# accuracy = (confuse_matrix[0,0]+confuse_matrix[1,1]) / total
# sensitivity = confuse_matrix[0,0] / (confuse_matrix[0,0]+confuse_matrix[0,1])
# specificty = confuse_matrix[1,1] / (confuse_matrix[1,0]+confuse_matrix[1,1])
# precision = precision_score(y_test, test_predict)
# precision = confuse_matrix[0,0] / (confuse_matrix[0,0]+confuse_matrix[1,0])

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

# print("Test RMSE: " + str(rmse_test))
# print("Test R^2 Score: " + str(r2_test))
# print("Confusion Matrix: \n" + str(confuse_matrix))
# print("Accuracy Score: " + str(test_accuracy))
# print("Sensitivity: " + str(sensitivity))
# print("Specificity: " + str(specificty))
# print("Precision: " + str(precision))

# Plot the ROC Curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predict)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="SVM")
display.plot()
plt.show()