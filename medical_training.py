import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# declaring the dataframes
df = pd.read_excel(r"C:\Users\hp\OneDrive\Desktop\manualData.xlsx")
dfDiabetes = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\diabetes.csv")
dfTest = pd.read_excel(r"C:\Users\hp\OneDrive\Desktop\test_data.xlsx")
testTest = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\testCase.txt")

# print(df.shape)
# print(df.head())

# defining x and y for training data
x = df.iloc[0:, 1:]
y = df.iloc[0:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.25)

# defining x and y for predicting data
x_T = dfTest.iloc[:, 1:]
y_T = dfTest.iloc[:, 0]

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
# print(confusion_matrix(y_test, y_pred))
# print((accuracy_score(y_test, y_pred)) * 100)
# print(x_T.shape)
# print(clf.predict(x_T))

#using .txt file for prediction
x_tt = testTest.iloc[: , :]
x_tt_transpose = np.transpose(x_tt)
print(x_tt_transpose)
result = clf.predict(x_tt_transpose)
# pickle.dump(clf, open('model.pkl', 'wb'))


