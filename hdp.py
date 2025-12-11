from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("./hdp.csv")

mapping = {'Presence': 1, 'Absence': 0}
data['Heart Disease'] = data['Heart Disease'].map(mapping)

X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

ACC = accuracy_score(y_test, y_pred)

print(ACC*100)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Disease(0)', 'Healthy(1)'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()