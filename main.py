from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

spam = pd.read_csv('spam.csv', encoding='ISO-8859-1')
spam = spam.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"] , axis=1)

# Extract features and labesl
x = spam['v2']
y = spam['v1']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35)

# NLP
cv = CountVectorizer()
# ML cannot deal with words, so transform basically transform words into numbers
features = cv.fit_transform(x_train)
features_test = cv.transform(x_test)

# ML Models
m_svm = svm.SVC()
m_svm.fit(features, y_train)
pred = m_svm.predict(features_test)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, y_train)
pred_tree = clf.predict(features_test)

forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(features, y_train)
pred_random = forest.predict(features_test)

# Average
print(m_svm.score(features_test, y_test))

# Metrics
array = metrics.confusion_matrix(y_test, pred)

df = pd.DataFrame(array, ["Spam", "Ham"], ["Spam", "Ham"])
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True, annot_kws={"size": 16})
plt.show()

# Recall, precisison, F1
print(metrics.classification_report(y_test, pred, target_names=["Spam", "Ham"]))
print(metrics.classification_report(y_test, pred_tree, target_names=["Spam", "Ham"]))
print(metrics.classification_report(y_test, pred_random, target_names=["Spam", "Ham"]))
