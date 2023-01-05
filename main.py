import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

spam = pd.read_csv('spam.csv', encoding='ISO-8859-1')
spam = spam.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"] , axis=1)

# Extract features and labesl
x = spam['v2']
y = spam['v1']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.35)

# NLP
cv = CountVectorizer()
features = cv.fit_transform(x_train)

# ML Models
m_svm = svm.SVC()
m_svm.fit(features, y_train)


# Metrics
# ML cannot deal with words, so transform basically transform words into numbers
features_test = cv.transform(x_test)
# Average
print(m_svm.score(features_test, y_test))

pred = m_svm.predict(features_test)

array = metrics.confusion_matrix(y_test, pred)

df = pd.DataFrame(array, ["Spam", "Ham"], ["Spam", "Ham"])
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True, annot_kws={"size": 16})
plt.show()

# Recall, precisison, F1
print(metrics.classification_report(y_test, pred, target_names=["Spam", "Ham"]))
