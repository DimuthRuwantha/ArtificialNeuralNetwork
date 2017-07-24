from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

print(cancer.keys())

X = cancer['data']
y = cancer['target']
print(cancer['DESCR'])
X_train, X_test, Y_train, Y_test = train_test_split(X, y)
print("done")
