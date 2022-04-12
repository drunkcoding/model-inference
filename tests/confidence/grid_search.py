from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
print(iris)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
fitted = clf.fit(iris.data, iris.target)
temp = sorted(clf.cv_results_.keys())

print(clf.cv_results_)
print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)