from sklearn.datasets import load_iris
from sklearn.tree import export_text
from sklearn import tree
from sklearn import metrics

X,y = load_iris(return_X_y=True)

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(X,y)
tree.plot_tree(dtree)
y_pred = dtree.predict(X)

print(X)
print(y)
print(metrics.accuracy_score(y,y_pred))

# import graphviz
# dot_data = tree.export_graphviz(dtree, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris_pdf")

iris = load_iris()
dot_data = tree.export_graphviz(dtree, out_file=None,
                               feature_names=iris.feature_names,
                               class_names=iris.target_names,
                               filled=True, rounded=True,
                               special_characters=True)
# graph2 = graphviz.Source(dot_data)

dtree2 = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
dtree2 = dtree2.fit(X,y)
r = export_text(dtree2, feature_names=iris['feature_names'])
print(r)