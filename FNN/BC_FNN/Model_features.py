from sklearn.datasets import load_breast_cancer

features = load_breast_cancer().feature_names
print(list(features))