import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values   # labels
    return X, y

def compute_confusion_matrix(y_true, y_pred):
    confusion_matrix = Counter()
    classes = sorted(set(y_true) | set(y_pred))
    
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion_matrix[(true_label, predicted_label)] += 1
    
    return confusion_matrix, classes

def show_confusion_matrix(confusion_matrix, classes):
    print("Confusion Matrix:")
    print("Pred >\t" + "\t".join(map(str, classes)))
    for i in classes:
        row = [confusion_matrix[(i, pred)] for pred in classes]
        print(f"\t{i}\t" + "\t".join(map(str, row)))

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    confusion_matrix, classes = compute_confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    show_confusion_matrix(confusion_matrix, classes)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")

def main():
    if len(sys.argv) != 3:
        print("How to use: python3 ensemble_classifiers.py <trainset_file> <testset_file>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
    # load datasets
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    
    # classifiers
    classifiers = {
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10), n_estimators=50, algorithm="SAMME"),
        "Bagging": BaggingClassifier(n_estimators=50),
        "Random Forest": RandomForestClassifier(n_estimators=50)
    }
    
    # train & evaluate each classifier
    for name, model in classifiers.items():
        train_and_evaluate(model, X_train, y_train, X_test, y_test, name)

if __name__ == "__main__":
    main()
