import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def load_dataset(filename):
    data = np.loadtxt(filename, delimiter=',')
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y

def train_svc(train_file, test_file, kernel, kernel_param, C):
    # load datasets
    X_train, y_train = load_dataset(train_file)
    X_test, y_test = load_dataset(test_file)
    
    # init & train SVC model
    model = SVC(kernel=kernel, C=C,
            gamma=float(kernel_param) if kernel == 'rbf' else 'scale', 
            degree=int(kernel_param) if kernel == 'poly' else 3,
            random_state=42)
    model.fit(X_train, y_train)
    
    # predict
    y_pred = model.predict(X_test)
    
    # evaluate model
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # result
    classes = sorted(set(y_test) | set(y_pred))
    print("Confusion Matrix:")
    print("Pred >\t" + "\t".join(map(str, classes)))
    for i in classes:
        row = [conf_matrix[i, j] if i < len(conf_matrix) and j < len(conf_matrix[i]) else 0 for j in classes]
        print(f"\t{i}\t" + "\t".join(map(str, row)))
    
    print(f"Accuracy: {acc * 100:.2f}%")

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 mlp-svm-ex4.py <trainset_file> <testset_file> <kernel> <kernel_param> <C>")
        return
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    kernel = sys.argv[3]  # linear|poly|rbf|sigmoid
    kernel_param = sys.argv[4]  # gamma for rbf | degree for poly
    C = float(sys.argv[5])
    
    train_svc(train_file, test_file, kernel, kernel_param, C)

if __name__ == "__main__":
    main()
