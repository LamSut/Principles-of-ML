import sys
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

def load_dataset(filename):
    data = pd.read_csv(filename, header=None)
    x = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values   # label
    return x, y

def main(train_file, test_file):
    # load datasets
    x_train, y_train = load_dataset(train_file)
    x_test, y_test = load_dataset(test_file)
    
    # train GaussianNB classifier
    model = GaussianNB()
    model.fit(x_train, y_train)
    
    # predict
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print('------------------------------')
    print(f'Accuracy: {acc:.2f}%')
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'True {i}' for i in range(len(cm))], 
                            columns=[f'Pred {i}' for i in range(len(cm))])
    print('------------------------------')
    print('Confusion Matrix:')
    print(cm_df.to_string())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("How to use: python Bayes-ex3.py <train_file> <test_file>")
        print("For example: python3 Bayes-ex3.py ../data/iris/iris.trn ../data/iris/iris.tst")
    else:
        main(sys.argv[1], sys.argv[2])