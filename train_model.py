import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Don','Dr','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title'] = df['Title'].map(
        {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df = df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'])
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FareBin', 'Embarked']
    X = pd.get_dummies(df[features])
    y = df['Survived']
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=150)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X, y = load_and_preprocess('data/titanic.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y)

    param_grid = {'max_depth': [3, 4, 5, 6, 8, 10]}
    base_clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=4,
        random_state=args.seed)

    grid = GridSearchCV(base_clf, param_grid, cv=5,
                        scoring='f1_macro', n_jobs=1)
    grid.fit(X_train, y_train)

    clf = grid.best_estimator_
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    val_f1 = f1_score(y_test, preds, average='macro')
    val_acc = accuracy_score(y_test, preds)

    checkpoint_name = f"rf_d{grid.best_params_['max_depth']}_n{args.n_estimators}_seed{args.seed}.pkl"
    with open(checkpoint_name, 'wb') as f:
        pickle.dump(clf, f)

    print(classification_report(y_test, preds))
    print(f"val_f1_macro: {val_f1:.4f} | val_accuracy: {val_acc:.4f} | checkpoint: {checkpoint_name}")

if __name__ == '__main__':
    main()