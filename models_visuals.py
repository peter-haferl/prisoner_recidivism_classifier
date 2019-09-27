import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix



def split_data(data, target):
    """splits data"""
    X = data.drop(columns=target)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def list_important_features(data, best_forest):
    """Lists features by order of importance in fitted Random Forest"""
    features = pd.DataFrame([best_forest.feature_importances_, data.columns]).T
    features = features.set_index(keys=features[1]).drop(columns=[1])
    return features.sort_values(by=0, ascending=False)


def roc_visual(best_model, X_train, X_test, y_test):
    """Creates ROC visual"""
    y_train_pred = best_model.predict(X_train)
    probs_test = best_model.predict_proba(X_test)[:, 1]
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 18}

    plt.rc('font', **font)
    plt.figure(figsize=(7, 7))
    plt.ylabel('True Positive Rate', fontdict=font)
    plt.xlabel('False Positive Rate', fontdict=font)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    fpr, tpr, thresholds = roc_curve(y_test, probs_test)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    # show the plot
    plt.show()


def nice_confusion(model, X_train, X_test, y_train, y_test):
    """Creates a nice looking confusion matrix"""
    plt.figure(figsize=(10, 10))
    plt.xlabel('Predicted Class', fontsize=18)
    plt.ylabel('True Class', fontsize=18)
    viz = ConfusionMatrix(
        model,
        cmap='PuBu', fontsize=18)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()


def baseline_logistic(X_train, X_test, y_train, y_test):
    """Runs baseline logistic regression model"""
    regr = LogisticRegression(C=1e5, solver='liblinear')
    regr.fit(X_train, y_train)
    y_train_pred = regr.predict(X_train)
    nice_confusion(regr, X_train, X_test, y_train, y_test)
    y_test_pred = regr.predict(X_test)
    print("Baseline logistic regression classification report: \n", classification_report(y_test, y_test_pred))
    return regr


def visual_metrics(model, X_train, X_test, y_train, y_test):
    """Creates confusion and ROC curve"""
    nice_confusion(model, X_train, X_test, y_train, y_test)
    roc_visual(model, X_train, X_test)
