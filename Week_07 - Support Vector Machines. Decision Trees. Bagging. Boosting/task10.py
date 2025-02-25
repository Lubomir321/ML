import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def make_meshgrid(x, y, h=.02, lims=None):
    """
    Create a mesh of points to plot in.

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    if lims is None:
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
    else:
        x_min, x_max, y_min, y_max = lims
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, proba=False, **params):
    """
    Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    if proba:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, -1]
        Z = Z.reshape(xx.shape)
        out = ax.imshow(Z,
                        extent=(np.min(xx), np.max(xx), np.min(yy),
                                np.max(yy)),
                        origin='lower',
                        vmin=0,
                        vmax=1,
                        **params)
        ax.contour(xx, yy, Z, levels=[0.5])
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_classifier(X,
                    y,
                    clf,
                    ax=None,
                    ticks=False,
                    proba=False,
                    lims=None):  # assumes classifier "clf" is already fit
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax is None:
        plt.figure()
        ax = plt.gca()
        show = True
    else:
        show = False

    cs = plot_contours(ax, clf, xx, yy, alpha=0.8, proba=proba)

    labels = np.unique(y)
    if len(labels) == 2:
        ax.scatter(X0[y == labels[0]],
                   X1[y == labels[0]],
                   s=60,
                   c='b',
                   marker='o',
                   edgecolors='k')
        ax.scatter(X0[y == labels[1]],
                   X1[y == labels[1]],
                   s=60,
                   c='r',
                   marker='^',
                   edgecolors='k')
    else:
        ax.scatter(X0, X1, c=y, s=50, edgecolors='k', linewidth=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if ticks:
        ax.set_xticks(())
        ax.set_yticks(())

def main():
    wine = load_wine()
    X = wine.data[:, :2]
    y = wine.target      



    svc_full = SVC(kernel='linear', random_state=42)
    svc_full.fit(X, y)

    lims = (11,15,0,6)
    plot_classifier(X, y, svc_full,lims=lims)
    plt.title("Training on the whole dataset")
    plt.tight_layout()
    plt.show()

    print(svc_full.support_)
    support_vectors = X[svc_full.support_]
    support_labels = y[svc_full.support_]

    svc_support = SVC(kernel='linear', C=1, random_state=42)
    svc_support.fit(support_vectors, support_labels)

    plot_classifier(support_vectors, support_labels, svc_support,lims=lims)
    plt.title("Training on the support vectors")
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()