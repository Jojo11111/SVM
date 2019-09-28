from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# X1=[3,7;6,6;4,6;5,6.5];
# X2=[1,2;3,5;7,3;3,4;6,2.7; 4,3;2,7];

x = np.array([[3, 7], [6, 6], [4, 6], [5, 6.5], [1, 2], [3, 5], [7, 3], [3, 4], [6, 2.7], [4, 3], [2, 7]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# clf = svm.SVC(C = 10.0,kernel='rbf')
clf = svm.SVC(C=10.0, kernel='poly', degree=3)
clf.fit(x, y)

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.winter)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, c='k')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.spring)

plt.show()
