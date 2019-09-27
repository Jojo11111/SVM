from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# X1=[3,7;6,4;4,6;7.5,6.5]
# X2=[1,3;5,2;7,3;3,4;6,1]

x = np.array([[3, 7], [6, 4], [4, 6], [7.5, 6.5], [1, 3], [5, 2], [7, 3], [3, 4], [6, 1]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

clf = svm.SVC(C=7.0, kernel='linear')
clf.fit(x, y)

w = clf.coef_[0]  # 获取w
a = -w[0] / w[1]  # 斜率
# 画图划线
xx = np.linspace(0, 10)
yy = a * xx - (clf.intercept_[0]) / w[1]  # xx带入y，截距

# 画出与点相切的线
b = clf.support_vectors_[0]
yy_up = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_down = a * xx + (b[1] - a * b[0])

plt.figure(figsize=(8, 4))
plt.plot(xx, yy, c='r')
plt.plot(xx, yy_down, c='g')
plt.plot(xx, yy_up, c='b')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, c='r')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.winter)

plt.show()
