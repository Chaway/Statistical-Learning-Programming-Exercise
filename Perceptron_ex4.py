# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import text


def sign(w, x):
    if np.dot(w.T, x) > 0 :
        return 1
    elif np.dot(w.T, x) < 0:
        return -1
    else:
        return 0


def SGD(w, alpha, x, size, y, theta, withdual=False):
    num = 0
    if not withdual:
        for i in range(size):
            if y[i] != sign(w, np.append(x[i, :], [1]).reshape(3, 1)):
                w = w + theta*y[i]*(np.append(x[i, :], [1]).reshape(3, 1))
                num += 1
    else:
        # print w
        for i in range(size):
            if y[i]*np.dot(w.T, np.append(x[i, :], [1]).reshape(3, 1)) <= 0:
                alpha[i] += theta
                num += 1
        # print alpha
        w = np.dot(np.append(x, np.ones((3, 1)), axis=1).T, alpha*y.reshape(3, 1))
    if num > 0:
        flag = True
    else:
        flag = False
    return w, flag


def drawLine(w, iterNum):
    xt = np.arange(-1., 5., 0.1)
    yt = (-w[-1] - w[0] * xt) / w[1]
    if iterNum == 0:
         ptext = colored(str(w.T), 'red')
         print 'Initial: w = ', ptext
    if w[1] != 0:
        plt.plot(xt, yt, label=str(iterNum))
    else:
        yt = np.arange(0, 11, 0.1)
        plt.plot(-w[-1] / w[0] * np.ones(len(yt)), yt)
    if iterNum > 0:
        ptext = str(iterNum)
    else:
        ptext = 'Initial'
    rota = - np.arctan(w[0, 0] / float(w[1, 0] * 2.7)) * 180 / np.pi
    plt.text(xt[-1], yt[-1], ptext, fontsize=4, rotation=rota, verticalalignment='center')


x = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])
labels_set = [1, -1]
label = ['Class 1', 'Class 2']

ds = plt.figure()
ds.subplots_adjust(left=0.05, bottom=0.05, right=1, top=1)
pp = PdfPages('w1.pdf')

plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#EFE8E2'
colors = ['darkorange', 'turquoise']
lw = 0.1

for color, i, target_name in zip(colors, labels_set, label):
    plt.scatter(x[y == i, 0], x[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)


theta = 1
size = 3
w = np.array([-2, 1., -1]).reshape(3, 1)
alpha = np.array([4., -2, 2.]).reshape(3, 1)

withdual = False

# xt = np.arange(-1, 5, 0.1)
# yt = (-w[-1]-w[0]*xt)/w[1]
# plt.plot(xt, yt)
flag = True

iterNum = 0
W = []

while(flag):
    if not withdual:
        drawLine(w, iterNum)
    else:
        w = np.dot(np.append(x, np.ones((3, 1)), axis=1).T, alpha * y.reshape(3, 1))
        drawLine(w, iterNum)
    w, flag= SGD(w, alpha, x, size, y, theta, withdual)
    iterNum = iterNum + 1
    text = colored(str(w.T), 'red')
    print str(iterNum) + ' Iteration:  w = ', text


        # W.append(w.T)
    # print W


# plt.yticks(np.arange(-1, 11, 2))
# plt.legend()
# plt.axis([0, 1.0, ylow, yup])
# plt.show()
pp.savefig(ds)
pp.close()








