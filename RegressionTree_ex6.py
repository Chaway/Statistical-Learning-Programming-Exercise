# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import text


def generateNoise(num, mu, sigma):
    return np.random.normal(mu, sigma, num)


def generateDataset(num, mu, sigma):
    x = np.random.random(num)
    x.sort()
    # print x
    # x = np.linspace(0, 1, num)
    noise = generateNoise(num, mu, sigma)
    return x, np.sin(2*np.pi*x) + noise


def predict(xt, omega):
    row, col = omega.shape
    primeNum = np.array(np.arange(col))
    return np.array([omegaPoly.dot(np.power(ele, primeNum)) for ele in xt]).reshape(len(xt), 1)


def linearRegressionPoly(X, Y, maxPrime):
    # print X.shape
    primeNum = np.array(np.arange(maxPrime + 1))
    phi = np.array([np.power(ele, primeNum) for ele in X])
    omega = np.linalg.inv(np.dot(phi.T, phi)).dot(phi.T).dot(Y.reshape(Y.shape[0], 1))
    omega = omega.reshape(1, maxPrime + 1)
    return omega


def treeRegression(X, Y, maxPrime):
    # primeNum = np.array(np.arange(maxPrime + 1))
    errorTreshold = 0.5
    omega = []
    index = []
    error = 100
    col = X.shape[0]
    # print col
    iter = 0
    errorSums = []
    lefterror = 100
    righterror = 100

    while iter == 0:
        iter += 1
        # print "iter=", iter, " , error=", error
        miniError = 100000
        miniIndex = 0
        omegasLeft = []
        errorsLeft = []
        omegasRight = []
        errorsRight = []
        errorSums = []
        for cut in range(4, col-3):
            # print cut
            omegaLeft = np.polyfit(X[:cut], Y[:cut], maxPrime)
            p = np.poly1d(omegaLeft)
            omegasLeft.append(omegaLeft)
            # print Y[:cut]
            errorVec = Y[:cut].reshape(cut, 1) - p(X[:cut])
            # print errorVec
            errorsLeft.append(errorVec)
            squareErrorLeft = np.sum(errorVec**2)/float(cut)
            # print "errorLeft = ", squareErrorLeft
            omegaRight = np.polyfit(X[cut:], Y[cut:], maxPrime)
            p = np.poly1d(omegaRight)
            omegasRight.append(omegaRight)
            errorVec = Y[cut:].reshape(col - cut, 1) - p(X[cut:])
            # print errorVec
            errorsRight.append(errorVec)
            squareErrorRight = np.sum(errorVec**2)/float(col - cut)
            # print "errorRight = ", squareErrorRight
            errorSum = squareErrorLeft + squareErrorRight
            errorSums.append(errorSum)
            if errorSum < miniError:
                miniError = errorSum
                lefterror = squareErrorLeft
                righterror = squareErrorRight
                miniIndex = cut - 4
        print miniIndex
        # error = miniError
        print len(omegasLeft)
        omega.append([omegasLeft[miniIndex], omegasRight[miniIndex]])
        # index.append(miniIndex)
        # Y = np.append(errorsLeft[miniIndex], errorsRight[miniIndex])
        # print Y
        # print Y

        # errorplot = plt.figure("error")
        # plt.plot(errorSums)
    return omega, miniIndex + 4, lefterror, righterror


num = 1000
mu = 0
s
xt = np.linspace(0, 1, num)
yt = np.sin(2*np.pi*xt)
ytree = np.zeros((num), dtype='float')
maxPrime = 3
# X, Y = np.zeros((1, num), dtype='float'), np.zeros((1, num), dtype='float')
X, Y = generateDataset(num, mu, sigma)

tree = [[0, num]]
errorTresh = 0.1
while len(tree) != 0:
    temptree =[]
    for i, subtree in enumerate(tree):
        print subtree
        omegaTree, index, lefterror, righterror = treeRegression(X[subtree[0]:subtree[1]], Y[subtree[0]:subtree[1]], maxPrime)
        print "Left Error is ", lefterror, "  Right error is ", righterror
        if lefterror >= errorTresh and index >= 9:
            temptree.append([subtree[0], subtree[0] + index])
        else:
            # print omegaTree
            pl = np.poly1d(omegaTree[0][0])
            print subtree[0], subtree[0] + index
            ytree[subtree[0]:subtree[0] + index] = pl(xt[subtree[0]:subtree[0] + index])
        if righterror >= errorTresh and subtree[1] - subtree[0] - index + 1 >= 9:
            temptree.append([subtree[0] + index, subtree[1]])
        else:
            pr = np.poly1d(omegaTree[0][1])
            print subtree[0] + index, subtree[1]
            ytree[subtree[0] + index: subtree[1]] = pr(xt[subtree[0] + index : subtree[1]])
    tree = temptree



# print(omegaTree)
omegaPoly = np.polyfit(X, Y, maxPrime)
print omegaPoly
p = np.poly1d(omegaPoly)
ypPoly = p(xt)
# primeNum = np.array(np.arange(maxPrime + 1))
# ypPoly = predict(xt, p(xt))
# pl = np.poly1d(omegaTree[0][0])
# pr = np.poly1d(omegaTree[0][1])
MSE1 = np.sum((yt - ypPoly)**2)
# print MSE1
rc('text', usetex=True)
reg = plt.figure("reg", figsize=(16, 6))
reg.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.95, wspace=0.17, hspace=0.42)
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#EFE8E2'
pdf = PdfPages('x0.1.pdf')
plt.subplot(1, 2, 1)
plt.plot(xt, yt, label='$y=\sin(2\pi x)$')
plt.plot(xt, ypPoly, '--', label='$Simple\ Linear\ Regression$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()

plt.subplot(1, 2, 2)
# treereg = plt.figure("tree")
plt.plot(xt, yt, label='$y=\sin(2\pi x)$')
plt.plot(xt, ytree, '--', label='$Regression\ Tree$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
MSE2 = np.sum((yt - ytree)**2)

print MSE1, MSE2
# plt.plot(xt[:index[0]], pl(xt[:index[0]]))
# plt.plot(xt[index[0]-1:], pr(xt[index[0]-1:]))
# plt.plot(xt, ypTree)
# error = plt.figure("error")
# plt.plot(errorSums)
pdf.savefig(reg)
pdf.close()
plt.show()