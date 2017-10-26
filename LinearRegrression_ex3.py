# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages


def generateNoise(num, mu, sigma):
    return np.random.normal(mu, sigma, num)


def generateDataset(num, mu, sigma):
    # x = np.random.random(num)
    # x.sort()
    # print x
    x = np.linspace(0, 1, 25)
    noise = generateNoise(num, mu, sigma)
    return x, np.sin(2*np.pi*x) + noise


num = 25
mu = 0
sigma = 0.16
xt = np.linspace(0, 1, 100)
yt = np.sin(2*np.pi*xt)

baseParaMeans = np.array([0.2*i for i in range(24)])
baseParaVar = 0.2*0.2
primeNum = np.array(np.arange(8))
print primeNum.shape
X, Y = np.zeros((100, 25), dtype='float'), np.zeros((100, 25), dtype='float')
for i in range(100):
    X[i, :], Y[i, :] = generateDataset(num, mu, sigma)

ylow = -1.5
yup = 1.5
# lnl = np.array([6.4, 5.0, 3.6, -0.31, -3.4, -6.4, -15])
lnl = np.array([6.4, 3.6, -3.4, -15])
rc('text', usetex=True)
pp = PdfPages('x.pdf')
bias = plt.figure("bais")
# plt.yticks(np.arange(-1.5, 1.5, 0.2))
# plt.style.use('ggplot')
# plt.rcParams['axes.facecolor'] = '#EFE8E2'
plt.axis([0, 1.0, ylow, yup])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.plot(xt, yt, label='$y=\sin(2\pi x)$', linewidth=1)
for index, lbda in enumerate(np.exp(lnl)):
    # f = plt.figure("lambda=" + str(lbda))
    var = plt.figure('var')
    var.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.33, hspace=0.42)
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = '#EFE8E2'
    # plt.subplot(1, len(lnl), index+1)
    plt.subplot(2, 2, index + 1)
    plt.title('$\ln \lambda =' + str(np.log(lbda)) + '$', fontsize=10)
    # plt.title('Fitted curve $\ln \lambda =' + str(np.log(lbda)) + '$')
    plt.yticks(np.arange(-1.5, 1.51, 0.5))
    plt.axis([0, 1, ylow, yup])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    sum = 0
    for i in range(100):
        # phi = np.array([np.exp(-(ele - baseParaMeans)**2 / (2*baseParaVar)) for ele in X[i, :]])
        phi = np.array([np.power(ele, primeNum) for ele in X[i, :]])
        # print X[i, :]
        # print phi
        # print phi.shape
        omega = np.linalg.inv(lbda * np.eye(8) + np.dot(phi.T, phi)).dot(phi.T).dot(Y[i, :].reshape(25, 1))
        omega = omega.reshape(1, 8)
        # print omega.shape
        # yp = np.array([omega.dot((np.exp(-(ele - baseParaMeans)**2 / (2*baseParaVar))).reshape(24, 1))
        #               for ele in xt]).reshape(100, 1)
        yp = np.array([omega.dot(np.power(ele, primeNum)) for ele in xt]).reshape(100, 1)
        sum = sum + yp
        plt.plot(xt, yp, linewidth=1)
        # print yp.shape, xt.shape
    plt.figure("bais")
    plt.plot(xt, sum/100.0, '--', label='$\ln \lambda=' + str(np.log(lbda)) + '$')
    plt.legend()

# f = plt.figure()
# plt.title('Predict Value')
# plt.title('Predict Value')


pp.savefig(var)
pp.close()
plt.show()