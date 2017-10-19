# -*- coding:utf-8 -*-
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import argsort, mean
from termcolor import colored, cprint


def str2float(x):
    for i in range(0, len(x)):
        x[i] = float(x[i])
    return x


lines = open("bezdekIris.data").readlines()
# print lines
data = {}
for l in lines:
    data_str = l[:-1].split(",")
    # print data_str
    if data.has_key(data_str[-1]):
        data[data_str[-1]].append(str2float(data_str[:-1]))
    else:
        data[data_str[-1]] = [str2float(data_str[:-1])]

# print data
# for key in data.keys():
#     print "Covariance Maxtrix of " + key + "is:"
#     re = np.cov(np.array(data[key]).T)
#     print re
#     print
#     print "Correlation coefficients Matrix of " + key + "is:"
#     re = np.corrcoef(np.array(data[key]).T)
#     print re
#     print "\n"
#     #cols = l.split(.whitespace + ",")
#     #print cols

# KL Transformation
labels_set = range(0, len(data.keys()))
data_mat = np.array([ele for key in data.keys() for ele in data[key]])
labels = np.array([label for label, key in zip(labels_set, data.keys()) for ele in data[key]])
# data_mat[:, 0] = np.arange(0, 150)*0.1
# data_mat[:, 1] = np.arange(0, 150)*0.2
# data_mat = data_mat[:, 0:2]
# angle = np.arange(0, 150) * 2 * np.pi / 150
# data_mat = np.array([[np.cos(theta), np.sin(theta)] for theta in angle])
# print labels.shape
# data_mat = data_mat.reshape(data_mat.shape[0]*data_mat.shape[1], data_mat.shape[2])
# print "Covariance Maxtrix is:"
text_cov = colored("Covariance Maxtrix is:", "red")
print text_cov
Cx = np.cov(data_mat.T)
print Cx
print

text_cor = colored("Correlation Coefficients Matrix is:", "blue")
print text_cor
Cox = np.corrcoef(data_mat.T)
print Cox
print 

pca = PCA(n_components=data_mat.shape[1])
pca.fit(data_mat)
y = pca.transform(data_mat)
Cy = np.cov(y.T)
print text_cov
print Cy
print
# print "Correlation coefficients Matrix is:"
print text_cor
Coy = np.corrcoef(y.T)
print Coy
print 

# print pca.explained_variance_

E, V = eigh(Cx)
text_eva = colored("Eigenvalues:", "yellow")
print text_eva
print E
print


text_eve = colored("Eigenvectors:","green")
print text_eve
print V
print

index = argsort(E)[::-1]
E, V = E[index], V[:, index]
print text_eva
print E
print

print text_eve
print V
print

mean_data = mean(data_mat, axis=0, keepdims=True)

text_pca = colored("Result of PCA module in sklearn:", "magenta")
print text_pca
print y[:10, :]
print

text_evd = colored("Result of eigh in numpy.linalg:", "cyan")
print text_evd
print (data_mat - mean_data).dot(V)[:10, :]
print

# plt.figure(1)
# for xi in range(0, data_mat.shape[1]):
#     for yi in range(xi + 1, data_mat.shape[1]):
#         # print xi, yi, xi*data_mat.shape[1] + yi + 1, yi*data_mat.shape[1] + xi + 1
#         plt.subplot(data_mat.shape[1], data_mat.shape[1], xi*data_mat.shape[1] + yi + 1)
#         colors = ['darkorange', 'navy', 'turquoise']
#         lw = 0.1
#         for color, i, target_name in zip(colors, labels_set, data.keys()):
#             plt.scatter(data_mat[labels == i, xi], data_mat[labels == i, yi], color=color, alpha=.8, lw=lw,
#                         label=target_name)
#         plt.subplot(data_mat.shape[1], data_mat.shape[1], yi*data_mat.shape[1] + xi + 1)
#         colors = ['darkorange', 'navy', 'turquoise']
#         lw = 0.1
#         for color, i, target_name in zip(colors, labels_set, data.keys()):
#             plt.scatter(y[labels == i, xi], y[labels == i, yi], color=color, alpha=.8, lw=lw,
#                         label=target_name)

f = plt.figure()
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#EFE8E2'


plt.subplot(1, 2, 1)
colors = ['darkorange', 'navy', 'turquoise']
lw = 0.1
for color, i, target_name in zip(colors, labels_set, data.keys()):
    plt.scatter(data_mat[labels == i, 2], data_mat[labels == i, 3], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.rcParams['axes.facecolor'] = '#EFE8E2'


plt.subplot(1, 2, 2)
colors = ['darkorange', 'navy', 'turquoise']
lw = 0.1
for color, i, target_name in zip(colors, labels_set, data.keys()):
    plt.scatter(y[labels == i, 2], y[labels == i, 3], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.rcParams['axes.facecolor'] = '#EFE8E2'
# plt.figure(2)
# for xi in range(0, data_mat.shape[1]):
#     for yi in range(0, data_mat.shape[1]):
#         if xi == yi:
#             continue
#         else:
#             plt.subplot(data_mat.shape[1], data_mat.shape[1], xi*data_mat.shape[1] + yi + 1)
#             colors = ['darkorange', 'navy', 'turquoise']
#             lw = 0.1
#             for color, i, target_name in zip(colors, labels_set, data.keys()):
#                 plt.scatter(y[labels == i, xi], y[labels == i, yi], color=color, alpha=.8, lw=lw,
#                             label=target_name)

# from matplotlib.backends.backend_pdf import PdfPages
# pdf = PdfPages("campare.pdf")
# pdf.savefig(f)
# plt.style.use('ggplot')
# # f = plt.figure()
# plt.rcParams['axes.facecolor'] = '#EFE8E2'
plt.show()
