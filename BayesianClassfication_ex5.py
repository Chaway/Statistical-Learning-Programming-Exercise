# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from termcolor import colored
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import text

labela = [1, 2, 3]
vara = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3]
labelb = ['s', 'm', 'l']
varb = [1, 2, 2, 1, 1, 1, 2, 2, 3, 3, 3]
vary = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1]

py = np.array([0., 0.])
pay = np.array([[0., 0., 0.], [0., 0., 0.]])
pby = np.array([[0., 0., 0], [0., 0., 0.]])

for index, eley in enumerate(vary):
    # print eley, index
    if eley == -1:
        py[0] += 1
        pay[0][vara[index] - 1] += 1
        pby[0][varb[index] - 1] += 1
    else:
        py[1] += 1
        pay[1][vara[index] - 1] += 1
        pby[1][varb[index] - 1] += 1


LaplaceSmoothing = True
alpha = 1

if LaplaceSmoothing:
    pay = pay + alpha
    pby = pby + alpha
    py = py + alpha

print py
print pay
print pby

pay[0] = pay[0]/np.sum(pay[0])
pay[1] = np.array(pay[1])/np.sum(pay[1])
pby[0] = np.array(pby[0])/np.sum(pby[0])
pby[1] = np.array(pby[1])/np.sum(pby[1])
py = py/(py[0] + py[1])


ptext = colored('Likelihood:', color='red')
print ptext
for i, ele in enumerate(zip(pay[0], pby[0])):
    # print ele
    ptexta = 'P(x1=' + str(labela[i]) + ' | y=-1) = ' + str(ele[0])
    ptextb = 'P(x2=' + labelb[i] + ' | y=-1) = ' + str(ele[1])
    print ptexta
    print ptextb

for i, ele in enumerate(zip(pay[1], pby[1])):
    # print ele
    ptexta = 'P(x1=' + str(labela[i]) + ' | y=1) = ' + str(ele[0])
    ptextb = 'P(x2=' + labelb[i] + ' | y=1) = ' + str(ele[1])
    print ptexta
    print ptextb


ptext = colored('\nPrior Probability:', color='red')
print ptext
ptext_1 = 'P(y=-1) = ' + str(py[0]) + ', '
ptext1 = 'P(y=1) = ' + str(py[1])
print ptext_1, ptext1


ptext = colored('\nPosterior Probability:', color='red')
print ptext


testa = 0
testb = 1

py1 = pay[0][testa] * pby[0][testb] * float(py[0])
py2 = pay[1][testa] * pby[1][testb] * float(py[1])
py1Nor = py1/(py1 + py2)
py2Nor = py2/(py1 + py2)
text1 = colored(str(py1Nor), 'red')
text2 = colored(str(py2Nor), 'red')
print 'P(y=-1 | x1=1,x2=m) =', text1
print 'P(y= 1 | x1=1,x2=m) =', text2



