import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file
from bokeh.io import export_svgs, export_png
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

# generate some synthetic time series for six different categories
cats = ["sepal length", "sepal width", "petal length", "petal width"]
print cats
yy = np.random.randn(2000)
g = np.random.choice(cats, 2000)
data_mat = np.array([ele for key in data.keys() for ele in data[key]])
# labels = np.array([key for key in data.keys() for ele in data[key]])
# # for i, l in enumerate(cats):
# #     yy[g == l] += i // 2
data_l = np.reshape(data_mat, data_mat.shape[0] * data_mat.shape[1], order='F')
labels = np.array([cats[i] for i in range(data_mat.shape[1]) for j in range(data_mat.shape[0])])
df = pd.DataFrame(dict(score=data_l, group=labels))
# print g
# print labels
# data_dict = {key:   for i, key in enumerate(cats)}

# find the quartiles and IQR for each category
groups = df.groupby('group')
print groups
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

print upper

print lower


# find the outliers for each category
def outliers(group):
    cat = group.name
    return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']


out = groups.apply(outliers).dropna()
print out
# prepare outlier data for plotting, we need coordinates for every outlier.
if not out.empty:
    outx = []
    outy = []
    for cat in cats:
        # only add outliers if they exist
        if not out.loc[cat].empty:
            for value in out[cat]:
                print cat
                outx.append(cat)
                outy.append(value)

p = figure(width=600, height=500, tools="save", background_fill_color="#EFE8E2", title="", x_range=cats)
# if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
qmin = groups.quantile(q=0.00)
qmax = groups.quantile(q=1.00)
upper.score = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'score']), upper.score)]
lower.score = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'score']), lower.score)]


def get_name(group):
    return group.name


cats = groups.apply(get_name)
# stems
p.segment(cats, upper.score, cats, q3.score, line_color="black")
p.segment(cats, lower.score, cats, q1.score, line_color="black")


#FFBB6C
# whiskers (almost-0 height rects simpler than segments)
p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

# outliers
if not out.empty:
    p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

# boxes
p.vbar(cats, 0.7, q2.score, q3.score, fill_color="#E08E79", fill_alpha=0.8, line_color="black")
p.vbar(cats, 0.7, q1.score, q2.score, fill_color="#3B8686", fill_alpha=0.8, line_color="black")

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = "white"
p.grid.grid_line_width = 2
p.xaxis.major_label_text_font_size = "12pt"
p.yaxis.axis_label = 'Values (cm) '
output_file("boxplot.html", title="boxplot.py example")


#show(p)
# p.background_fill_color = "#EFE8E2"
# p.border_fill_color = None
# export_png(p, filename="plot.png")
# p.output_backend = "svg"
# export_svgs(p, filename="plot.svg")
# export_svgs(obj, filename=None, height=None, width=None, webdriver=None)[source]

# print plt.style.available
plt.style.use('ggplot')
f = plt.figure()
plt.rcParams['axes.facecolor'] = '#EFE8E2'
y = np.random.randn(100) * 10
plt.plot(y)
with PdfPages('foo.pdf') as pdf:
    pdf.savefig(f)

plt.show()
