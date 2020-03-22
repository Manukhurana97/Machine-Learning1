# Driving linear regression algo

# y = mx+b
# m = mean(x)mean(y)-mean(xy)/mean(x)square-((x)square)mean
# b = mean(y) - m(mean(x))

#          ^     _
# y = 1-SE(y)/SE(y)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float)


# create a random data points
# varience : how much each point can vary from the previous point
# step : how far to step on average per point

def create_dataset(howmany, varience, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(howmany):
        y = val+random.randrange(-varience, varience)
        ys.append(y)

        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_dataset(40, 10, correlation='pos')


def best_fit(xs, ys):
    #                  _ _   __     _       ___
    # formula  :->  m=(x.y - xy) / (x)^2 - (x^2)
    #                 _    _
    #               b=y - mx

    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs))**2 - mean(xs**2)))

    b = mean(ys) - m*mean(xs)

    return m, b


m, b = best_fit(xs, ys)


# y = mx+b
regression_line = [(m*x)+b for x in xs]


# predictions
given_x = []
for i in range(41, 50, 1):
    given_x.append(i)
predict_y = [(m*x)+b for x in given_x]


#                     ^     _
# Formula  r^2 = 1-SE(y)/SE(y)
def square_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):

    y_mean_line = [mean(ys_orig) for ys in ys_orig]
    print(y_mean_line)

    square_error_regr = square_error(ys_orig, ys_line)
    square_error_y_mean = square_error(ys_orig, y_mean_line)

    return 1 - (square_error_regr / square_error_y_mean)


r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)


# plot
plt.scatter(xs, ys, label='data')
plt.legend(loc=4)
plt.plot(given_x, predict_y, color='red')
plt.plot(xs, regression_line)
plt.show()
