#
# Copyright (C) 2017 Stephen Makonin (http://makonin.com)
#

import numpy as np
from filterpy.stats import gaussian, plot_gaussian_pdf

def mean_r(x, t, mean):
    return (t-1)/t*mean+1/t*x

def variance_r(x, t, mean, var):
    if t == 1: return 0
    return (t-1)*var/t+((x-mean)**2)/(t-1)


class GaussianPDF(object):
    def __init__(self, mean=0, var=1):
        self.t = 0
        self.mean = mean
        self.var = var
        self.last_x = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '𝒩(μ=%f,σ²=%f)' % (self.mean, self.var)

    def __call__(self, x):
        if self.stdev == 0: self.var = 1
        p = 0
        if np.isscalar(x):
            p = gaussian(x, self.mean, self.var)
        else:
            p = np.array([gaussian(x[i], self.mean, self.var) for i in range(x.size)])
        return p

    @property
    def stdev(self):
        return np.sqrt(self.var)

    def update(self, x):
        self.t += 1
        self.last_x = x
        self.mean = mean_r(x, self.t, self.mean)
        self.var = variance_r(x, self.t, self.mean, self.var)

    def plot(self, xlim=None, label=None):
        plot_gaussian_pdf(mean=self.mean, variance=self.var, xlim=xlim, label=label)
