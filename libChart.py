#
# Copyright (C) 2017 Stephen Makonin (http://makonin.com)
#

import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt


###
#   A handy matplotlib charting function
###
def plotit(X, fig, pos, title='', xlabel='', ylabel=''):
    subfig = fig.add_subplot(pos)
    subfig.set_title(title)
    subfig.set_xlabel(xlabel)
    subfig.set_ylabel(ylabel)

    if (type(X) is list or type(X) is tuple) and len(X) == 2:
        subfig.plot(X[0], X[1], label='The signal')
    else:
        subfig.plot(X, label='The signal')
