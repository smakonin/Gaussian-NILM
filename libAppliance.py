#
# Copyright (C) 2017 Stephen Makonin (http://makonin.com)
#

import itertools, numpy as np
from libGaussian import GaussianPDF
from filterpy.stats import multivariate_gaussian, plot_covariance_ellipse

class Appliance(object):

    def __init__(self, t, name='Unknown', record=True):
        self.name = name
        self.PDF_demand = GaussianPDF()
        self.PDF_on = GaussianPDF()
        self.PDF_off = GaussianPDF()
        self.duration = 0
        self.demand = 0
        self.is_on = False
        self.x = 0
        self.times_on = 0
        self.turned_on = np.array([[0,0]])
        self.turned_off = np.array([[0,0]])
        self.record = record
        if self.record:
            self.X = np.full(t, None)
            if t > 0: self.X[-1] = 0

    @property
    def avg_demand(self):
        return self.PDF_demand.mean

    @property
    def avg_duration(self):
        return self.PDF_on.mean

    def turn_on(self, spike_on):
        if self.is_on: return
        self.PDF_off.update(self.duration)
        self.turned_on = np.vstack((self.turned_on, [self.avg_demand,self.duration]))
        self.duration = 0
        self.demand = spike_on
        self.is_on = True
        self.x = spike_on

    def turn_off(self):
        if not self.is_on: return
        self.PDF_on.update(self.duration)
        self.turned_off = np.vstack((self.turned_off, [self.avg_demand,self.duration]))
        self.times_on += 1
        self.duration = 0
        self.demand = 0
        self.is_on = False

    def update(self, delta_p, t=1):
        self.duration += t
        if not self.is_on:
            if self.record: self.X = np.append(self.X, 0)
            return
        self.x += delta_p
        if self.record: self.X = np.append(self.X, self.x)
        self.PDF_demand.update(self.x)

    def prob(self, delta_p, for_state):
        delta_p  = np.abs(delta_p)
        msg = '%-12s: ON? %5s, ΔD: %0f | mean: demand = %0f, dur_on = %0f, dur_off = %0f' % (self.name, self.is_on, self.duration, self.PDF_demand.mean, self.PDF_on.mean, self.PDF_off.mean)
        if for_state == 'ON':
            if self.is_on: return (0, '%-12s: Is already ON, ignore!' % (self.name))
            x = np.array([delta_p, self.duration])
            if self.times_on > 0:
                mu = np.array([self.PDF_demand.mean, self.PDF_off.mean])
                #cov = np.cov(self.turned_off[1:,0], self.turned_off[1:,1], bias=1)
                cov = np.array([[self.PDF_demand.var, 1], [1, self.PDF_off.var]])
            else:
                mu = np.array([self.PDF_demand.mean, self.PDF_on.mean])
                #cov = np.cov(self.turned_on[1:,0], self.turned_on[1:,1], bias=1)
                cov = np.array([[self.PDF_demand.var, 1], [1, self.PDF_on.var]])
            #if np.sum(cov) == 0:
            #    cov = np.array([[self.PDF_demand.var, 1], [1, self.PDF_off.var]])
            #print(cov)
            #try:
            #p = multivariate_gaussian(delta_p, mu, cov)
            p = multivariate_gaussian(x, mu, cov)
            #p = self.PDF_demand(delta_p)
            #except OverflowError:
            #    print('\t\t\t OverflowError error occured!')
            #    p = 1000
            return (p, msg)
        elif for_state == 'OFF':
            if not self.is_on: return (0, '%-12s: Is already OFF, ignore!' % (self.name))
            x = np.array([delta_p, self.duration])
            mu = np.array([self.PDF_demand.mean, self.PDF_on.mean])
            #cov = np.cov(self.turned_on[1:,0], self.turned_on[1:,1], bias=1)
            #if np.sum(cov) == 0:
            cov = np.array([[self.PDF_demand.var, 2], [2, self.PDF_on.var]])
            #print(cov)
            #try:
            #p = multivariate_gaussian(delta_p, mu, cov)
            p = multivariate_gaussian(x, mu, cov)
            #except OverflowError:
            #    print('\t\t\t OverflowError error occured!')
            #    p = 1000
            return (p, msg)
        else:
            raise Exception(-1, 'state must be either ON or OFF.')

    def plot(self):#, xlim=None, ylim=None, label=None):
        mu = np.array([self.PDF_demand.mean, self.PDF_on.mean])
        cov = np.array([[self.PDF_demand.var, 1], [1, self.PDF_on.var]])
        plot_covariance_ellipse(mu, cov=cov)#, xlim=xlim, ylim=ylim, label=label)



# {'combo_name': GaussianPDF}
#
# A = list(range(1, M+1))
# for i in range(1, M+1):
#     print()
#
# >>> import itertools
# >>> set(itertools.combinations(l, 2))
# {(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (2, 4)}
# convolve the combos:
# X1+X2 ~ Gaussian(mean1+mean2, sqrt(std1^2 + std2^2))



def find_likely(x, appliances, for_state, verbose):

    # A = list(range(len(appliances)))
    # Pr = np.array([])
    # C = []
    # for a in A:
    #     for M in itertools.combinations(A, a):
    #         mean = var = 0
    #         for m in M:
    #             mean += appliances[m].mean
    #             var += appliances[m].var
    #         pdf = GaussianPDF(mean, var)
    #
    #         (p, msg) = pdf.prob(x, for_state)
    #         if verbose: print('\t\t %s | prob = %0f' % (msg, p))
    #         C.append(M)
    #         Pr.append(p)
    #
    # if Pr.size == 0 or np.max(Pr) <= 0:
    #     return ((-1,), 0)
    # i = np.argmax(Pr)
    # return (C[i], Pr[m])


    M = len(appliances)
    Pr = np.zeros(M)
    for i in range(M):
        (Pr[i], msg) = appliances[i].prob(x, for_state)
        if verbose: print('\t\t %s | prob = %0f' % (msg, Pr[i]))
    if Pr.size == 0 or np.max(Pr) <= 0:
        return ([-1], 0)
    m = np.argmax(Pr)
    return ([m], Pr[m])
