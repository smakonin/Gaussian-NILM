#
# Copyright (C) 2017 Stephen Makonin (http://makonin.com)
#

import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from libChart import plotit


def edge_filter(X, show=False):

    ###
    #   Apply a median filter on signal X (Figure 3a)
    ###
    window_size = 5 ### FREE PARAMETER!
    I = medfilt(X, window_size)


    ###
    #   Calculate the ct(u) signal (Figure 3b)
    ###
    Iʹ = np.append([0], np.diff(I)) # compute the derivative of I
    Ω = len(I)
    σs = 100 ### FREE PARAMETER!
    σr = 1 ### FREE PARAMETER!
    ct = np.zeros(Ω, np.intc)
    for u in range(1, Ω):
        ct[u] = ct[u-1] + np.sum(1 + σs/σr * abs(Iʹ[u]))  # Eq. (11) & (12)


    ###
    #   Plot in the transformed domain (Figure 3c)
    ###
    Ωw = np.max(ct) + 1
    Iw = np.zeros(Ωw)
    for u in range(Ω):
        Iw[ct[u]] = I[u]
    Iwʹ = np.copy(Iw) # save a copy for later, will need to know where the 0s are

    too_small = σs/σr * 10 ### FREE PARAMETER!
    for w in range(1, Ωw-1):
        if Iw[w] != 0 and Iw[w-1] == 0 and Iw[w+1] == 0:
            l = 1
            while Iw[w-l] == 0:
                l += 1
            r = 1
            while Iw[w+r] == 0:
                r += 1
            if r > too_small and l > too_small:
                print('\t Found:', 'left', l, 'right', r, 'Iw[w]', Iw[w], 'right value is', Iw[w-l-1:w-l+3])
                Iw[w-l+1] = Iw[w-l]
                Iw[w] = 0
                Iwʹ[w-l+1] = Iwʹ[w-l]
                Iwʹ[w] = 0
                print('\t Found:', 'left', l, 'right', r, 'Iw[w]', Iw[w], 'right value is', Iw[w-l-1:w-l+3])
                print()

    # draw lines to fill in 0 values in Iw
    start = -1
    end = -1
    for w in range(Ωw):

        if Iw[w] == 0:
            if start == -1:
                start = w - 1
            else:
                end = w + 1
        else:
            if end == -1:
                continue

            rise = Iw[end] - Iw[start]
            run = end - start
            slope = rise / run
            if run > too_small:
                #print('found gap size', run)
                for i in range(start+1, end):
                    Iw[i] = Iw[i-1] + slope

            start = -1
            end = -1

    #print('Before too small rm, len:', 'Iwʹ', len(Iwʹ), 'Iw', len(Iw))
    Iwʹ= Iwʹ[Iw>0]
    Iw = Iw[Iw>0]
    Ωw = len(Iw)
    #print('After too small rm, len: ', 'Iwʹ', len(Iwʹ), 'Iw', len(Iw))


    ###
    #   Filter in Ωw with a 1D Gaussian (Figure 3d)
    ###
    GIw = np.zeros(Ωw, np.intc)
    σ = np.max(I) * 2 ### FREE PARAMETER!
    H = gaussian_filter(Iw, σ, output=GIw, mode='nearest')


    ###
    #   Reverse transformed domain (Figure 3e)
    ###
    GwI = GIw[Iwʹ>0] # remove all 0 values from the final Signal

    return GwI

    ###
    #   Clean up the rise and fall edges
    ###
    ct2 = np.zeros(Ω, np.intc)
    for u in range(1, Ω):
        ct2[u] = ct2[u-1] + np.sum(1 + σs/σr * abs(GwI[u]))  # Eq. (11) & (12)
    Ωw2 = np.max(ct2)
    Iw2 = np.zeros(Ωw2+1)
    for u in range(Ω):
        Iw2[ct2[u]] = GwI[u]
    Iw2ʹ = np.copy(Iw2) # save a copy for later, will need to know where the 0s are

    # draw lines to fill in 0 values in Iw
    start = -1
    end = -1
    too_small = σs/σr * 10 ### FREE PARAMETER!
    for w in range(Ωw2):
        if start != -1 and Iw2[w] != 0 and Iw2[w+1] == 0:
            for i in range(w+1, Ωw2):
                if Iw2[i] != 0:
                    break
            print('i', i, 'start', start, 'i - start', abs(i - start), 'too_small', too_small)
            if abs(i - start) > too_small:
                print('\t', 'Iw2[w]', Iw2[w], 'Iw2[start+1]', Iw2[start+1], 'Iw2[start]', Iw2[start])
                Iw2[w] = 0
                Iw2[start+1] = Iw2[start]
                start += 1

        if Iw2[w] == 0:
            if start == -1:
                start = w - 1
            else:
                end = w + 1
        else:
            if end == -1:
                continue
            start = -1
            end = -1

    GwI2 = Iw2[Iw2>0] # remove all 0 values from the final Signal

    ###
    #   Make a pretty sub-figures example
    ###
    if show:
        fig = plt.figure()
        plotit(X, fig, 231, title='Figure 3a-1: Original Signal (w/o median filter)', xlabel='Time (s)', ylabel='Power (W)')
        plotit(I, fig, 232, title='Figure 3a-2: Input Signal I (median filter, win size ' + str(window_size) + ')', xlabel='Ω', ylabel='I')
        plotit(ct, fig, 233, title='Figure 3b: ct(u) - Ramping Signal Plot', xlabel='Ω', ylabel='ct(u)')
        plotit(Iw, fig, 234, title='Figure 3c: Signal I plotted in the transformed domain (Ωw)', xlabel='Ωw', ylabel='Iw')
        plotit(GIw, fig, 235, title='Figure 3d: Signal I filtered in Ωw with a 1D Gaussian', xlabel='Ωw', ylabel='G{Iw}')
        plotit(GwI, fig, 236, title='Figure 3e: Signal I filtered plotted in Ω', xlabel='Ω', ylabel='Gw{I}')
        plt.show()


    return GwI2
