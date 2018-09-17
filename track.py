#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from libSignal import edge_filter
from libAppliance import Appliance, find_likely
from libChart import plotit

debug = True

print('Loading data...')

house = 2
title_part = '24hrs'
p_factor = 1000  # W to kW
p_units = 'kW'
d_factor = 3600 #60 #3600 #1
d_units = 'hrs' #'min' #'hrs' #'secs'


if house == 1:
    datafile = './house1.csv'
    df = pd.read_csv(datafile)
    ts = df['unix_ts'].values[0]
    df = df.set_index('unix_ts')

    df['cool'] = df['sub8'] + df['sub20'] # Fridge and Circuit w/ Chest Freezer
    df['heatpump'] = df['sub13'] + df['sub14']
    df['dryer'] = df['sub5'] + df['sub6']
    df['oven'] = df['sub1'] + df['sub2']
    df['other'] = df['sub9'] + df['sub10'] # Clothes Washer and Dishwasher
    df['agg'] = df['cool'] + df['dryer'] + df['heatpump'] + df['dryer'] + df['oven'] #+ df['other']

    signal = df['cool'].values
    #signal = df['8'].values
    #signal = df['20'].values
    #signal = df['8'].values + df['20'].values
    #signal = signal / p_factor
elif house == 2:
    df = pd.read_csv('./house2.csv')
    ts = df['unix_ts'].values[0]
    df = df.set_index('unix_ts')
    df['agg'] = df['sub1'] + df['sub2']

    #signal = df['sub11'].values[:20000] # for house 2

    sample_window = 1 ### will create a sample size 10,000
    signal = df['agg'].values[10000 * sample_window:10000 * (sample_window + 1)]


else:
    print('ERROR: House', house, 'does not exist!')
    exit(1)

# apply edge preserving filter
print('Apply edge preserving filter...')

X = edge_filter(signal, show=True)
signal = signal / p_factor
X = X / p_factor
T = len(signal)
vstep = 10 / p_factor # other papers use 80W for a step


# print('[Drawing Chart] Median Filter Results...')
# plt.figure()
# plt.title('Edge Preserving Filter Results (%s)' % (title_part))
# plt.xlabel('Time (s)')
# plt.ylabel('Power (%s)' % p_units)
# plt.plot(signal, label='Actual Signal')
# plt.plot(X, label='Filtered Signal')
# plt.legend()
# plt.show()
#exit(0)

x0 = x1 = 0
#X = np.zeros(T)
dX = np.zeros(T)
M = 0
loads = []
contant_on = 0
for t in range(1, T):

    # store changes in power
    dx = X[t] - X[t-1]
    dX[t] = dx

    #check to see if an appliance turns OFF
    if dx <= -vstep:
        if debug: print()
        if debug: print('OFF spike of', dx, p_units, '@ ts=', ts + t, '(t =', t, ')')
        (Mc, p) = find_likely(dx, loads, 'OFF', debug)
        if Mc[0] > -1 and p > 0.01:
            for m in Mc:
                if debug: print('\t Appliance', m+1, 'turned OFF was ON for', loads[m].duration, d_units)
                #if cluster_data: cluster_points.append([loads[m].duration, loads[m].avg_power])
                loads[m].turn_off()
        else:
            if debug: print('\t Something turned OFF, but we do not know what!')
        dx = 0

    #check to see if an appliance turns ON
    if dx >= vstep:
        if debug: print()
        if debug: print('ON spike of', dx, p_units, '@ ts=', ts + t, '(t =', t, ')')
        (Mc, p) = find_likely(dx, loads, 'ON', debug)
        if Mc[0] > -1 and p > 0.01:
            for m in Mc:
                if debug: print('\t Appliance', m+1, 'turned ON was OFF for', loads[m].duration, d_units)
                loads[m].turn_on(dx)
        else:
            if debug: print('\t We found a new appliance! id:', M+1)
            load = Appliance(t, name='Appliance %i' % (M+1), record=True)
            load.turn_on(dx)
            loads.append(load)
        dx = 0

    M = len(loads)
    for m in range(M):
        loads[m].update(dx, t=1/d_factor) # + np.random.normal(0, 1)


M = len(loads)
print('Different loads found:', M)

print('[Drawing Chart] Median Filter Results...')
plt.figure()
plt.title('Edge Preserving Filter Results (%s)' % (title_part))
plt.xlabel('Time (s)')
plt.ylabel('Power (%s)' % p_units)
plt.plot(signal, label='Actual Signal')
plt.plot(X, label='Filtered Signal')
plt.legend()

print('[Drawing Chart] ON/OFF Power Spikes...')
plt.figure()
plt.title('ON/OFF Power Spikes (%s)' % title_part)
plt.xlabel('Time (s)')
plt.ylabel('Power (%s)' % p_units)
plt.plot(dX)

print('[Drawing Chart] Binary Appliance Traces...')
plt.figure()
plt.title('Binary Appliance Traces (%s)' % title_part)
plt.xlabel('Time (s)')
plt.ylabel('Power (%s)' % p_units)
plt.plot(X, label='Actual Signal')
for m in range(M):
    plt.plot(loads[m].X, label=loads[m].name)
plt.legend()

print('[Drawing Chart] Appliance Multivariate Gaussian PDFs...')
plt.figure()
plt.title('Appliance Multivariate Gaussian PDFs (%s)' % title_part)
#plt.xlabel('Power (%s)' % p_units)
#plt.ylabel('Probability')
for m in range(M):
    #l = 'Appliance %i' % (m+1)
    loads[m].plot()#label=l)
#plt.legend()

# print('[Drawing Chart] Appliance Power Gaussian PDFs...')
# plt.figure()
# plt.title('Appliance Power Gaussian PDFs (%s)' % title_part)
# plt.xlabel('Power (%s)' % p_units)
# plt.ylabel('Probability')
# for m in range(M):
#     avg = loads[m].avg_demand
#     l = 'Appliance %i (μ = %0.3f %s)' % (m+1, avg, p_units)
#     loads[m].PDF_demand.plot(xlim=(X.min(),X.max()+3), label=l)
# plt.legend()
#
# print('[Drawing Chart] Appliance Duration Gaussian PDFs...')
# plt.figure()
# plt.title('Appliance Duration Gaussian PDFs (%s)' % title_part)
# plt.xlabel('Duration (%s)' % d_units)
# plt.ylabel('Probability')
# for m in range(M):
#     avg = loads[m].avg_duration
#     l = 'Appliance %i (μ = %0.3f %s)' % (m+1, avg, d_units)
#     loads[m].PDF_on.plot(xlim=(0,7200/d_factor), label=l)
# plt.legend()





plt.show()
