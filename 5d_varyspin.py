# System imports
import sys
import os
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import matplotlib.gridspec as gridspec
import matplotlib
import pickle
from tqdm import tqdm

sys.path.append('/home/karen.kang/LIGOSURF23/Q')
import algo as p
style.use('/home/karen.kang/LIGOSURF23/plotting.mplstyle')
cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)


dimensions = ['q', 'spin1', 'spin2', 'theta1', 'theta2']
models = ['90', '270']


lam = [2/9, 0,0, 0.5, 0,0, 0.5]
lambdas = []

# Values you want to iterate over for the second and third elements
values_to_vary = [0,0.05,0.1,0.15,0.2,0.25,0.3, 0.35,0.4, 0.45]

for value in values_to_vary:
    new_lam = [2/9] + [value,value,0.5, value,value,0.5]
    lambdas.append(new_lam)
    

for m in models:
    p.load_and_update_model(m)
    directory = f'nocut/5D/'+m+'vs/'
    os.makedirs(directory, exist_ok=True)
    for i in range(len(lambdas)):
        lam0 = lambdas[i]
        bbh = p.BBH(lam = lam0)
        start = [bbh.q, bbh.spin1, bbh.spin2, bbh.theta1, bbh.theta2]
        mapper = p.MapDegeneracyND(lam0=lam0, start=start, dimensions=dimensions, stepsize = 0.1, sample = 50000, SNR = 22.5) #OK for 270
        mapper.run_mapping_bothways()
        fig = plt.figure(figsize=(7,6))
        ets = [item[0]/(item[0]+1)**2 for item in mapper.points]
        chiefs = [(item[0]*item[1]*np.cos(item[3])+ item[2]*np.cos(item[4]))/(item[0]+1) for item in mapper.points]
        cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
        fig = plt.figure(figsize=(7,6), constrained_layout = True)
        ps = p.ParameterSpace(lam0=lam0)
        sc = plt.scatter(ps.eta, ps.chi_eff, c = ps.mismatch, alpha = 0.8, cmap = cmap)
        plt.scatter(ets, chiefs, color ='#BFA89E', s = 60, edgecolors = 'white',zorder = 10)
        plt.plot(bbh.eta, bbh.chi_eff, '*', markersize= 18, color ='black',zorder = 30)
        plt.xlabel('$\eta$', fontsize = 24)
        plt.ylabel('$\chi_{\mathrm{eff}}$', fontsize = 24)
        plt.grid(False)
        plt.xlim(0.0826,0.25)
        plt.ylim(-1,1)
        plt.colorbar(sc,label= '$\mathcal{MM}$',fraction=0.13, cmap = cmap)
        plt.title('5D: vary spin')  
        plt.savefig(directory + m+'_vs'+ str(i) +'.png', dpi = 300)
        with open(directory + m+'_vs'+ str(i) +'.pkl', 'wb') as f:
            pickle.dump({"mapper": mapper, "bbh": bbh}, f)
        print('File saved')