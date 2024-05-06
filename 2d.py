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


models = ['270']

dimensions = ['eta', 'chieff']


values = [6,4, 1.5]
base_values = [value / (1 + value)**2 for value in values]
base_pattern = [0., 0, -0.5, 0., 0, -0.5]
lambdas = []


for i in range(9):
    base_value = base_values[i // 3]
    val = values[i % 3]
    
    # Generate lambda
    if i % 3 == 0:
        lam = [base_value] + base_pattern
    elif i % 3 == 1:
        lam = [base_value] + [0, 0, 0, 0, 0, 0]
    else:
        lam = [base_value] + [0, 0, 0.5, 0, 0, 0.5]
    
    lambdas.append(lam)

# lambdas = [[1.5/(2.5)**2,0,0,-0.5,0,0,-0.5],[1.5/(2.5)**2,0,0,0,0,0,0]]


for m in models:
    p.load_and_update_model(m)
    directory = f'nocut/2D/'+m+'/'
    os.makedirs(directory, exist_ok=True)

    for i in range(len(lambdas)):
        lam0 = lambdas[i]
        bbh = p.BBH(lam = lam0)
        start = [bbh.eta, bbh.chi_eff]
        mapper = p.MapDegeneracyND(lam0=lam0, start=start, dimensions=dimensions, sample = 150000, SNR = 60)
        mapper.run_mapping_bothways()
        fig = plt.figure(figsize=(7,6))
        ets = [item[0] for item in mapper.points]
        chiefs = [item[1] for item in mapper.points]
        fig = plt.figure(figsize=(7,6), constrained_layout = True)
        plt.plot(bbh.eta, bbh.chi_eff, '*', markersize= 18, color ='black',zorder = 50)
        ps = p.ParameterSpace(lam0=lam0)
        sc = plt.scatter(ps.eta, ps.chi_eff, c = ps.mismatch, alpha = 0.8, cmap = cmap)
        plt.scatter(ets, chiefs, color ='#BFA89E', s = 60, edgecolors = 'white',zorder = 10)
        plt.xlabel('$\eta$', fontsize = 24)
        plt.ylabel('$\chi_{\mathrm{eff}}$', fontsize = 24)
        plt.grid(False)
        plt.xlim(0.0826,0.25)
        plt.ylim(-1,1)
        plt.colorbar(sc,label= '$\mathcal{MM}$',fraction=0.13, cmap = cmap)
        plt.savefig(directory + m+'_'+ str(i) +'.png', dpi = 300)
        with open(directory + m+'_'+ str(i) +'.pkl', 'wb') as f:
            pickle.dump({"mapper": mapper, "bbh": bbh}, f)
        print('File saved')