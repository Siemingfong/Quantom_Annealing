# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import numpy as np
import pandas as pd
from utils import run_from_ipython, report2df

if run_from_ipython():
    import matplotlib.pyplot as plt
    import seaborn as sns

def show_cm_list(cm_list, class_names):
    cm_avg = np.mean(cm_list, axis=0)
    if run_from_ipython():
        df_cm = pd.DataFrame(cm_avg, index=class_names, columns=class_names)
        plt.figure(figsize=(8, 8))
        sns.heatmap(df_cm, annot=True)
    else:
        print(cm_avg)

def show_rp_list(rp_list):
    rp_avg = None
    n = len(rp_list)
    for rp in rp_list:
        if rp_avg is None:
            rp_avg = rp
        else:
            for category in rp_avg:
                for metric in rp_avg[category]:
                    rp_avg[category][metric] += rp[category][metric]
    for category in rp_avg:
        for metric in rp_avg[category]:
            rp_avg[category][metric] = rp_avg[category][metric] / n
    rp_df = report2df(rp_avg)
    if not run_from_ipython():
        print(rp_df)
    # for rp in rp_list:
    #     print(report2df(rp))
    return rp_df