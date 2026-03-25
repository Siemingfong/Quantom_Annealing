# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import pandas as pd
from datetime import datetime


def parse_date(string):
    """
    Parse date strings safely.
    """
    try:
        date = datetime.strptime(string, '%Y-%m-%dT%H:%M:%SZ').replace(hour=0, minute=0, second=0, microsecond=0)
    except:
        date = datetime.strptime(string, '%Y-%m-%dT%H:%M:%S.%fZ').replace(hour=0, minute=0, second=0, microsecond=0)
    return date

def parse_timestamp(ts):
    """
    Parse timestamps of dates, zeroing the time.
    """
    date = datetime.utcfromtimestamp(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    return date

def mean(x):
    """
    Safe mean().
    """
    return sum(x) / len(x) if len(x) > 0 else 0

def divide(a, b):
    """
    Safe divide().
    """
    return a / b if b != 0 else 0

def run_from_ipython():
    """
    Whether the script is running from iPython.
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def np2df(X, y):
    df = pd.DataFrame(X, y)
    return df

def df2np(df):
    return np.array(df.values), np.array(df.index)

def report2df(rp):
    return pd.DataFrame(rp, index=['precision', 'recall', 'f1-score', 'support']).transpose().round(2)