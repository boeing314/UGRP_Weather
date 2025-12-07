import pandas as pd
import random
import numpy as np

def year_conv(s):
    d=0
    if int(s[:4])%4==0:
        mnth=[31,29,31,30,31,30,31,31,30,31,30,31]
        for i in range(int(s[5:7])-1):
            d+=mnth[i]
        d+=int(s[8:])
        return (d/366)*2*np.pi
    mnth=[31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(int(s[5:7])-1):
        d+=mnth[i]
    d+=int(s[8:])
    return (d/365)*2*np.pi

def time_conv(t):
    return (((int(t[:2])*60)+int(t[3:5]))/1440)*2*np.pi
def compute_humidity_ratio( d2m, sp=101325):
    """
    t2m, d2m: in Celsius
    sp: surface pressure in Pascals
    """
    d2c=d2m-273.15
    # Actual vapor pressure from dew point (Magnus formula)
    e = 6.112 * np.exp((17.67 * d2c) / (d2c + 243.5)) * 100  # convert hPa to Pa
    e = np.minimum(e, 0.99999 * sp)
    # Humidity ratio
    w = 0.622 * e / (sp - e)
    return w
