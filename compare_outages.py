# -*- coding: utf-8 -*-
"""
Compare outage predictions
"""

import tkinter
import tkinter.filedialog
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi'] = 300
plt.close('all')

root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
root.update()

#%% Inputs
source_train = tkinter.filedialog.askopenfilename(
    title='Select event.nc',
    initialdir='./results/',
    filetypes=[('NetCDF files', '*.nc')])

source_infer1 = tkinter.filedialog.askopenfilename(
    title='Select porbabilility file',
    initialdir='./',
    filetypes=[('NetCDF files', '*.nc')])

source_infer2 = tkinter.filedialog.askopenfilename(
    title='Select porbabilility file',
    initialdir='./',
    filetypes=[('NetCDF files', '*.nc')])

#%% Initialization
ds_train=xr.open_dataset(source_train)
ds_infer1=xr.open_dataset(source_infer1)
ds_infer2=xr.open_dataset(source_infer2)

#%% Plots
plt.figure(figsize=(18,10))
plt.plot(ds_infer1.time,ds_infer1.outage_probability,'r',label='Inference (Mesonet, Breckinridge)')
plt.plot(ds_infer2.time,ds_infer2.outage_probability,'b',label='Inference (AWAKEN, A1)')
plt.plot(ds_train.event,ds_train.rf_prediction,'.k',markersize=20,label='Training')
plt.plot(ds_train.event[ds_train.is_outage==1],ds_train.is_outage[ds_train.is_outage==1],'xg',markersize=20,label='Real outage')
plt.grid()
plt.xlabel('Time (UTC)')
plt.legend()
plt.ylabel('Outage probability')