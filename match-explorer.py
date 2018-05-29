'''
Interactively plots MATCH composite populations. 
Requires MATCH style photometry files; inteded to 
be used with a grid of files created via MATCH's 
program 'fake'.

Use the ``bokeh serve`` command to run by executing:
    bokeh serve mist-explorer.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/match-explorer
in your browser.

To do:

 -- TBD...

'''
import numpy as np
import glob
import matplotlib as mpl

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource, ContinuousColorMapper, ColorBar
from bokeh.models.widgets import Slider, TextInput, Select, RadioButtonGroup, Div, Panel, Tabs, RangeSlider
from bokeh.plotting import figure

from MIST_codes.scripts import read_mist_models as rmm
from tqdm import tqdm
import sys
import os
import argparse

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# argument parser:
parser = argparse.ArgumentParser()
parser.add_argument("-pf", "--photfile",
                    help="specify a desired data file (format should be two magnitude columns, e.g.: V, I).")
parser.add_argument("-fc", "--filtercode", const='Tycho_BTycho_V', nargs='?', type=str,
                    help="A code specifying the filter set. E.g., Tycho_BTycho_V for Tycho B and V filters; will plot as B-V vs. V.")
args = parser.parse_args()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# check if data should be loaded from a given photometry file (intended to be *real* data).
# data is plotted as v-i, i, as per MATCH customs.
if args.photfile:
    photfn = args.photfile
    photd = np.genfromtxt(photfn)
    photv = photd.T[0]
    photi = photd.T[1]
    phot_source = ColumnDataSource(data=dict(x=photv-photi, y=photi))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# assign x and y label names; currently we're only plotting CMDs, so these are color & mag.
if 'Tycho_B' in args.filtercode:
    x_label = "BT - VT"
    y_label = "VT"
elif 'UVIS' in args.filtercode:
    names = args.filtercode.split('UVIS')
    x_label = 'F{:s} - F{:s}'.format(names[1], names[2])
    y_label = 'F{:s}'.format(names[2])
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Some useful functions:
def cutbads(x, y):
    # cut out bad mags (i.e., values of 99):
    xcut = x[(abs(x) < 99) & (abs(y) < 99)]
    ycut = y[(abs(x) < 99) & (abs(y) < 99)]
    
    return xcut, ycut

def getz(x, y, z):
    # get z (something like mass, or another param) for models w/ valid mags:
    zcut = z[(abs(x) < 99) & (abs(y) < 99)]
    
    return zcut

def getxyz(x, y, z, lim=[-99, 99] ,limz=[-99,99]):

    # get x,y, z w/i limits; designed with x, y = magnitudes in mind:
    xcut = x[(x < max(lim)) & (x > min(lim)) &
             (y < max(lim)) & (y > min(lim)) &
             (z < max(limz)) & (z > min(limz))]
    
    ycut = y[(x < max(lim)) & (x > min(lim)) &
             (y < max(lim)) & (y > min(lim)) &
             (z < max(limz)) & (z > min(limz))]
    
    zcut = z[(x < max(lim)) & (x > min(lim)) &
             (y < max(lim)) & (y > min(lim)) &
             (z < max(limz)) & (z > min(limz))]
    
    return xcut, ycut, zcut

def getdata(feh_val, vvc_val, bf_val, av_val, age_val):

    if isinstance(vvc_val, str):
        d = np.genfromtxt(glob.glob("models/bf{:.2f}_av{:.1f}_SFR0.0001_"\
                                    "t{:.1f}_{:.2f}_logZ{:.2f}_vvc{:s}_"\
                                    "{:s}.phot".format(bf_val, av_val, age_val,
                                                      age_val+0.02, feh_val,
                                                      vvc_val, args.filtercode))[0])
    else:
        d = np.genfromtxt(glob.glob("models/bf{:.2f}_av{:.1f}_SFR0.0001_"\
                                    "t{:.1f}_{:.2f}_logZ{:.2f}_vvc{:.1f}_"\
                                    "{:s}.phot".format(bf_val, av_val, age_val,
                                                      age_val+0.02, feh_val,
                                                      vvc_val, args.filtercode))[0])

    return d

def get_mag_mass(feh, vvc, bf, av, lage, limz=[0.1,8.0]):

    # gets magnitudes and initial masses for a given set of feh, v/vc, etc.:
    # (returns a potentially cut range of masses for highlighting purposes, 
    #  and the full range.)
    data = getdata(feh, vvc, bf, av, lage)
    v = data.T[0]
    i = data.T[1]
    initm = data.T[2]
    d = getxyz(v, i, initm,limz=limz)
    dfull = getxyz(v, i, initm,limz=[0.1,8.0])

    return {'v':d[0], 'i':d[1], 'imass':d[2]}, {'v':dfull[0], 'i':dfull[1], 'imass':dfull[2]}

def mass_colors(initm, model):
    # returns a list of rgb colors, used to help delineate initial masses:
    # (uses Spectral color map at the moment.)
    if model == 'single':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.Reds(mpl.colors.Normalize()(initm))]
    elif model == 'gauss':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.Blues(mpl.colors.Normalize()(initm))]
    elif model == 'flat':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.Purples(mpl.colors.Normalize()(initm))]

    return colors

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Using feh = -0.30, v/vc = 0.0, bf = 0.0, av = 0.0, log age = 8.9 as defaults...

# Get data for the single v/vc models:
data = get_mag_mass(-0.30, 0.0, 0.0, 0.0, 8.9)
# for the Gaussian dist. model:
gdata = get_mag_mass(-0.30, 'gauss', 0.0, 0.0, 8.9)
fldata = get_mag_mass(-0.30, 'flat', 0.0, 0.0, 8.9)

# Set up plots...
plot_CMD = figure(plot_height=800, plot_width=600,
                  tools="box_zoom,crosshair,pan,reset,save,wheel_zoom",
                  x_range=[min(data[0]['v']-data[0]['i']), max(data[0]['v']-data[0]['i'])], 
                  y_range=[max(data[0]['i']), min(data[0]['i'])])

# with black background:
plot_CMD.background_fill_color = "black"
plot_CMD.xgrid.visible = False
plot_CMD.ygrid.visible = False

# data for single vvc model:
source = ColumnDataSource(data=dict(x=data[0]['v']-data[0]['i'], y=data[0]['i'], 
                                    alpha=[0.6]*len(data[0]['i']), c=mass_colors(data[0]['imass'], 'single')))
# full data (i.e., this cannot be cut via initial mass):
fsource = ColumnDataSource(data=dict(x=data[1]['v']-data[1]['i'], y=data[1]['i'], 
                                      alpha=[0.6]*len(data[1]['i']), c=mass_colors(data[1]['imass'], 'single')))
# same, but for Gaussian model:
gsource = ColumnDataSource(data=dict(x=gdata[0]['v']-gdata[0]['i'], y=gdata[0]['i'], 
                                     alpha=[0.0]*len(gdata[0]['i']), c=mass_colors(gdata[0]['imass'], 'gauss')))
gfsource = ColumnDataSource(data=dict(x=gdata[1]['v']-gdata[1]['i'], y=gdata[1]['i'], 
                                      alpha=[0.0]*len(gdata[1]['i']), c=mass_colors(gdata[1]['imass'], 'gauss')))

# same, but for Flat model:
flsource = ColumnDataSource(data=dict(x=fldata[0]['v']-fldata[0]['i'], y=fldata[0]['i'], 
                                     alpha=[0.0]*len(fldata[0]['i']), c=mass_colors(fldata[0]['imass'], 'flat')))
flfsource = ColumnDataSource(data=dict(x=fldata[1]['v']-fldata[1]['i'], y=fldata[1]['i'], 
                                      alpha=[0.0]*len(fldata[1]['i']), c=mass_colors(fldata[1]['imass'], 'flat')))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Now actually plot data:

# Place (real) data on CMD if provided:
if args.photfile:
    plot_CMD.scatter('x', 'y', source=phot_source, alpha=0.6)

# Draw CMD scatter for models:
plot_CMD.scatter('x', 'y', source=source, line_color=None, fill_alpha='alpha', fill_color='c')
plot_CMD.scatter('x', 'y', source=fsource, line_color=None, fill_alpha='alpha', fill_color='c')

plot_CMD.scatter('x', 'y', source=gsource, line_color=None, fill_alpha='alpha', fill_color='c')
plot_CMD.scatter('x', 'y', source=gfsource, line_color=None, fill_alpha='alpha', fill_color='c')

plot_CMD.scatter('x', 'y', source=flsource, line_color=None, fill_alpha='alpha', fill_color='c')
plot_CMD.scatter('x', 'y', source=flfsource, line_color=None, fill_alpha='alpha', fill_color='c')

# x, y axis labels:
plot_CMD.xaxis.axis_label = x_label
plot_CMD.yaxis.axis_label = y_label

#color_bar = ColorBar(color_mapper=color_mapper,
#                     label_standoff=12, border_line_color=None, location=(0,0))

#plot_CMD.add_layout(color_bar, 'right')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set up widgets...

lage = Slider(title="log(age)", value=8.9, start=8.0, end=10.0, step=0.1)
vvc = Slider(title=r"V/Vc", value=0.0, start=0.0, end=0.6, step=0.1)
feh = Slider(title="[Fe/H]", value=-0.30, start=-0.50, end=-0.30, step=0.10)
dmod = Slider(title="Distance modulus [mag]", value=0.0, start=0, end=30, step=0.01)
bf = Slider(title="Binary Fraction", value=0.0, start=0, end=0.3, step=0.1)
av = Slider(title=r"Extinction", value=0.0, start=0, end=0.2, step=0.1)
mi_slider = RangeSlider(start=0.1, end=8.0, value=(0.1, 8.0), step=0.1, title="Initial Mass")
alpha = Slider(title=r"Fixed Alpha", value=0.6, start=0.0, end=1, step=0.1)
galpha = Slider(title=r"Gauss Alpha", value=0.0, start=0.0, end=1, step=0.1)
flalpha = Slider(title=r"Flat Alpha", value=0.0, start=0.0, end=1, step=0.1)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Now handle updating/callbacks for widgets & plotted data:

def update_data(attrname, old, new):
    
    # Get the current slider values
    lage_val = float("{:.2f}".format(lage.value))
    vvc_val = float("{:.1f}".format(vvc.value))
    feh_val = float("{:.2f}".format(feh.value))
    bf_val = float("{:.2f}".format(bf.value))   
    mu_val = float("{:.2f}".format(dmod.value))
    av_val = float("{:.2f}".format(av.value)) 
    alpha_val = float("{:.2f}".format(alpha.value))
    galpha_val = float("{:.2f}".format(galpha.value))
    flalpha_val = float("{:.2f}".format(flalpha.value))        
        

    # update CMD data:
    # get current range for initial mass slider:
    mis, mie = mi_slider.value

    # Get data for the single v/vc models:
    data = get_mag_mass(feh_val, vvc_val, bf_val, av_val, lage_val)
    # for the Gaussian dist. model:
    gdata = get_mag_mass(feh_val, 'gauss', bf_val, av_val, lage_val)
    # for the Flat dist. model:
    fldata = get_mag_mass(feh_val, 'flat', bf_val, av_val, lage_val)

    # re-assign source data...
    # data for single vvc model:
    source.data = dict(x=data[0]['v']-data[0]['i'], y=data[0]['i']+mu_val, 
                       alpha=[alpha_val]*len(data[0]['i']), c=mass_colors(data[0]['imass'], 'single'))
    # full data (never but via initial mass):
    fsource.data = dict(x=data[1]['v']-data[1]['i'], y=data[1]['i']+mu_val, 
                        alpha=[alpha_val]*len(data[1]['i']), c=mass_colors(data[1]['imass'], 'single'))
    gsource.data = dict(x=gdata[0]['v']-gdata[0]['i'], y=gdata[0]['i']+mu_val, 
                        alpha=[galpha_val]*len(gdata[0]['i']), c=mass_colors(gdata[0]['imass'], 'gauss'))
    gfsource.data = dict(x=gdata[1]['v']-gdata[1]['i'], y=gdata[1]['i']+mu_val, 
                         alpha=[galpha_val]*len(gdata[1]['i']), c=mass_colors(gdata[1]['imass'], 'gauss'))
    flsource.data = dict(x=fldata[0]['v']-fldata[0]['i'], y=fldata[0]['i']+mu_val, 
                        alpha=[flalpha_val]*len(gdata[0]['i']), c=mass_colors(fldata[0]['imass'], 'flat'))
    flfsource.data = dict(x=fldata[1]['v']-fldata[1]['i'], y=fldata[1]['i']+mu_val, 
                         alpha=[flalpha_val]*len(fldata[1]['i']), c=mass_colors(fldata[1]['imass'], 'flat'))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Update param values as widgets change:
for w in [lage, vvc, feh, bf, av, dmod, alpha, galpha, flalpha]:
    
    w.on_change('value', update_data)

# update initial mass range on change of slider:
mi_slider.on_change('value', update_data)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Set up layouts and add to document
inputs = widgetbox(lage, vvc, feh, bf, av, dmod, mi_slider, alpha, galpha, flalpha)
curdoc().add_root(row(inputs, plot_CMD, width=800))
curdoc().title = "MATCH Explorer"
