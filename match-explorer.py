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
#from bokeh.palettes import Category20

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
parser.add_argument("-fc", "--filtercode", const='BTVT', nargs='?', type=str,
                    help="A code specifying the filter set. E.g., BTVT for Tycho B and V filters; will plot as B-V vs. V.")
parser.add_argument("-as", "--agestep", const=0.5, nargs='?', type=float,
                    help="The step size for the log age sliders.")
args = parser.parse_args()

def filtercode_converter(filtercode):
  if filtercode == "BTVT":
    return "Tycho_BTycho_V"
  elif filtercode == "F438F814":
    return "UVIS438WUVIS814W"
  elif filtercode == "F475F814":
    return "UVIS475WUVIS814W"

# convert filtercode string to proper format:
args.filtercode = filtercode_converter(args.filtercode)
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

def getdata(feh_val, vvc_val, bf_val, av_val, age_val, model):

    #if isinstance(vvc_val, str):
    if model == 'single':
        d = np.genfromtxt(glob.glob("models_sparse/bf{:.2f}_av{:.1f}_SFR0.0001_"\
                                    "t{:.1f}_{:.2f}_logZ{:.2f}_vvc{:.1f}_"\
                                    "{:s}.phot".format(bf_val, av_val, age_val,
                                                      age_val+0.02, feh_val,
                                                      vvc_val, args.filtercode))[0])
    else:
        d = np.genfromtxt(glob.glob("models_sparse/bf{:.2f}_av{:.1f}_SFR0.0001_"\
                                    "t{:.1f}_{:.2f}_logZ{:.2f}_vvc{:s}_"\
                                    "{:s}.phot".format(bf_val, av_val, age_val,
                                                      age_val+0.02, feh_val,
                                                      model, args.filtercode))[0])

    return d

def get_mag_mass(feh, vvc, bf, av, lage, model, limz=[0.1,8.0]):

    # gets magnitudes and initial masses for a given set of feh, v/vc, etc.:
    # (returns a potentially cut range of masses for highlighting purposes, 
    #  and the full range.)
    data = getdata(feh, vvc, bf, av, lage, model)
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
                  mpl.cm.autumn(mpl.colors.Normalize()(initm))]
    elif model == 'gauss':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.autumn(mpl.colors.Normalize()(initm))]
    elif model == 'flat':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.autumn(mpl.colors.Normalize()(initm))]

    return colors

def mass_colors_cut(cutm, model):
    # returns a list of rgb colors, used to help delineate initial masses:
    # (uses Spectral color map at the moment.)
    if model == 'single':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.BrBG(mpl.colors.Normalize()(cutm))]
    elif model == 'gauss':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.BrBG(mpl.colors.Normalize()(cutm))]
    elif model == 'flat':
        colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                  for r, g, b, _ in 255* \
                  mpl.cm.BrBG(mpl.colors.Normalize()(cutm))]

    return colors

def model_select(model_index):
  models = ['single', 'gauss', 'flat']
  return models[model_index]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Using feh = -0.30, v/vc = 0.0, bf = 0.0, av = 0.0, log age = 8.9 as defaults...

# Get data for the single v/vc models:
data = get_mag_mass(-0.30, 0.0, 0.0, 0.0, 8.5, model='single', limz=[1.2,1.7])
# for the Gaussian dist. model:
gdata = get_mag_mass(-0.30, 0.0, 0.0, 0.0, 8.0, model='single', limz=[1.2,1.7])
fldata = get_mag_mass(-0.30, 0.0, 0.0, 0.0, 8.5, model='single', limz=[1.2,1.7])

# Set up plots...
plot_CMD = figure(plot_height=800, plot_width=600,
                  tools="box_zoom,crosshair,pan,reset,save,wheel_zoom",
                  x_range=[min(gdata[1]['v']-gdata[1]['i']), max(gdata[1]['v']-gdata[1]['i'])], 
                  y_range=[max(gdata[1]['i']), min(gdata[1]['i'])])

# with black background:
plot_CMD.background_fill_color = "black"
plot_CMD.xgrid.visible = False
plot_CMD.ygrid.visible = False

# data for single vvc model:
source = ColumnDataSource(data=dict(x=data[0]['v']-data[0]['i'], y=data[0]['i'], 
                                    alpha=[0.6]*len(data[0]['i']), c=mass_colors_cut(data[0]['imass'], 'single')))
# full data (i.e., this cannot be cut via initial mass):
fsource = ColumnDataSource(data=dict(x=data[1]['v']-data[1]['i'], y=data[1]['i'], 
                                      alpha=[0.6]*len(data[1]['i']), c=mass_colors(data[1]['imass'], 'single')))
# same, but for Gaussian model:
gsource = ColumnDataSource(data=dict(x=gdata[0]['v']-gdata[0]['i'], y=gdata[0]['i'], 
                                     alpha=[0.6]*len(gdata[0]['i']), c=mass_colors_cut(gdata[0]['imass'], 'single')))
gfsource = ColumnDataSource(data=dict(x=gdata[1]['v']-gdata[1]['i'], y=gdata[1]['i'], 
                                      alpha=[0.6]*len(gdata[1]['i']), c=mass_colors(gdata[1]['imass'], 'single')))

# same, but for Flat model:
flsource = ColumnDataSource(data=dict(x=fldata[0]['v']-fldata[0]['i'], y=fldata[0]['i'], 
                                     alpha=[0.0]*len(fldata[0]['i']), c=mass_colors_cut(fldata[0]['imass'], 'single')))
flfsource = ColumnDataSource(data=dict(x=fldata[1]['v']-fldata[1]['i'], y=fldata[1]['i'], 
                                      alpha=[0.0]*len(fldata[1]['i']), c=mass_colors(fldata[1]['imass'], 'single')))


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set up widgets...
lage_step = args.agestep

# identical sets for 3 separate populations:
model1_txt = Div(text="""<b>Model 1</b>""",
width=200, height=20)

radio_button_group1 = RadioButtonGroup(
        labels=["Single", "Gaussian", "Flat"], active=0)

lage1 = Slider(title="log(age)", value=8.5, start=8.0, end=10.0, step=lage_step)
vvc1 = Slider(title=r"V/Vc", value=0.0, start=0.0, end=0.6, step=0.1)
feh1 = Slider(title="[Fe/H]", value=-0.30, start=-0.50, end=-0.30, step=0.10)
dmod1 = Slider(title="Distance modulus [mag]", value=0.0, start=0, end=30, step=0.01)
bf1 = Slider(title="Binary Fraction", value=0.0, start=0, end=0.3, step=0.1)
av1 = Slider(title=r"Extinction", value=0.0, start=0, end=0.2, step=0.1)
mi_slider1 = RangeSlider(start=0.1, end=8.0, value=(1.2, 1.7), step=0.1, title="Initial Mass")
alpha1 = Slider(title=r"Alpha", value=0.6, start=0.0, end=1, step=0.1)

model2_txt = Div(text="""<b>Model 2</b>""",
width=200, height=20)
radio_button_group2 = RadioButtonGroup(
        labels=["Single", "Gaussian", "Flat"], active=0)

lage2 = Slider(title="log(age)", value=8.0, start=8.0, end=10.0, step=lage_step)
vvc2 = Slider(title=r"V/Vc", value=0.0, start=0.0, end=0.6, step=0.1)
feh2 = Slider(title="[Fe/H]", value=-0.30, start=-0.50, end=-0.30, step=0.10)
dmod2 = Slider(title="Distance modulus [mag]", value=0.0, start=0, end=30, step=0.01)
bf2 = Slider(title="Binary Fraction", value=0.0, start=0, end=0.3, step=0.1)
av2 = Slider(title=r"Extinction", value=0.0, start=0, end=0.2, step=0.1)
mi_slider2 = RangeSlider(start=0.1, end=8.0, value=(1.2, 1.7), step=0.1, title="Initial Mass")
alpha2 = Slider(title=r"Alpha", value=0.6, start=0.0, end=1, step=0.1)

model3_txt = Div(text="""<b>Model 3</b>""",
width=200, height=20)
radio_button_group3 = RadioButtonGroup(
        labels=["Single", "Gaussian", "Flat"], active=0)

lage3 = Slider(title="log(age)", value=8.5, start=8.0, end=10.0, step=lage_step)
vvc3 = Slider(title=r"V/Vc", value=0.0, start=0.0, end=0.6, step=0.1)
feh3 = Slider(title="[Fe/H]", value=-0.30, start=-0.50, end=-0.30, step=0.10)
dmod3 = Slider(title="Distance modulus [mag]", value=0.0, start=0, end=30, step=0.01)
bf3 = Slider(title="Binary Fraction", value=0.0, start=0, end=0.3, step=0.1)
av3 = Slider(title=r"Extinction", value=0.0, start=0, end=0.2, step=0.1)
mi_slider3 = RangeSlider(start=0.1, end=8.0, value=(1.2, 1.7), step=0.1, title="Initial Mass")
alpha3 = Slider(title=r"Alpha", value=0.0, start=0.0, end=1, step=0.1)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Now actually plot data:

# Place (real) data on CMD if provided:
if args.photfile:
    plot_CMD.scatter('x', 'y', source=phot_source, alpha=0.6, line_color=None, fill_color='lavender')

# Draw CMD scatter for models:
# e.g., color ='c' refers to source's (above) 'c' argument.
cmd1 = plot_CMD.scatter('x', 'y', source=fsource, line_color=None, fill_alpha='alpha', fill_color='c')#'yellow')
plot_CMD.scatter('x', 'y', source=source, line_color=None, fill_alpha='alpha', fill_color='c')#'red')#'c')

plot_CMD.scatter('x', 'y', source=gfsource, line_color=None, fill_alpha='alpha', fill_color='c')#'yellow')
plot_CMD.scatter('x', 'y', source=gsource, line_color=None, fill_alpha='alpha', fill_color='c')#'blue')

plot_CMD.scatter('x', 'y', source=flfsource, line_color=None, fill_alpha='alpha', fill_color='c')#'yellow')
plot_CMD.scatter('x', 'y', source=flsource, line_color=None, fill_alpha='alpha', fill_color='c')#'magenta')

# x, y axis labels:
plot_CMD.xaxis.axis_label = x_label
plot_CMD.yaxis.axis_label = y_label

#color_bar = ColorBar(color_mapper=color_mapper,
#                     label_standoff=12, border_line_color=None, location=(0,0))

#plot_CMD.add_layout(color_bar, 'right')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Now handle updating/callbacks for widgets & plotted data:

def update_data(attrname, old, new):
    
    # Get the current slider values
    lage_val1 = float("{:.1f}".format(lage1.value))
    vvc_val1 = float("{:.1f}".format(vvc1.value))
    feh_val1 = float("{:.2f}".format(feh1.value))
    bf_val1 = float("{:.2f}".format(bf1.value))   
    mu_val1 = float("{:.2f}".format(dmod1.value))
    av_val1 = float("{:.2f}".format(av1.value)) 
    alpha_val1 = float("{:.2f}".format(alpha1.value))
    model1 = model_select(radio_button_group1.active)     

    # update CMD data:
    # get current range for initial mass slider:
    mis1, mie1 = mi_slider1.value

      # Get the current slider values
    lage_val2 = float("{:.1f}".format(lage2.value))
    vvc_val2 = float("{:.1f}".format(vvc2.value))
    feh_val2 = float("{:.2f}".format(feh2.value))
    bf_val2 = float("{:.2f}".format(bf2.value))   
    mu_val2 = float("{:.2f}".format(dmod2.value))
    av_val2 = float("{:.2f}".format(av2.value)) 
    alpha_val2 = float("{:.2f}".format(alpha2.value))   
    model2 = model_select(radio_button_group2.active)     

    # update CMD data:
    # get current range for initial mass slider:
    mis2, mie2 = mi_slider2.value

    # Get the current slider values
    lage_val3 = float("{:.1f}".format(lage3.value))
    vvc_val3 = float("{:.1f}".format(vvc3.value))
    feh_val3 = float("{:.2f}".format(feh3.value))
    bf_val3 = float("{:.2f}".format(bf3.value))   
    mu_val3 = float("{:.2f}".format(dmod3.value))
    av_val3 = float("{:.2f}".format(av3.value)) 
    alpha_val3 = float("{:.2f}".format(alpha3.value))   
    model3 = model_select(radio_button_group3.active)     

    # update CMD data:
    # get current range for initial mass slider:
    mis3, mie3 = mi_slider3.value

    # Get data for the single v/vc models:
    data = get_mag_mass(feh_val1, vvc_val1, bf_val1, av_val1, lage_val1, model=model1, limz=[mis1,mie1])
    # re-assign source data...
    # data for single vvc model:
    source.data = dict(x=data[0]['v']-data[0]['i'], y=data[0]['i']+mu_val1, 
                       alpha=[alpha_val1]*len(data[0]['i']), c=mass_colors_cut(data[0]['imass'], model1))
    # full data (never but via initial mass):
    fsource.data = dict(x=data[1]['v']-data[1]['i'], y=data[1]['i']+mu_val1, 
                        alpha=[alpha_val1]*len(data[1]['i']), c=mass_colors(data[1]['imass'], model1))


    # for the Gaussian dist. model:
    try:
        gdata = get_mag_mass(feh_val2, vvc_val2, bf_val2, av_val2, lage_val2, model=model2, limz=[mis2,mie2])

        gsource.data = dict(x=gdata[0]['v']-gdata[0]['i'], y=gdata[0]['i']+mu_val2, 
                        alpha=[alpha_val2]*len(gdata[0]['i']), c=mass_colors_cut(gdata[0]['imass'], model2))
        gfsource.data = dict(x=gdata[1]['v']-gdata[1]['i'], y=gdata[1]['i']+mu_val2, 
                         alpha=[alpha_val2]*len(gdata[1]['i']), c=mass_colors(gdata[1]['imass'], model2))
    except Exception as e:
      pass

    try:
        # for the Flat dist. model:
        fldata = get_mag_mass(feh_val3, vvc_val3, bf_val3, av_val3, lage_val3, model=model3, limz=[mis3,mie3])

        flsource.data = dict(x=fldata[0]['v']-fldata[0]['i'], y=fldata[0]['i']+mu_val3, 
                        alpha=[alpha_val3]*len(fldata[0]['i']), c=mass_colors_cut(fldata[0]['imass'], model3))
        flfsource.data = dict(x=fldata[1]['v']-fldata[1]['i'], y=fldata[1]['i']+mu_val3, 
                         alpha=[alpha_val3]*len(fldata[1]['i']), c=mass_colors(fldata[1]['imass'], model3))
    except Exception as e:
      pass
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Update param val1ues as widgets change:
for w in [lage1, vvc1, feh1, bf1, av1, dmod1, alpha1,
          lage2, vvc2, feh2, bf2, av2, dmod2, alpha2,
          lage3, vvc3, feh3, bf3, av3, dmod3, alpha3]:
    
    w.on_change('value', update_data)

# update initial mass range on change of slider:
mi_slider1.on_change('value', update_data)
mi_slider2.on_change('value', update_data)
mi_slider3.on_change('value', update_data)

radio_button_group1.on_change('active', update_data)
radio_button_group2.on_change('active', update_data)
radio_button_group3.on_change('active', update_data)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Set up layouts and add to document
inputs = widgetbox(radio_button_group1, lage1, vvc1, feh1, bf1, av1, dmod1, mi_slider1, alpha1,
                   radio_button_group2, lage2, vvc2, feh2, bf2, av2, dmod2, mi_slider2, alpha2,
                   radio_button_group3, lage3, vvc3, feh3, bf3, av3, dmod3, mi_slider3, alpha3)
curdoc().add_root(row(inputs, plot_CMD, width=1600))
curdoc().title = "MATCH Explorer"
