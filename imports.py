# Data manipulation
import pandas as pd
import numpy as np
from numpy import linalg

# Support libs
import glob
import json
import datetime
from scipy import stats
import os

import requests
import subprocess

import re
import random

# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

# Statistical tools
from scipy.spatial.distance import squareform, pdist
from scipy import stats
from scipy.stats import gamma, pearsonr
from scipy import signal as sp_signal

from sklearn import ensemble
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import statsmodels.api as sm

### Zip Code database API
from uszipcode import ZipcodeSearchEngine

# Display full column width
pd.set_option('display.max_colwidth', -1)
from IPython.display import display, HTML
import plotly
from plotly import tools
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.graph_objs as go
import colorlover as clv

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.offline.offline import _plot_html

import cufflinks as cf
plotly.offline.init_notebook_mode() # run at the start of every notebook
import plotly.offline as offline