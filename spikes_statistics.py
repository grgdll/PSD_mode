import pandas as pd
from matplotlib import pyplot as plt
import glob
import numpy as np
import optics_rig as orig
from scipy.interpolate import interp1d
from scipy import signal as signal
from scipy.stats import gamma, burr12

from iteration_utilities import flatten

import os
import glob
from scipy import optimize
import copy
import dill

import colorcet as cc

from scipy.integrate import trapz


  
