#testing library

import numpy as np

import xarray as xr  # work with arrays with labeled axes
import xrscipy.signal as dsp  # xarray signal filtering etc.

import scipy as sps
from cdb_extras import xarray_support as cdbxr  # access to COMPASS Database (CDB)

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union


