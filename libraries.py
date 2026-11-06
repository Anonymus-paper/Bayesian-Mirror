import argparse
import os
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import gammaln
import sklearn
from sklearn.linear_model import Lasso, LassoCV
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.preprocessing import StandardScaler
from packaging.version import Version
from joblib import Parallel, delayed, parallel_config
from knockpy import smatrix

try:                                                
    from pydsdp.dsdp5 import dsdp                    
except ImportError as exc:                           
    raise ImportError(
        "knockpy's SDP solver needs pydsdp, provided by the scikit-dsdp "
        "package (pip install scikit-dsdp). See the README for build notes."
    ) from exc

LCV_KW = ({"alphas": 40} if Version(sklearn.__version__) >= Version("1.7")
          else {"n_alphas": 40})