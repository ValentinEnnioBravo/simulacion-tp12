import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter
import os

# Autodescarga
AD_dist = stats.gennorm.rvs(beta=39.87762517430417, loc=0.02008661643245054, scale=0.010103831003727536, size=4000)
def AD(): return np.random.choice(AD_dist)

# Costo Falla
CF_dist = stats.pearson3.rvs(skew=1.2672861956124433, loc=7955.175469640658, scale=3763.2127890221027, size=4000)
def CF(): return np.random.choice(CF_dist)

# Demanda Primer Semestre
DV_dist = stats.laplace_asymmetric.rvs(kappa=0.5078151312975224, loc=2575.569999998423, scale=452.20815824224394, size=4000)
def DV(): return np.random.choice(DV_dist)

# Demanda Segundo Semestre
DI_dist = stats.gumbel_r.rvs(loc=3876.0965773518615, scale=653.1078007257158, size=4000)
def DI(): return np.random.choice(DI_dist)

# Generacion diaria CC1
GD1_dist = stats.gennorm.rvs(beta=3.0613478036342627, loc=719.3923999911451, scale=127.21163991375852, size=4000)
def GD1(): return np.random.choice(GD1_dist)

# Generacion diaria CC2
GD2_dist = stats.gennorm.rvs(beta=4.9423185984406155, loc=610.589328055618, scale=124.80973709648973, size=4000)
def GD2(): return np.random.choice(GD2_dist)

# Generacion diaria TV
GDTV_dist = stats.gennorm.rvs(beta=5.703506370721026, loc=550.2687715652102, scale=124.52965269900714, size=4000)
def GDTV(): return np.random.choice(GDTV_dist)

# Potencia Perdida
PP_dist = stats.tukeylambda.rvs(lam=1.0018233713540219, loc=5.499999999999986, scale=2.5045584283850713, size=4000)
def PP(): return np.random.choice(PP_dist)

print(AD())
print(CF())
print(DV())
print(DI())
print(GD1())
print(GD2())
print(GDTV())
print(PP())
