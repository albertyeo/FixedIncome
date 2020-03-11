import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy import interpolate
import matplotlib.pylab as plt
import datetime as dt
from scipy.optimize import least_squares
from math import log, sqrt, exp
from scipy.optimize import fsolve
from scipy import integrate


data_SABR=pd.read_csv('data_SABR.csv', header = 0, index_col = 0)
data_DD=pd.read_csv('data_DD.csv', header = 0, index_col = 0)
#------------------------------------------------------------------------------
def Black76Call(S, K, disc, sigma, T):
    d1 = (np.log(S/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return disc*(S*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Put(S, K, disc, sigma, T):
    d1 = (np.log(S/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return disc*(K*norm.cdf(-d2) - S*norm.cdf(-d1))

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*log(F/X)
        zhi = log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma
#------------------------------------------------------------------------------
data_interpolate = pd.DataFrame({'Expiry':['1Y', '1.5Y', '2Y', '2.5Y', '3Y','3.5Y', '4Y', '4.5Y', '5Y'],
                                 'Tenor': ['10Y']*9,
                                 'Alpha': np.NaN,
                                 'Rho': np.NaN,
                                 'Nu': np.NaN,
                                 'Beta': np.NaN,
                                 'Sigma': np.NaN}, index = np.arange(1, 10))

data_interpolate.loc[[1, 9], ['Alpha', 'Rho', 'Nu']] = data_SABR.loc[[4, 9],
                                                                     ['Alpha', 'Rho', 'Nu']].values
data_interpolate.loc[[1, 9], ['Beta', 'Sigma']] = data_DD.loc[[4, 9], 
                                                              ['Beta', 'Sigma']].values
data_interpolate = data_interpolate.interpolate()
#------------------------------------------------------------------------------
data_interpolate2 = pd.DataFrame({'Expiry':['1Y','1.25Y', '1.5Y','1.75Y', '2Y','2.25Y', '2.5Y', '2.75Y','3Y','3.25Y','3.5Y','3.75Y', '4Y','4.25Y', '4.5Y','4.25Y', '5Y'],
                                 'Tenor': ['2Y']*17,
                                 'Alpha': np.NaN,
                                 'Rho': np.NaN,
                                 'Nu': np.NaN,
                                 'Beta': np.NaN,
                                 'Sigma': np.NaN}, index = np.arange(1, 18))

data_interpolate2.loc[[1, 17], ['Alpha', 'Rho', 'Nu']] = data_SABR.loc[[1, 6], 
                                                                       ['Alpha', 'Rho', 'Nu']].values
data_interpolate2.loc[[1, 17], ['Beta', 'Sigma']] = data_DD.loc[[1, 6], 
                                                                ['Beta', 'Sigma']].values
data_interpolate2 = data_interpolate2.interpolate()

#------------------------------------------------------------------------------
data_interpolate3 = pd.DataFrame({'Expiry':['5Y','5.25Y','5.5Y', '5.75Y','6Y','6.25Y', '6.5Y','6.75Y', '7Y','7.25Y','7.5Y','7.75Y', '8Y', '8.25Y','8.5Y', '8.75Y','9Y','9.25Y', '9.5Y','9.75Y', '10Y'],
                                 'Tenor': ['2Y']*21,
                                 'Alpha': np.NaN,
                                 'Rho': np.NaN,
                                 'Nu': np.NaN,
                                 'Beta': np.NaN,
                                 'Sigma': np.NaN}, index = np.arange(17, 38))

data_interpolate3.loc[[17, 37], ['Alpha', 'Rho', 'Nu']] = data_SABR.loc[[6, 11], 
                                                                       ['Alpha', 'Rho', 'Nu']].values
data_interpolate3.loc[[17, 37], ['Beta', 'Sigma']] = data_DD.loc[[6, 11], 
                                                                ['Beta', 'Sigma']].values
data_interpolate3 = data_interpolate3.interpolate()

#------------------------------------------------------------------------------
print(data_interpolate)
print(data_interpolate2)
print(data_interpolate3)
#------------------------------------------------------------------------------
#DataFrame

IRSData = pd.read_excel (r'IR Data.xlsx', sheet_name='IRS')
data_DF = pd.read_csv('Discount Factors.csv', header = 0, index_col = 0)
data_DF2 = pd.read_csv('Discount Factors 2.csv', header = 0, index_col = 0)
data_CMS10 = pd.read_csv('CMS10ySwap.csv', header = 0, index_col = 0)
data_CMS2 = pd.read_csv('CMS2ySwap.csv', header = 0, index_col = 0)
data_FS = pd.read_csv('Forward Swap.csv', header = 0, index_col = 0)
#------------------------------------------------------------------------------

def IRR(x,N,m):
    IRR=np.zeros(N*m)
    IRRS=0
    for i in range(N*m):
        IRR[i]= 1/m / (1+x/m)**i
    IRRS=np.sum(IRR[0:20])
    return IRRS

def IRRf(x,N,m):
    dx = 0.05 * x
    IRRplus= IRR(x+dx,N,m)
    IRRminus = IRR (x-dx,N,m)
    IRRf = (IRRplus - IRRminus) / (2*dx)
    return IRRf

def IRRff(x,N,m):
    dx = 0.05 * x
    IRRplus= IRR(x+dx,N,m)
    IRRx = IRR(x,N,m)
    IRRminus = IRR (x-dx,N,m)
    IRRff = (IRRplus - 2*IRRx + IRRminus) / (dx**2)
    return IRRff

def integral1(x,N,m,F,disc,sigma,T):
    term1 = IRR(x, N, m) * (-3 / 16 * x ** (-3/4))
    term2 = IRRff(x, N, m) * (x**(1/4) - 0.2)
    term3 = 2 * IRRf(x, N, m) * (1/4) * x ** (-3/4)
    term4 = 2 * IRRf(x, N, m) ** 2 * (x ** (1/4) - 0.2)
    h= (term1 - term2 - term3) / (IRR(x,N,m)**2) + term4 / (IRR(x,N,m)**3)
    Vrec=Black76Put(F, x, disc, sigma, T)
    return h*Vrec

def integral2(x,N,m,F,disc,sigma,T):
    term1 = IRR(x, N, m) * (-3 / 16 * x ** (-3/4))
    term2 = IRRff(x, N, m) * (x**(1/4) - 0.2)
    term3 = 2 * IRRf(x, N, m) * (1/4) * x ** (-3/4)
    term4 = 2 * IRRf(x, N, m) ** 2 * (x ** (1/4) - 0.2)
    h= (term1 - term2 - term3) / (IRR(x,N,m)**2) + term4 / (IRR(x,N,m)**3)
    Vpay=Black76Call(F, x, disc, sigma, T)
    return h*Vpay

def hf(x, N, m):
    term1 = IRR(x, N, m) * (1/4) * x ** (-3 / 4)
    term2 = (x ** (1/4) - 0.2) * IRRf(x, N, m)
    return (term1 - term2) / (IRR(x, N, m)**2)
#------------------------------------------------------------------------------
print(data_FS)
print(data_DF)
print(data_interpolate)

F = data_FS.loc[9, 'Forward Swap']
D = data_DF.loc[9, 'DF_OIS']
alpha = data_interpolate.loc[9, 'Alpha']
beta = 0.9
rho = data_interpolate.loc[9, 'Rho']
nu = data_interpolate.loc[9, 'Nu']
sigmasabr = SABR(F, 0.0016, 5, alpha, beta, rho, nu)

#Question 1
term1 = F ** (1/4) - 0.2
term2 = hf(0.0016, 15, 2) * (Black76Call(F, 0.0016, IRR(F, 15, 2), sigmasabr, 5) - Black76Put(F, 0.0016, IRR(F, 15, 2), sigmasabr, 5))
term3 = integrate.quad(lambda x: integral1(x, 15, 2, F, IRR(F, 15, 2), sigmasabr, 5), 0, 0.0016)
term4 = integrate.quad(lambda x: integral2(x, 15, 2, F, IRR(F, 15, 2), sigmasabr, 5), 0.0016, np.inf)
PVoption = (term1 + term2 + np.sum(term3 + term4)) * D

#Question 2
term5 = hf(0.0016, 15, 2) * Black76Call(F, 0.0016, IRR(F, 15, 2), sigmasabr, 5)
term6 = integrate.quad(lambda x: integral2(x, 15, 2, F, IRR(F, 15, 2), sigmasabr, 5), 0.0016, np.inf)
PVoption2 = (term5 + np.sum(term6)) * D

print(PVoption)
print(PVoption2)
