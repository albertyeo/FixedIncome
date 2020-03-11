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
#------------------------------------------------------------------------------
#PART 2
#------------------------------------------------------------------------------
def Black76Call(S, K, disc, sigma, T):
    d1 = (np.log(S/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return disc*(S*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Put(S, K, disc, sigma, T):
    d1 = (np.log(S/K)+sigma**2/2*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return disc*(K*norm.cdf(-d2) - S*norm.cdf(-d1))

def DisplacedDiffusionCall(S, K, disc, sigma, T, beta):
    A = S / beta
    B = K + (1 - beta) * S / beta
    C = sigma * beta
    d1 = (np.log(A/B)+C**2/2*T) / (C*np.sqrt(T))
    d2 = d1 - C*np.sqrt(T)
    return disc*(A*norm.cdf(d1) - B*norm.cdf(d2))

def DisplacedDiffusionPut(S, K, disc, sigma, T, beta):
    A = S / beta
    B = K + (1 - beta) * S / beta
    C = sigma * beta
    d1 = (np.log(A/B)+C**2/2*T) / (C*np.sqrt(T))
    d2 = d1 - C*np.sqrt(T)
    return disc*(B*norm.cdf(-d2) - A*norm.cdf(-d1))

def impliedCallVolatility(S, K, disc, price, T):
    impliedVol = fsolve(lambda x: price -
                        Black76Call(S, K, disc, x, T),
                        0.5)
    return impliedVol[0]

def impliedPutVolatility(S, K, disc, price, T):
    impliedVol = fsolve(lambda x: price -
                        Black76Put(S, K, disc, x, T),
                        0.5)
    return impliedVol[0]
#------------------------------------------------------------------------------
swaption = pd.read_excel (r'IR Data.xlsx', sheet_name='Swaption', skiprows = 2, header = 0)
swaption = swaption.iloc[0:15,0:13]
swaption.iloc[:, 2:] = swaption.iloc[:, 2:] / 100

data_fs = pd.read_csv('Forward Swap.csv', header = 0, index_col = 0)

bps = [0 if i == 7 else float(swaption.columns[i][0:-3]) / 10000
       for i in range(2, len(swaption.columns))]

strikes = pd.DataFrame(swaption.iloc[:, 0:2], index = swaption.index)
for i in range(len(swaption.columns[2:])):
    strikes.loc[:, swaption.columns[2 + i]] = data_fs.iloc[:, 2] + bps[i]
#------------------------------------------------------------------------------
#1
#------------------------------------------------------------------------------
def ddcalibration(x, S, strikes, disc, vols, T):
    err = 0.0
    for i, vol in enumerate(vols):
        price = DisplacedDiffusionCall(S, strikes[i], disc, x[1], T, x[0])
        err += (vol - impliedCallVolatility(S, strikes[i], disc, price, T))**2
        
    return err

data_DD = pd.DataFrame(swaption.iloc[:, 0:2], columns = swaption.columns, 
                       index = swaption.index)
data_DD.loc[:, 'Beta'] = np.NaN
data_DD.loc[:, 'Sigma'] = np.NaN

initialGuess = [0.2, 0.25]

for i in range(len(swaption.index)):
    S = data_fs.loc[i, 'Forward Swap']
    disc = data_fs.loc[i, 'PVBP']
    T = data_fs.iloc[i, 0]
    
    res = least_squares(lambda x: ddcalibration(x, 
                                                S, 
                                                strikes.iloc[i, 2:],
                                                disc,
                                                swaption.iloc[i, 2:],
                                                T),
                                                initialGuess,
                                                bounds = ((0, 0), (1, np.inf))
                                                )
    beta_dd = res.x[0]
    sigma_dd = res.x[1]
    
    data_DD.iloc[i, 13] = beta_dd
    data_DD.iloc[i, 14] = sigma_dd

    for j in range(2, 13):
        K = strikes.iloc[i, j]
        if j <= 8:        
            price_dd = DisplacedDiffusionPut(S, K, disc, sigma_dd, T, beta_dd)
            impliedvol_dd = impliedPutVolatility(S, K, disc, price_dd, T)
        
        elif j > 8:
            price_dd = DisplacedDiffusionCall(S, K, disc, sigma_dd, T, beta_dd)
            impliedvol_dd = impliedCallVolatility(S, K, disc, price_dd, T)
        
        data_DD.iloc[i, j] = impliedvol_dd
#------------------------------------------------------------------------------
#2
#------------------------------------------------------------------------------
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

def sabrcalibration(x, strikes, vols, F, T, beta):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2

    return err
    
data_SABR = pd.DataFrame(swaption.iloc[:, 0:2], columns = swaption.columns,
                         index = swaption.index)
data_SABR.loc[:, 'Alpha'] = np.NaN
data_SABR.loc[:, 'Rho'] = np.NaN
data_SABR.loc[:, 'Nu'] = np.NaN

beta_SABR = 0.9
initialGuess = [0.18, -0.45, 0.5]

for i in range(len(swaption.index)):
    res_SABR = least_squares(lambda x: sabrcalibration(x,
                                                       strikes.iloc[i, 2:],
                                                       swaption.iloc[i, 2:],
                                                       data_fs.loc[i, 'Forward Swap'],
                                                       data_fs.iloc[i, 0],
                                                       beta_SABR),
                                                       initialGuess)
        
    alpha = res_SABR.x[0]
    rho = res_SABR.x[1]
    nu = res_SABR.x[2]
    
    data_SABR.iloc[i, 13] = alpha
    data_SABR.iloc[i, 14] = rho
    data_SABR.iloc[i, 15] = nu
    
    for j in range(2, 13):
        data_SABR.iloc[i, j] = SABR(data_fs.loc[i, 'Forward Swap'],
                                    strikes.iloc[i, j],
                                    data_fs.iloc[i, 0],
                                    alpha,
                                    beta_SABR,
                                    rho,
                                    nu) 
#------------------------------------------------------------------------------
data_interpolate = pd.DataFrame({'Expiry':['1Y', '2Y', '3Y', '4Y', '5Y'],
                                 'Tenor': ['10Y']*5,
                                 'Alpha': np.NaN,
                                 'Rho': np.NaN,
                                 'Nu': np.NaN,
                                 'Beta': np.NaN,
                                 'Sigma': np.NaN}, index = np.arange(1, 6))

data_interpolate.loc[[1, 5], ['Alpha', 'Rho', 'Nu']] = data_SABR.loc[[4, 9],
                                                                     ['Alpha', 'Rho', 'Nu']].values
data_interpolate.loc[[1, 5], ['Beta', 'Sigma']] = data_DD.loc[[4, 9], 
                                                              ['Beta', 'Sigma']].values
data_interpolate = data_interpolate.interpolate()
#------------------------------------------------------------------------------
data_interpolate2 = pd.DataFrame({'Expiry':['5Y', '6Y', '7Y', '8Y', '9Y', '10Y'],
                                 'Tenor': ['10Y']*6,
                                 'Alpha': np.NaN,
                                 'Rho': np.NaN,
                                 'Nu': np.NaN,
                                 'Beta': np.NaN,
                                 'Sigma': np.NaN}, index = np.arange(5, 11))

data_interpolate2.loc[[5, 10], ['Alpha', 'Rho', 'Nu']] = data_SABR.loc[[9, 14], 
                                                                       ['Alpha', 'Rho', 'Nu']].values
data_interpolate2.loc[[5, 10], ['Beta', 'Sigma']] = data_DD.loc[[9, 14], 
                                                                ['Beta', 'Sigma']].values
data_interpolate2 = data_interpolate2.interpolate()
#------------------------------------------------------------------------------
data_CMS10 = pd.read_csv('CMS10ySwap.csv', header = 0, index_col = 0)
S2y10y = data_CMS10.loc[3,'Forward Swap CMS10y']
PVBP2y10y = data_CMS10.loc[3,'PVBP']
T2y10y=2

data_CMS10v2 = pd.read_csv('CMS10ySwap_v2.csv', header = 0, index_col = 0)
S8y10y = data_CMS10v2.loc[15,'Forward Swap CMS10y']
PVBP8y10y = data_CMS10v2.loc[15,'PVBP']
T8y10y=8

new_K=np.arange(0.01,0.09,0.01)
Table_Swaptions=pd.DataFrame(index=['Pay 2yx10y DD','Pay 2yx10y SABR','Rec 8yx10y DD','Rec 8yx10y SABR'],columns=new_K)


for K in new_K:
    DD_priceCall=0
    DD_pricePut=0
    DD_priceCall=DisplacedDiffusionCall(S2y10y, K, PVBP2y10y, data_interpolate.loc[2,'Sigma'], T2y10y, data_interpolate.loc[2,'Beta'])
    DD_pricePut=DisplacedDiffusionPut(S8y10y, K, PVBP8y10y, data_interpolate2.loc[8,'Sigma'], T8y10y, data_interpolate2.loc[8,'Beta'])
    
    sigma2y10y=0
    sigma8y10y=0
    sigma2y10y=SABR(S2y10y, K, T2y10y, data_interpolate.loc[2,'Alpha'], 0.9, data_interpolate.loc[2,'Rho'], data_interpolate.loc[2,'Nu'])
    sigma8y10y=SABR(S8y10y, K, T8y10y, data_interpolate2.loc[8,'Alpha'], 0.9, data_interpolate2.loc[8,'Rho'], data_interpolate2.loc[8,'Nu'])
    
    SABR_priceCall=0
    SABR_pricePut=0
    SABR_priceCall=Black76Call(S2y10y, K, PVBP2y10y, sigma2y10y, T2y10y)
    SABR_pricePut=Black76Put(S8y10y, K, PVBP8y10y, sigma8y10y, T8y10y)
    
    Table_Swaptions.loc['Pay 2yx10y DD',K]=DD_priceCall
    Table_Swaptions.loc['Rec 8yx10y DD',K]=DD_pricePut
    Table_Swaptions.loc['Pay 2yx10y SABR',K]=SABR_priceCall
    Table_Swaptions.loc['Rec 8yx10y SABR',K]=SABR_pricePut
#------------------------------------------------------------------------------
#Prints
print(strikes)
print(swaption)
print(data_DD)
print(data_SABR)
print(Table_Swaptions)
data_DD.to_csv('data_DD.csv')
data_SABR.to_csv('data_SABR.csv')
Table_Swaptions.to_csv('Table_Swaptions.csv')
print(data_interpolate)
print(data_interpolate2)
for i in range(len(swaption.index)):
    plt.figure()
    plt.plot(strikes.iloc[i, 2:], swaption.iloc[i, 2:13], 'go', label = 'Market')
    plt.plot(strikes.iloc[i, 2:], data_DD.iloc[i, 2:13], 'r:', label = 'DD Model')
    plt.plot(strikes.iloc[i, 2:], data_SABR.iloc[i, 2:13], 'b', label = 'SABR Model')
    plt.title('Swap ' + swaption.iloc[i, 0] + 'X' + swaption.iloc[i, 1])
    plt.xlabel('Strikes')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()
#------------------------------------------------------------------------------
