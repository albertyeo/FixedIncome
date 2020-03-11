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
#For IRR & Integral function

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
    h= (-IRRff(x,N,m) * x - 2 * IRRf(x,N,m) ) / (IRR(x,N,m)**2) + 2 * IRRf(x,N,m)**2 * x / IRR(x,N,m)**3
    Vrec=Black76Put(F, x, disc, sigma, T)
    return h*Vrec

def integral2(x,N,m,F,disc,sigma,T):
    h= (-IRRff(x,N,m) * x - 2 * IRRf(x,N,m) ) / (IRR(x,N,m)**2) + 2 * IRRf(x,N,m)**2 * x / IRR(x,N,m)**3
    Vpay=Black76Call(F, x, disc, sigma, T)
    return h*Vpay

#------------------------------------------------------------------------------
#CMS10y & CMS2y

F=data_CMS10['Forward Swap CMS10y']
CMS10yrTenor=np.arange(0.5,5.5,0.5)

DF_OIS=data_DF['DF_OIS']
CMS10yrLeg=[0]*10
PVCMS10=0
for i in range(len(CMS10yrTenor)):
    term1,term2=0,0
    if i==0:
        sigmasabr=SABR(F[i], F[i], CMS10yrTenor[i], data_interpolate.loc[1, 'Alpha'], 0.9, data_interpolate.loc[1, 'Rho'], data_interpolate.loc[1, 'Nu'])
    else:
        sigmasabr=SABR(F[i], F[i], CMS10yrTenor[i], data_interpolate.loc[i, 'Alpha'], 0.9, data_interpolate.loc[i, 'Rho'], data_interpolate.loc[i, 'Nu'])
    term1=integrate.quad(lambda x:integral1(x,10,2,F[i],IRR(F[i],10,2),sigmasabr,CMS10yrTenor[i]),0,F[i])
    term2=integrate.quad(lambda x:integral2(x,10,2,F[i],IRR(F[i],10,2),sigmasabr,CMS10yrTenor[i]),F[i],np.inf)
    CMS10yrLeg[i]= F[i]+np.sum(term1 + term2)
PVCMS10=0.5*np.sum(DF_OIS[i]*CMS10yrLeg[i])
print('PV of a leg receiving CMS10y semi-annually over the next 5 years:',PVCMS10)

F1=data_CMS2['Forward Swap CMS2y']
CMS2yrTenor=np.arange(0.25,10.25,0.25)

DF_OIS2=data_DF2['DF_OIS']
CMS2yrLeg=[0]*40
PVCMS2=0
for i in range(len(CMS2yrTenor)):
    term1,term2=0,0
    if i<3:
        sigmasabr=SABR(F1[i], F1[i], CMS2yrTenor[i], data_interpolate2.loc[1, 'Alpha'], 0.9, data_interpolate2.loc[1, 'Rho'], data_interpolate2.loc[1, 'Nu'])
    elif i>=3 and i<19:
        sigmasabr=SABR(F1[i], F1[i], CMS2yrTenor[i], data_interpolate2.loc[i-2, 'Alpha'], 0.9, data_interpolate2.loc[i-2, 'Rho'], data_interpolate2.loc[i-2, 'Nu'])
    else:
        sigmasabr=SABR(F1[i], F1[i], CMS2yrTenor[i], data_interpolate3.loc[i-2, 'Alpha'], 0.9, data_interpolate3.loc[i-2, 'Rho'], data_interpolate3.loc[i-2, 'Nu'])
    term1=integrate.quad(lambda x:integral1(x,2,4,F1[i],IRR(F1[i],2,4),sigmasabr,CMS2yrTenor[i]),0,F1[i])
    term2=integrate.quad(lambda x:integral2(x,2,4,F1[i],IRR(F1[i],2,4),sigmasabr,CMS2yrTenor[i]),F1[i],np.inf)
    CMS2yrLeg[i]= F1[i]+np.sum(term1 + term2)
PVCMS2=0.25*np.sum(DF_OIS2[i]*CMS2yrLeg[i])
print('PV of a leg receiving CMS2y quarterly over the next 10 years:',PVCMS2)


#------------------------------------------------------------------------------
#Forward Swap Rates and CMS rate Comparison
    
fs=data_FS['Expiries'].values
st=data_FS['Tenors'].values
FSR=data_FS['Forward Swap'].values
CMSLeg=[0]*15
for i in range(15):
    idx=int(fs[i]*2) #DF idx of first payment
    lc=int(st[i]*2)  #leg count
    term1,term2=0,0
    sigmasabr=SABR(FSR[i], FSR[i], fs[i], data_SABR.loc[i, 'Alpha'], 0.9, data_SABR.loc[i, 'Rho'], data_SABR.loc[i, 'Nu'])
    term1=integrate.quad(lambda x:integral1(x,int(st[i]),2,FSR[i],IRR(FSR[i],int(st[i]),2),sigmasabr,fs[i]),0,FSR[i])
    term2=integrate.quad(lambda x:integral2(x,int(st[i]),2,FSR[i],IRR(FSR[i],int(st[i]),2),sigmasabr,fs[i]),FSR[i],np.inf)
    CMSLeg[i]= FSR[i]+np.sum(term1 + term2)
data_FS['CMS Rate']=CMSLeg

data_FS.drop(data_FS.columns[data_FS.columns=='PVBP'],inplace=True,axis=1)
print(data_FS)

plt.plot(st[0:5],FSR[0:5],label='1y Expiry FSR')
plt.plot(st[0:5],CMSLeg[0:5],label='1y Expiry CMS')
plt.xlabel('Tenor')
plt.ylabel('Rates')
plt.title('Forward Swap Rates vs CMS Rates - 1y Expiry')
plt.legend()
plt.show()

plt.plot(st[5:10],FSR[5:10],label='5y Expiry FSR')
plt.plot(st[5:10],CMSLeg[5:10],label='5y Expiry CMS')
plt.xlabel('Tenor')
plt.ylabel('Rates')
plt.title('Forward Swap Rates vs CMS Rates - 5y Expiry')
plt.legend()
plt.show()

plt.plot(st[10:15],FSR[10:15],label='10y Expiry FSR')
plt.plot(st[10:15],CMSLeg[10:15],label='10y Expiry CMS')
plt.xlabel('Tenor')
plt.ylabel('Rates')
plt.title('Forward Swap Rates vs CMS Rates - 10y Expiry')
plt.legend()
plt.show()
