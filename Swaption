import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

########## OIS ###########
def OISdiscountFactor(f, N):
    #f = Forward rate
    #N = Number of years
    return 1 / ((1 + f/360)**(360*N))

def fForward(f, N):
    #f = Forward rate
    return ((1 + f/360)**(360*N)) - 1

dataOIS = pd.read_excel('IR Data.xlsx', sheet_name = 'OIS', usecols = ['Tenor', 'Rate'])

dataOIS['Tenor'] = dataOIS['Tenor'].str[:-1]
dataOIS.iloc[0, 0] = 0.5
dataOIS[['Tenor']] = dataOIS[['Tenor']].apply(pd.to_numeric).apply(lambda x: x/1, axis = 1)
dataOIS = dataOIS.set_index('Tenor')

dataOIS.loc[0.5, 'f'] = ((dataOIS.loc[0.5, 'Rate'] + 1)**(1/720) - 1) * 360
dataOIS.loc[0.5, 'OIS Discount Factor'] = OISdiscountFactor (dataOIS.loc[0.5, 'f'], 0.5)


dataOIS.loc[1.0, 'f'] = ((dataOIS.loc[1.0, 'Rate'] + 1)**(1/360) - 1) * 360
dataOIS.loc[1.0, 'OIS Discount Factor'] = OISdiscountFactor (dataOIS.loc[1.0, 'f'], 1)

for i in range(2, 11):
    start = dataOIS.index[i-1]
    end = dataOIS.index[i]
    period = end - start
    if period == 1.0:
        term1 = dataOIS.loc[1.0:start, 'OIS Discount Factor'].values

        result = fsolve(lambda x: ((np.sum(term1) + term1[-1] * 
                                    OISdiscountFactor(x, 1)) * 
                                   dataOIS.loc[end, 'Rate']) -
                                   1 + 
                                   term1[-1] *
                                   OISdiscountFactor(x, 1), 0.003)[0]

    else:
        term1 = dataOIS.loc[1.0:start, 'OIS Discount Factor'].values

        result = fsolve(lambda x: ((np.sum(term1) + term1[-1] * 
                                    np.sum([OISdiscountFactor(x, j) for j in range(1, int(period)+1)])) *
                                    dataOIS.loc[end, 'Rate']) -
                                  (1 - 
                                   term1[-1] *
                                   OISdiscountFactor(x, period)), 0.003)
    
    dataOIS.loc[end, 'f'] = result
    dataOIS.loc[end, 'OIS Discount Factor'] = (dataOIS.loc[start, 'OIS Discount Factor'] *
                                               OISdiscountFactor(result, period))
    
OIS = pd.DataFrame(columns = dataOIS.columns, index = np.arange(0.5, 30.5, 0.5))
OIS.loc[:, :] = dataOIS.loc[:, :]
OIS[['f']] = OIS[['f']].fillna('N/A')
OIS[['OIS Discount Factor']] = OIS[['OIS Discount Factor']].fillna('N/A')

for i in range(58, -1, -1):
    if OIS.iloc[i, 1] == 'N/A':
        OIS.iloc[i, 1] = OIS.iloc[i+1, 1]

for i in range(1, len(OIS.index)):
    if OIS.iloc[i, 2] == 'N/A':
        OIS.iloc[i, 2] = OIS.iloc[i-1, 2] * OISdiscountFactor(OIS.iloc[i, 1], 0.5)

print('OIS Discount Factor with semi-annual frequency')
print(OIS)
OIS.to_csv('OIS Data.csv')


OIS2 = pd.DataFrame(columns = OIS.columns, index = np.arange(0.25, 30.25, 0.25))
OIS2.loc[:, :] = OIS.loc[:, :]
OIS2[['f']] = OIS2[['f']].fillna('N/A')
OIS2[['OIS Discount Factor']] = OIS2[['OIS Discount Factor']].fillna('N/A')

for i in range(118, 0, -1):
    if OIS2.iloc[i, 1] == 'N/A':
        OIS2.iloc[i, 1] = OIS2.iloc[i+1, 1]
        
for i in range(2, len(OIS2.index)):
    if OIS2.iloc[i, 2] == 'N/A':
        OIS2.iloc[i, 2] = OIS2.iloc[i-1, 2] * OISdiscountFactor(OIS2.iloc[i, 1], 0.25)

temp = OIS2.loc[0.50, 'f']
OIS2.loc[0.25, 'f'] = temp
temp2 = OIS2.loc[0.50, 'OIS Discount Factor'] / OISdiscountFactor(temp, 0.25)
OIS2.loc[0.25, 'OIS Discount Factor'] = temp2

print('\n')
print('OIS Discount Factor with quarter-annual frequency')
print(OIS2.loc[:10, :])
print(OIS2.loc[10:20, :])
print(OIS2.loc[20:, :])
OIS2.to_csv('OIS Data 2.csv')



df1 = pd.read_excel (r'IR Data.xlsx', sheet_name='OIS')
df1['Year']=df1.Tenor.str.extract('(\d+)').astype(float)
df1.loc[0,'Year']=0.5
Tenor=np.linspace(0.5,30,60)

data_OIS = pd.read_csv('OIS Data.csv', header = 0, index_col = 0)
data_OIS2 = pd.read_csv('OIS Data 2.csv', header = 0, index_col = 0)
DF0=data_OIS['OIS Discount Factor'].values
DF02=data_OIS2.loc[0.25:12,'OIS Discount Factor'].values
#print(DF0)
print('lenDF01:',len(DF02))

########## IRS ###########
df2 = pd.read_excel (r'IR Data.xlsx', sheet_name='IRS')
df2['Year']=df1.Tenor.str.extract('(\d+)').astype(float)
df2.loc[0,'Year']=0.5
#print(df2)
R1=df2['Rate'].values
Y1=df2['Year'].values
DF1=np.zeros(len(Tenor))
DF1[0]=1/(1+ R1[0]/2)

t1=(DF0[0]+DF0[1])*R1[1]
t2=DF0[0]*2*(1/DF1[0]-1)
t3= 2*DF0[1]
t4= (t1-t2) / t3 + 1
DF1[1]= DF1[0]/t4

for i in range(2,11):
    p=int(Y1[i-1]*2) #p=legs with known DF
    q=int(Y1[i]*2)   #q=total legs
    PVfix=np.sum(DF0[0:q])*R1[i]
    flt=np.zeros(p)
    flt[0]=DF0[0]*2*(1/DF1[0] -1)
    for k in range(1,p):
        flt[k]=DF0[k]*2*(DF1[k-1]/DF1[k] -1)
    r=q-p   #r=legs with unknown DF
    
    def f(x):
        fun=np.zeros(r)
        term=np.zeros(r+1)
        term[0]=DF0[p]*2*(DF1[p-1]/x[0] -1)
        for z in range(r-1):
            term[z+1]=DF0[p+z+1]*2*(x[z]/x[z+1] -1)
            if z>0:
                fun[z+1]=2*x[z]-x[z+1]-x[z-1]
            else:
                fun[z+1]=2*x[z]-x[z+1]-DF1[p-1]
        fun[0]=PVfix-np.sum(flt[0:p])-np.sum(term[0:r+1])
        return fun
    
    DF1[p:q]=fsolve(f,np.ones(r))

QuarterTenor = np.arange(.25,12.25,.25)
f2=interp1d(Tenor,DF1,kind='slinear')
DF2=np.zeros(len(QuarterTenor))
DF2[0]=(DF1[0]+1)/2
DF2[1]=DF1[0]
DF2[2:len(QuarterTenor)]=f2(QuarterTenor[2:len(QuarterTenor)])
print('lenDF2:',len(DF2))



##### LIBOR from Libor DF ########
Libor=np.zeros(len(Tenor))
for i in range(len(Tenor)):
    Libor[i]=( 1/DF1[i] -1 ) / Tenor[i]
fwdLibor=np.zeros(len(Tenor))
fwdLibor[0]=Libor[0]
for i in range(1,len(Tenor)):
    fwdLibor[i]=2 * ( DF1[i-1]/DF1[i] -1 )
OISC=np.zeros(len(Tenor))
for i in range(len(Tenor)):
    OISC[i]=( 1/DF0[i] -1 ) / Tenor[i]

Libor2=np.zeros(len(QuarterTenor))
for i in range(len(QuarterTenor)):
    Libor2[i]=( 1/DF2[i] -1 ) / QuarterTenor[i]
fwdLibor2=np.zeros(len(QuarterTenor))
fwdLibor2[0]=Libor2[0]
for i in range(1,len(QuarterTenor)):
    fwdLibor2[i]=4 * ( DF2[i-1]/DF2[i] -1 )


Table=pd.DataFrame()
Table['Tenor']=Tenor
Table['DF_OIS']=DF0
Table['DF_LIBOR']=DF1
#Table['LIBOR']=Libor
Table['Forward LIBOR']=fwdLibor
print(Table)          

Table4=pd.DataFrame()
Table4['QTenor']=QuarterTenor
Table4['DF_OIS']=DF02
Table4['DF_LIBOR']=DF2
Table4['Forward LIBOR']=fwdLibor2   
print(Table4)         

plt.plot(Tenor,DF0, color = 'blue')
plt.xlabel('Tenor')
plt.ylabel('Discount Factor')
plt.title('OIS Discount Factor')
plt.show()

plt.plot(Tenor,DF1, color = 'red')
plt.xlabel('Tenor')
plt.ylabel('Discount Factor')
plt.title('LIBOR Discount Factor')
plt.show()


plt.plot(Tenor,DF0,label='OIS', color = 'blue')
plt.plot(Tenor,DF1,label='Libor', color = 'red')
plt.xlabel('Tenor')
plt.ylabel('Discount Factor')
plt.title('OIS & LIBOR Discount Factor')
plt.legend()
plt.show()
########## Forward Swap Rates ###########

start=[1,5,10]
swaptenor=[1,2,3,5,10]
Table2=pd.DataFrame()
n=0
for i in range(len(start)):
    for j in range(len(swaptenor)):
        Table2.loc[n,'Expiries']=start[i]
        Table2.loc[n,'Tenors']=swaptenor[j]
        n+=1


for i in range(15):
    fs=Table2['Expiries'].values
    st=Table2['Tenors'].values
    idx=int(fs[i]*2) #DF idx of first payment
    lc=int(st[i]*2)  #leg count
    ### using Libor
    SwPVfix=np.zeros(15)
    SwPVflt=np.zeros(15)
    FSR=np.zeros(15)
    SwPVfix[i]=0.5*np.sum(DF0[idx:idx+lc])
    temp=np.zeros(lc)
    for j in range(lc):
        temp[j]=DF0[idx+j]*fwdLibor[idx+j]
    SwPVflt[i]=0.5*np.sum(temp[0:lc])
    FSR[i]=SwPVflt[i]/SwPVfix[i]
    Table2.loc[i,'Forward Swap']=FSR[i]
    Table2.loc[i,'PVBP']=SwPVfix[i]
    #Table2.loc[i,'PVFloat']=SwPVflt[i]

CMS10yrTenor=np.arange(0.5,5.5,0.5)
Table3=pd.DataFrame()
Table3['Expiries']=CMS10yrTenor
for i in range(len(CMS10yrTenor)):
    idx=int(CMS10yrTenor[i]*2) #DF idx of first payment
    lc=int(10*2)  #leg count
    ### using Libor
    SwPVfix1=np.zeros(len(CMS10yrTenor))
    SwPVflt1=np.zeros(len(CMS10yrTenor))
    CMSFSR=np.zeros(len(CMS10yrTenor))
    SwPVfix1[i]=0.5*np.sum(DF0[idx:idx+lc])
    temp1=np.zeros(lc)
    for j in range(lc):
        temp1[j]=DF0[idx+j]*fwdLibor[idx+j]
    SwPVflt1[i]=0.5*np.sum(temp1[0:lc])
    CMSFSR[i]=SwPVflt1[i]/SwPVfix1[i]
    Table3.loc[i,'Forward Swap CMS10y']=CMSFSR[i]
    Table3.loc[i,'PVBP']=SwPVfix1[i]
    #Table2.loc[i,'PVFloat']=SwPVflt1[i]
    
CMS10yrTenor1=np.arange(0.5,8.5,0.5)
Table6=pd.DataFrame()
Table6['Expiries']=CMS10yrTenor1
for i in range(len(CMS10yrTenor1)):
    idx=int(CMS10yrTenor1[i]*2) #DF idx of first payment
    lc=int(10*2)  #leg count
    ### using Libor
    SwPVfix3=np.zeros(len(CMS10yrTenor1))
    SwPVflt3=np.zeros(len(CMS10yrTenor1))
    CMSFSR2=np.zeros(len(CMS10yrTenor1))
    SwPVfix3[i]=0.5*np.sum(DF0[idx:idx+lc])
    temp3=np.zeros(lc)
    for j in range(lc):
        temp3[j]=DF0[idx+j]*fwdLibor[idx+j]
    SwPVflt3[i]=0.5*np.sum(temp3[0:lc])
    CMSFSR2[i]=SwPVflt3[i]/SwPVfix3[i]
    Table6.loc[i,'Forward Swap CMS10y']=CMSFSR2[i]
    Table6.loc[i,'PVBP']=SwPVfix3[i]
    #Table2.loc[i,'PVFloat']=SwPVflt1[i]
    
CMS2yrTenor=np.arange(0.25,10.25,0.25)
Table5=pd.DataFrame()
Table5['Expiries']=CMS2yrTenor
for i in range(len(CMS2yrTenor)):
    idx=int(CMS2yrTenor[i]*4) #DF idx of first payment
    lc=int(2*4)  #leg count
    ### using Libor
    SwPVfix2=np.zeros(len(CMS2yrTenor))
    SwPVflt2=np.zeros(len(CMS2yrTenor))
    CMS2yFSR=np.zeros(len(CMS2yrTenor))
    SwPVfix2[i]=0.5*np.sum(DF02[idx:idx+lc])
    temp2=np.zeros(lc)
    for j in range(lc):
        temp2[j]=DF02[idx+j]*fwdLibor2[idx+j]
    SwPVflt2[i]=0.5*np.sum(temp2[0:lc])
    CMS2yFSR[i]=SwPVflt2[i]/SwPVfix2[i]
    Table5.loc[i,'Forward Swap CMS2y']=CMS2yFSR[i]
    Table5.loc[i,'PVBP']=SwPVfix2[i]
    
print(Table2)
print(Table3)
print(Table6)
print(Table5)
Table.to_csv('Discount Factors.csv')
Table4.to_csv('Discount Factors 2.csv')
Table2.to_csv('Forward Swap.csv')
Table3.to_csv('CMS10ySwap.csv')
Table5.to_csv('CMS2ySwap.csv')
Table6.to_csv('CMS10ySwap_v2.csv')
