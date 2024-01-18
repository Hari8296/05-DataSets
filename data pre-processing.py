Name :- Hari singh r
batch Id :-DSWDMCOD 25082022 B


import pandas as pd # data manipulation 
import numpy as np # using numerical python
import seaborn as sen # data visulazation  
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

###########################################################################################################
1.	Prepare the dataset by performing the preprocessing techniques, to treat the outliers.

 outliers

data=pd.read_csv("D:/assignments of data science/DataSets/boston_data.csv")
data.dtypes

sen.boxplot(data.black)

IQR=data['black'].quantile(0.75)-data['black'].quantile(0.25)
IQR
lower_limit=data['black'].quantile(0.25)-(IQR*1.5)
upper_limit=data['black'].quantile(0.75)+(IQR*1.5)

outliers_data=np.where(data['black']>upper_limit,True,np.where(data['black']<lower_limit,True ,False))
data_trimmed=data.loc[~(outliers_data), ]
data.shape,data_trimmed.shape

sen.boxplot(data_trimmed.black)


data['data_replaced'] = pd.DataFrame(np.where(data['black'] > upper_limit, upper_limit, np.where(data['black'] < lower_limit, lower_limit, data['black'])))
sns.boxplot(data.data_replaced)

###########################################################################################################
 Find which columns of the given dataset with zero variance, explore various techniques used to remove the zero variance from the dataset to perform certain analysis. 

 remove the zero variance from the dataset 
 
df=pd.read_csv("D:/assignments of data science/DataSets/Z_dataset.csv")
df
df.var()
df.var()==0
df.var(axis=0)==0

###########################################################################################################

Imputation

dt=pd.read_csv("D:/assignments of data science/DataSets/claimants.csv")
dt
dt.isna().sum()
sen.boxplot(dt.CLMSEX)
mean_imputation=SimpleImputer(missing_values=np.nan,strategy='mean')
dt["CLMSEX"]=pd.DataFrame(mean_imputation.fit_transform(dt[["CLMSEX"]]))
dt.CLMSEX=dt.CLMSEX.fillna(dt.CLMSEX.mean())

dt["CLMSEX"].isna().sum()
sen.boxplot(dt.CLMINSUR)
median_imputation=SimpleImputer(missing_values=np.nan,strategy='median')
dt["CLMINSUR"]=pd.DataFrame(median_imputation.fit_transform(dt[["CLMINSUR"]]))
dt.CLMINSUR=dt.CLMINSUR.fillna(dt.CLMINSUR.median())

sen.boxplot(dt.SEATBELT)
median_imputation=SimpleImputer(missing_values=np.nan,strategy='median')
dt["SEATBELT"]=pd.DataFrame(median_imputation.fit_transform(dt[["SEATBELT"]]))
dt.SEATBELT=dt.SEATBELT.fillna(dt.SEATBELT.median())

sen.boxplot(dt.CLMAGE)
dt["CLMAGE"]=pd.DataFrame(median_imputation.fit_transform(dt[["CLMAGE"]]))
dt.CLMAGE=dt.CLMAGE.fillna(dt.CLMAGE.median())

dt.isna().sum()


###########################################################################################################
data = pd.read_csv("D:/assignments of data science/DataSets/OnlineRetail.csv",encoding='latin1')

data.dtypes

A 
For the given dataset perform the type casting (convert the datatypes, ex. float to int)

data.UnitPrice= data.UnitPrice.astype('int64')
data.dtypes

B
Check for the duplicate values, and handle the duplicate values 

duplicate=data.duplicated()
duplicate
sum(duplicate)

df=data.drop_duplicates()
df

C
Do the data analysis 

plt.hist(data.UnitPrice) # histogram 

plt.boxplot(data.UnitPrice) # boxplot

plt.scatter(data['UnitPrice'],data['CustomerID']) 
#############################################################################################################

DISCRETIZATION

tf=pd.read_csv("D:/assignments of data science/DataSets/iris.csv")
tf.head()

tf.describe()

tf['Sepal']=pd.cut(tf['Sepal.Length'],bins=2, labels=["low","high"])

tf.Sepal.value_counts()

tf['Width']=pd.cut(tf['Sepal.Width'],bins=2, labels=["low","high"])

tf.Width.value_counts()

tf['Petal']=pd.cut(tf['Petal.Length'],bins=2, labels=["low","high"])
tf.Petal.value_counts()

tf['width']=pd.cut(tf['Petal.Width'],bins=2, labels=["low","high"])
tf.width.value_counts()

#############################################################################################################

Dummy Variables

animal=pd.read_csv("D:/assignments of data science/DataSets/animal_category.csv")
animal

animal.columns
animal.dtypes
animal.shape

animal.drop(['Homly'],axis =1 , inplace= True)
animal_new= pd.get_dummies(animal)
animal_new_1 = pd.get_dummies(animal,drop_first = True)
animal_new_1
 
#############################################################################################################

Standardization & Normalization 

Standardization

from sklearn.preprocessing import StandardScaler

seeds = pd.read_csv("D:/assignments of data science/DataSets/Seeds_data.csv")
seeds

f=seeds.describe()
f

data =seeds.describe()
scaler= StandardScaler()
df= scaler.fit_transform(seeds)
dataset=pd.DataFrame(df)
res=dataset.describe()
res

Normalization

ethic = pd.read_csv("D:/assignments of data science/DataSets/Seeds_data.csv")
ethic

z = ethic.describe()
z 
ethic.columns

ethic=pd.get_dummies(ethic,drop_first= True)
ethic
 
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(ethic)
df_norm

a=df_norm.describe()
a

#############################################################################################################

Transformation 

import scipy.stats as stat
import pylab
import pandas as pd
calories = pd.read_csv("D:/assignments of data science/DataSets/calories_consumed.csv")
calories

stat.probplot(calories.Calories_Consumed, dist="norm", plot=pylab)

stat.probplot(calories.Weight_gained,dist="norm",plot=pylab)

stat.probplot(np.log(calories.Weight_gained),dist="norm", plot=pylab)

stat.probplot(np.log(calories.Calories_Consumed),dist="norm",plot=pylab)

#############################################################################################################

String manipulations 

1

A

string="grow gratitude"
print(string[0])
out put= g

B

print(len(string))
out put = 14

C

string="grow gratitude"
substring="g"    
count=string.count(substring)
count
out put = 2

2 

string="Being aware of a single shortcoming within yourself is far more useful than being aware of a thousand in someone else."
string.isalnum()
string.isalpha()
total = 0
for i in range(len(string)):
    total= total+1
print(total)

3

string="Idealistic as it may sound, altruism should be the driving force in business, not just competition and a desire for wealth"
print(len(string))

string[:10]

string[:16]

string[-18:]

4

string = "stay positive and optimistic"

print(string.split())


5

🪐 
print('{}\n{}\n{}'.format("🪐".center(29),(29*"🪐").center(29),(29*"🪐").center(29)))

print((str("🪐")split())*118)

7

string="grow gratitude"
print(string.replace("grow","growth of"))

print(string)  

8

def reverse(string):
    string=string[::-1]
    return string
s = ".elgnujehtotniffo deps mehtfohtoB .eerfnoilehttesotseporeht no dewangdnanar eh ,ylkciuQ .elbuortninoilehtdecitondnatsapdeklawesuomeht ,nooS .repmihwotdetratsdnatuotegotgnilggurts saw noilehT .eert a tsniagapumihdeityehT .mehthtiwnoilehtkootdnatserofehtotniemacsretnuhwef a ,yad enO .ogmihteldnaecnedifnocs’esuomeht ta dehgualnoilehT ”.emevasuoy fi yademosuoyotplehtaergfo eb lliw I ,uoyesimorp I“ .eerfmihtesotnoilehtdetseuqeryletarepsedesuomehtnehwesuomehttaeottuoba saw eH .yrgnaetiuqpuekow eh dna ,peels s’noilehtdebrutsidsihT .nufroftsujydobsihnwoddnapugninnurdetratsesuom a nehwelgnujehtnignipeelsecno saw noil A"

print(reverse(s))

output
A lion was oncesleepinginthejunglewhen a mousestartedrunningupanddownhisbodyjustforfun. Thisdisturbedthelion’s sleep, and he wokeupquiteangry. He was abouttoeatthemousewhenthemousedesperatelyrequestedtheliontosethimfree. “I promiseyou, I will be ofgreathelptoyousomeday if yousaveme.” Thelionlaughed at themouse’sconfidenceandlethimgo. One day, a fewhunterscameintotheforestandtookthelionwiththem. Theytiedhimupagainst a tree. Thelion was strugglingtogetoutandstartedtowhimper. Soon, themousewalkedpastandnoticedthelionintrouble. Quickly, he ranandgnawed on theropestosetthelionfree. Bothofthem sped offintothejungle.

#############################################################################################################

INFERENTIAL STATISTICS

1
Three Coins are tossed, find the probability that two heads and one tail are obtained?
Ans= 3/8

2
Two Dice are rolled, find the probability that sum is

a
Equal to 1
Ans=0

b
less then or equal to 4
Ans=1/6

c
Sum is divisible by 2 and 3
Ans=1/6

3
A bag contains 2 red, 3 green and 2 blue balls. Two balls are drawn at random. What is the probability that none of the balls drawn is blue?
Ans=10/12

4
Calculate the Expected number of candies for a randomly selected child:
    
a
Child A – probability of having 1 candy is 0.015
Ans= 3.09

b
Child B – probability of having 4 candies is 0.2
Ans = 3.09

5
Calculate Mean, Median, Mode, Variance, Standard Deviation, Range & comment about the values / draw inferences, for the given dataset

import pandas as pd
import statistics as st


dataset = pd.read_excel("D:/assignments of data science/DataSets/Assignment_module.xlsx")

MEAN

a= st.mean(dataset.Points)
Ans=3.596

b= st.mean(dataset.Score)
Ans= 3.212

c=st.mean(dataset.Weigh)
Ans=17.848

MEDIAN

d=st.median(dataset.Points)
Ans=3.695

e=st.median(dataset.Score)
Ans=3.325

f=st.median(dataset.Weigh)
Ans=17.71

MODE

print(st.mode(dataset.Points))
Ans=3.92

print(st.mode(dataset.Score))
Ans=3.44

print(st.mode(dataset.Weigh))
Ans=17.02

VARIANCE

print(st.variance(dataset.Points))
Ans=0.285

print(st.variance(dataset.Score))
Ans=0.932

print(st.variance(dataset.Weigh))
Ans=3.193

STANDARD DEVIATION

print(st.stdev(dataset.Points))
Ans=0.534

print(st.stdev(dataset.Score))
Ans=0.965

print(st.stdev(dataset.Weigh))
Ans=1.787

RANGE

min =min(dataset.Points)
max =max(dataset.Points)
Ans= 2.76 , 4.93

min1 =min(dataset.Score)
max1 =max(dataset.Score)
Ans= 1.513 , 5.424

min2=min(dataset.Weigh)
min2= min(dataset.Weigh)
Ans=14.5 , 22.9

6

Calculate Expected Value for the problem below

a)	The weights (X) of patients at a clinic (in pounds), are
    
a=(108, 110, 123, 134, 135, 145, 167, 187, 199)

print(st.mean(a))

Ans=145.333

7
1 What is the most likely monetary outcome of the business venture?
ANS:- Max. P = 0.3 for P(2000). So most likely outcome is 2000

2 Is the venture likely to be successful? Explain
ANS:- P(x>0) = 0.6, implies there is a 60% chance that the venture would yield profits or greater 
than expected returns. P(Incurring losses) is only 0.2. So the venture is likely to be successful

3 What is the long-term average earning of business ventures of this kind? Explain
ANS:- Weighted average = x*P(x) = 800. This means the average expected earnings over a long 
period of time would be 800(including all losses and gains over the period of time)


 