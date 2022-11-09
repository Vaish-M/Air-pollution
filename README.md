# Air-pollution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv('C:\\Users\\student.IT.000\\Desktop\\IT\DATA\\new.csv')

df1.head()

df1['City'].unique()

df = df1[df1['City']=='Delhi']

df.head()

df.shape

df.info()

def encode()

df1.isnull().sum()


df1['AQI']=df1['AQI'].fillna(df1['AQI'].mean())
df1['PM10']=df1['PM10'].fillna(df1['PM10'].mean())
df1['NO']=df1['NO'].fillna(df1['NO'].mean())
df1['NO2']=df1['NO2'].fillna(df1['NO2'].mean())
df1['NOx']=df1['NOx'].fillna(df1['NOx'].mean())
df1['NH3']=df1['NH3'].fillna(df1['NH3'].mean())
df1['SO2']=df1['SO2'].fillna(df1['SO2'].mean())
df1['O3']=df1['O3'].fillna(df1['O3'].mean())
df1['Benzene']=df1['Benzene'].fillna(df1['Benzene'].mean())
df1['Toluene']=df1['Toluene'].fillna(df1['Toluene'].mean())
df1['Xylene']=df1['Xylene'].fillna(df1['Xylene'].mean())
df1['CO']=df1['CO'].fillna(df1['CO'].mean())

df1.isnull().sum()

sns.heatmap(df.corr())

df.describe()

sns.boxplot(data=df)

drop_outlier = df[(df['AQI']>500) | (df['PM2.5']>180) | (df['NO']>65) |(df['NH3']>50) | (df['NO2']>90) | (df['NOx']>100) | (df['PM10']>450)].index

df = df.drop(drop_outlier)


df.info()

sns.boxplot(data= df)

plt.figure(figsize=(8,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM2.5', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

plt.figure(figsize=(8,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NO2', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

plt.figure(figsize=(8,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NH3', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

plt.figure(figsize=(8,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'NOx', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

sns.heatmap(df.corr())

df1

sns.heatmap(df1.corr())

plt.figure(figsize=(8,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM10', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

drop_outlier1 = df[(df['AQI']<165) & (df['PM10']>200)].index
drop_outlier2 = df[(df['AQI']>200) & (df['PM10']<110)].index
df = df.drop(drop_outlier1)
df = df.drop(drop_outlier2)



plt.figure(figsize=(8,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM10', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

plt.figure(figsize=(12,4),dpi=200)
palette ={'Good': "g", 'Poor': "C0", 'Very Poor': "C1",'Severe': "r","Moderate": 'b',"Satisfactory":'y'}
sns.scatterplot(x= 'AQI', y= 'PM2.5', data=df,hue ='AQI_Bucket',palette = palette, ci= None)

df.info()





sns.boxplot(data=X)

sns.pairplot(data=X)

X.info()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()

scaled_X_train = Scaler.fit_transform(X_train)
scaled_X_test = Scaler.transform(X_test)

from sklearn.svm import SVR

param_grid = {'C':[0.001,0.01,0.1,0.5,1],
             'kernel':['linear','rbf','poly'],
              'gamma':['scale','auto'],
              'degree':[2,3,4],
              'epsilon':[0,0.01,0.1,0.5,1,2]}

from sklearn.model_selection import GridSearchCV

svr = SVR()
grid = GridSearchCV(svr,param_grid=param_grid)

grid.fit(scaled_X_train,y_train)

grid.best_params_

model_predict=grid.predict(scaled_X_test)

model_predict[:10]

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_squared_error(model_predict, y_test)

np.sqrt(mean_squared_error(y_test,model_predict))

x=np.arange(0,len(y_test))
print(x)

fig = plt.figure(figsize =(12,8), dpi=200)

# Add set of axes to figure
axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(x[0:50], y_test[0:50], label='y_test')
axes.plot(x[0:50], model_predict[0:50], label='predict')
plt.legend()
plt.show()









