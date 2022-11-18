# REGRESSION

## I. Introduce Dataset

  This is company's dataset in USA showing revenue and profit of this company from 2014 to 2018. The dataset has 9994 rows and includes 21 columns such as:
    
    -Order ID: unique ID of the order
    
    -Order Date: 
    
    -Ship Date: delivery standard
    
    -Customer ID: ID of custonmer
    
    -Customer Name: Name of customer
    
    -Segment: classify customer
    
    -Country: only USA
    
    -City
    
    -State
    
    -Postal Code
    
    -Region
    
    -Product ID
    
    -Category
    
    -Sub-Category
    
    -Product Name
    
    -Sales: revenue of company
    
    -Quantity
    
    -Discount
    
    -Profit
    
 With this dataset, I will predict columns profit of company.
    
## II. Data Processing And Visualization
   #### A. Data Processing
   
  Check columns to see if any have NAN values?
```php
df.isnull().sum()
```
 Output:
 
 ![](https://scontent.fsgn15-1.fna.fbcdn.net/v/t1.15752-9/308537140_658095595938222_5151017212716355642_n.png?_nc_cat=111&ccb=1-7&_nc_sid=ae9488&_nc_ohc=n9tRqz5qq_gAX8H6bDy&_nc_ht=scontent.fsgn15-1.fna&oh=03_AdTGwr4hpyPX0cb_mE4RDjp-Kqzh3C7LTuK7WytlqYAfJw&oe=6398473A)
 
 This dataset does not have NAN values
 
This data has two time columns which are Order Date and Ship Date. Now I will get the number of days from Order Date to Ship Date to see how long it took to prepare the orders to ship.
```php
df['order-ship'] = (pd.to_datetime(df['Ship Date']) - pd.to_datetime(df['Order Date'])).dt.days
```
Now, the dataset will appear order-ship columns showing the number of days from Order Date to Ship Date.
![](https://scontent.fsgn15-1.fna.fbcdn.net/v/t1.15752-9/308498394_1170442966886788_8807345468326570683_n.png?_nc_cat=100&ccb=1-7&_nc_sid=ae9488&_nc_ohc=u7wWYmUOLmwAX85zybL&tn=-Fc4noKWOTfEC8FP&_nc_ht=scontent.fsgn15-1.fna&oh=03_AdSjjqxhuQrmtd7KZZhgl8ezzlWtDIVWhV5D-M3gjcUpyw&oe=639880AB)

Look at the standard deviation and the mean of the dataset
```php
df.describe()
```
![image](https://user-images.githubusercontent.com/110837675/201741386-ef3477d5-01b6-4d34-99b2-2d745c507139.png)


  #### B. Visualization
  Data visualization by scatter chart to see the correlation of the profit column with other columns
```php
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
sns.set(rc={'axes.facecolor':'lightblue','figure.facecolor':'lightgreen'})
sns.scatterplot(ax=axes[0, 0], data=df, x='Sales', y='Profit')
axes[0,0].set_title('correlation between variable Profit and Sales')
sns.scatterplot(ax=axes[0, 1], data=df, x='Quantity', y='Profit')
axes[0,1].set_title('correlation between variable Profit and Quantity')
sns.scatterplot(ax=axes[1, 0], data=df, x='Discount', y='Profit')
axes[1,0].set_title('correlation between variable Profit and Discount')
sns.scatterplot(ax=axes[1, 1], data=df, x='order-ship', y='Profit')
axes[1,0].set_title('correlation between variable Profit and order-ship');
```
Output:

![image](https://user-images.githubusercontent.com/110837675/202071698-b4d4df7f-5294-4586-a93a-991b91e7ab0d.png)


```php
plt.figure(figsize=(12,6))
plt.hist(df['Profit'], bins=20, rwidth= 0.8, density=True)
plt.xlabel('Profit')
plt.ylabel('count')

rng= np.arange(df['Profit'].min(), df['Profit'].max(),0.1)
plt.plot(rng, norm.pdf(rng, df['Profit'].mean(), df['Profit'].std()));
plt.grid(True)
```
Output:

![image](https://user-images.githubusercontent.com/110837675/201525782-b0cce9a2-a406-4808-a9f9-c20a58e548a1.png)

Visualization by heatmap chart to see correlation of variables.
```php
plt.figure(figsize=(20,10))
corr1= df.corr()
sns.heatmap(corr1, square= True, annot= True, fmt= '.2f', annot_kws= {'size':16}, cmap='viridis');
```
Output:

![image](https://user-images.githubusercontent.com/110837675/202073955-adfaed70-c5f6-4151-8e42-7a74614cd8a1.png)

Visualize to see the concentration level of variables Sales and Profit compare with variable Segment.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Sales', y ='Profit',
                   size='Sales'  , color = 'Segment')
fig.show();
```
![image](https://user-images.githubusercontent.com/110837675/202698545-d276b3aa-e691-45ec-ba73-d2a60899b728.png)

```php
plt.figure(figsize=(12,6))
sns.barplot(data=df, x='Segment',y='Sales',hue='Category');
```
![image](https://user-images.githubusercontent.com/110837675/202243824-11e0fa02-6118-4913-8a53-bb836fde5a78.png)


  ### III. Data processing with Scikit-learn
Remove columns such as: Row ID, Order ID, Order Date, Ship Date, Customer ID, Product ID , Customer Name , City, Postal Code, Country, Product Name, Sub-Category, State. These are columns unnecessary for training model.
```php
data= df.drop(['Row ID','Order ID','Order Date','Ship Date','Customer ID','Product ID','Customer Name','City', 'Postal Code','Country','Product Name','Sub-Category','State'], axis= 'columns')
```
![image](https://user-images.githubusercontent.com/110837675/201527068-4ffea0be-cbb1-49cc-b762-291a0095f325.png)

Next, Select features for X and Y. Features for X are independent variables. Feature for Y is predict variable (dependent). Then predict Y based on X. In this dataset, X keeps all the columns but removing profit column. Y is the Profit column.

```php
df1= data.drop(['Profit'], axis='columns')
x= df1.values
y= data['Profit']
```
To make sure that the model have a good result, we must scaler data about the same range of values 0 and 1. I will use MinmaxScaler for data X from fourth column to the final column (from Sales column to order-ship column) to return the same range of values 0 and 1.

```php
from sklearn.preprocessing import MinMaxScaler
mn= MinMaxScaler(feature_range=(0,1))
x[:, 4:]= mn.fit_transform(x[:,4:])
```
Then, for columns which have words, we have to encode them as numbers because the computer only understands numbers and doesn't understander words. I will use method OnehotEncoder to encode columns having words (Ship Mode, Segment, Region, Category) to return only has number 0 and 1.
```php
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
ohe= make_column_transformer((OneHotEncoder(),[0,1,2,3]), remainder= 'passthrough')
x= ohe.fit_transform(x)
```
Next, I will split X and Y into 2 sets of train and test( X_train,  X_test,  y_train,  y_test). 80% data is used for training model and 20% data is used for testing to evaluate the result of model.
```php
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)
```
# III. Build And Evaluate Model.
   I will call the models out and transfer the training set into models so that the models can learns this traning set. Then I will evaluate models on testing set based on criteria 'Mean_square_error, mean_squared_error and score'. For Mean_square_error, mean_squared_error, the smaller the index is, the better the models work. Criteria 'score' ,the closer the index to 1, the better the models work.

```php
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import  SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS, mean_absolute_error, mean_squared_error
models= [LinearRegression(),RandomForestRegressor(n_estimators=5),GradientBoostingRegressor(learning_rate=0.03, random_state=0),
         AdaBoostRegressor(learning_rate=0.05),
         DecisionTreeRegressor(random_state=0, max_depth=3),
         SVR(kernel='linear', gamma='scale',degree=4),KNeighborsRegressor(n_neighbors=2), 
         xgb.XGBRegressor(learning_rate=0.01), lgb.LGBMRegressor(learning_rate=0.05)];
CV = 10 
entries = []
i=0
for model in models:
    mae_l = []
    mse_l = []
    score= []
    for j in range(CV):
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        score_model= model.score(X_test,y_test) 
        score.append(score_model)
        mae_l.append(mae)
        mse_l.append(mse)
    entries.append([model_name, np.array(mae_l).mean(),np.array(mse_l).mean(),np.array(score).mean()])
    i += 1
model_df = pd.DataFrame(entries, columns=['model_name', 'Mean MAE','Mean MSE','score'])
model_df.sort_values(by=['score'], ascending=False)
```
![image](https://user-images.githubusercontent.com/110837675/201585214-b08f4442-42fa-407c-beab-0222e3ef88e4.png)

Look at the result, we can see that all the models have not returned good result yet. To make the models work even better, the parameters passed to the models must be reasonable. In order to do that, I will use GridSearchCV to find the best parameters for each model.

```php
#GridSearchCV for RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
parameter_RandomForestRegressor= {
    'n_estimators': range(1,20)
    }

grid_RandomForestRegressor= GridSearchCV(RandomForestRegressor(random_state=0), parameter_RandomForestRegressor, cv=5,verbose=2, refit=True )
grid_RandomForestRegressor.fit(X_train, y_train)
print('Best parameter_RandomForestRegressor: ',grid_RandomForestRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201585877-8e386562-50ae-4fb1-879b-95b5a300456e.png)

Now, I will see the best parameter for RandomForestRegressor model. Next, let try with the rest of the models.

```php
# GridSearchCV for DecisionTreeRegressor model
from sklearn.tree import DecisionTreeRegressor
parameter_DecisionTreeRegressor= {
    'max_depth': range(1,50)
}
grid_DecisionTreeRegressor= GridSearchCV(DecisionTreeRegressor(), parameter_DecisionTreeRegressor, cv=5,verbose=2, refit=True )
grid_DecisionTreeRegressor.fit(X_train, y_train)
print('Best parameter_DecisionTreeRegressor: ',grid_DecisionTreeRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201587113-fcfdc3dc-6b35-433c-b90c-0fa1f955720a.png)

```php
# GridSearchCV for KNeighborsRegressor model
from sklearn.neighbors import KNeighborsRegressor
parameter_KNeighborsRegressor= {
    'n_neighbors': range(1,50)
}
grid_NeighborsRegressor= GridSearchCV(KNeighborsRegressor(), parameter_KNeighborsRegressor, cv=5,verbose=2, refit=True )
grid_NeighborsRegressor.fit(X_train, y_train)
print('Best parameter_KNeighborsRegressor: ', grid_NeighborsRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201575214-cb263eac-4547-44a4-a51f-fedf1280bdfe.png)

```php
# GridSearchCV for GradientBoostingRegressor model
from sklearn.ensemble import GradientBoostingRegressor
parameter_GradientBoostingRegressor= {
    'learning_rate': [0.03,0.05],
    'max_depth' : [5,10,12,15,20],
    'n_estimators' :[10,15,20,100,1000]
}
grid_GradientBoostingRegressor= GridSearchCV(GradientBoostingRegressor(), parameter_GradientBoostingRegressor, cv=5,verbose=2, refit=True )
grid_GradientBoostingRegressor.fit(X_train, y_train)
print('Best parameter_GradientBoostingRegressor: ', grid_GradientBoostingRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201575266-515fe2f7-aece-4583-9ee4-2983f4bdb298.png)

```php
# GridSearchCV for XGBRegressor model
import xgboost as xgb
parameter_XGBRegressor= {
    'max_depth': [5,7,10,20],
    'n_estimators' :[20,100,1000],
    'subsample': [0.5,0.7,0.8]
}
grid_XGBRegressor= GridSearchCV(xgb.XGBRegressor(), parameter_XGBRegressor, cv=5,verbose=2, refit=True )
grid_XGBRegressor.fit(X_train, y_train)
print('Best parameter_XGBRegressor: ', grid_XGBRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201575346-42250817-71a2-45a9-b318-87ca6b64fe7c.png)

```php
# GridSearchCV for SVR model
from sklearn.svm import SVR
parameter_SVR= {
    'kernel': [ 'poly','linear'],
    'degree': [3,4],
    'gamma': ['scale','auto']
    }
grid_SVR= GridSearchCV(SVR(), parameter_SVR, cv=5,verbose=2, refit=True )
grid_SVR.fit(X_train, y_train)
print('Best parameter_SVR: ', grid_SVR.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201575395-1b69cfac-c2a8-462c-b353-080389e3555e.png)

```php
# GridSearchCV for LGBMRegressor model
import lightgbm as lgb
parameter_LGBMRegressor= {
    'max_depth': [ 5,10,20],
    'n_estimators': [100,1000,2000],
    }
grid_LGBMRegressor= GridSearchCV(lgb.LGBMRegressor(), parameter_LGBMRegressor, cv=5,verbose=2, refit=True )
grid_LGBMRegressor.fit(X_train, y_train)
print('Best parameter_LGBMRegressor: ', grid_LGBMRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201575461-e611e72e-363e-4e6d-9b3e-c899af9fd6c1.png)

```php
# GridSearchCV for AdaBoostRegressor model
from sklearn.ensemble import AdaBoostRegressor
parameter_AdaBoostRegressor= {
    'learning_rate':[0.05,0.08],
    'loss': ['linear', 'square','exponential'],
    'n_estimators': [50,100,1000],
    'base_estimator': [DecisionTreeRegressor(max_depth=5),DecisionTreeRegressor(max_depth=10),DecisionTreeRegressor(max_depth=20)]
    }
grid_AdaBoostRegressor= GridSearchCV(AdaBoostRegressor(), parameter_AdaBoostRegressor, cv=5,verbose=2, refit=True )
grid_AdaBoostRegressor.fit(X_train, y_train)
print('Best parameter_AdaBoostRegressor: ', grid_AdaBoostRegressor.best_params_ )
```
![image](https://user-images.githubusercontent.com/110837675/201580965-47e129ca-1b83-408d-940e-92eb6e442d8b.png)

After I found best parameters for each model, I will replace them into the models again, to see the best result of models.

```php
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import  SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb                                             
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import SCORERS, mean_absolute_error, mean_squared_error
models= [LinearRegression(),RandomForestRegressor(n_estimators=12, random_state=0),GradientBoostingRegressor(learning_rate=0.05, max_depth=5,n_estimators=1000),
         DecisionTreeRegressor(random_state=0, max_depth=7),
         SVR(kernel='linear', gamma='scale',degree=4),KNeighborsRegressor(n_neighbors=7), 
         xgb.XGBRegressor(learning_rate=0.08,max_depth=5,n_estimators=100, subsample=0.8), lgb.LGBMRegressor(learning_rate=0.05,max_depth=5,n_estimators=100),
         AdaBoostRegressor(learning_rate=0.05, base_estimator= DecisionTreeRegressor(max_depth=5),loss='linear', n_estimators=1000)] ;
CV = 10 
entries = []
i=0                                             
for model in models:
    mae_l = []
    mse_l = []
    score= []
    for j in range(CV):
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        score_model= model.score(X_test,y_test) 
        score.append(score_model)
        mae_l.append(mae)
        mse_l.append(mse)
    entries.append([model_name, np.array(mae_l).mean(),np.array(mse_l).mean(),np.array(score).mean()])
    i += 1
model_df = pd.DataFrame(entries, columns=['model_name', 'Mean MAE','Mean MSE','score'])
model_df.sort_values(by=['score'], ascending=False)
```
![image](https://user-images.githubusercontent.com/110837675/201587937-7695de76-4ae1-4f3b-b426-fe1317f3c118.png)

Now, Compare result before and after passed parameters into the model.

![image](https://user-images.githubusercontent.com/110837675/201597803-e4290ae9-f5ce-4d28-ac89-32df3cb3530a.png)

Try with PolynomialFeatures model.
```php
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg= PolynomialFeatures(degree=3)
x_poly= poly_reg.fit_transform(X_train)
poly_reg.fit(x_poly,y_train)
lin_reg= LinearRegression()
lin_reg.fit(x_poly,y_train)
x_test_poly= poly_reg.fit_transform(X_test)
lin_score= lin_reg.score(x_test_poly, y_test)
y_pred_poly= lin_reg.predict(x_test_poly)
lin_mse= mean_squared_error(y_test,y_pred_poly)
lin_mae= mean_absolute_error(y_test,y_pred_poly)
print('MSE_POLY: ',lin_mse)
print('MAE_POLY: ',lin_mae)
print('SCORE POLY: ',lin_score)
```
![image](https://user-images.githubusercontent.com/110837675/201605769-c9b3a9e2-184d-4d24-a118-248e03afb214.png)

Now, I will try with BaggingRegressor.

```php
from sklearn.ensemble import BaggingRegressor
bg= BaggingRegressor(n_estimators=1000, max_samples=7995,random_state=0)
bg.fit(X_train,y_train)
score= bg.score(X_test,y_test)
y_pred= bg.predict(X_test)
mse= mean_squared_error(y_test,y_pred)
mae= mean_absolute_error(y_test, y_pred)
print('MSE: ', mse)
print('MAE: ', mae)
print('SCORRE: ', score)
```
![image](https://user-images.githubusercontent.com/110837675/201606048-e6810f94-7168-4359-a725-b90c5a6c90ec.png)

- Bagging with LinearRegression

```php
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor_LinearRegression = BaggingRegressor(LinearRegression(), random_state=0)
model_BaggingRegressor_LinearRegression.fit(X_train, y_train)
score = model.score(X_test,y_test)
y_pred= model_BaggingRegressor_LinearRegression.predict(X_test)
mse= mean_squared_error(y_test,y_pred)
mae= mean_absolute_error(y_test, y_pred)
print('MSE: ', mse)
print('MAE: ', mae)
print('SCORRE: ', score)
```
![image](https://user-images.githubusercontent.com/110837675/201606369-f194fe02-136d-496d-8bde-4743f3c25caa.png)

- Bagging with RandomForestRegressor

```php
model = BaggingRegressor(RandomForestRegressor(n_estimators=12), random_state=0)
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
y_pred= model.predict(X_test)
mse= mean_squared_error(y_test,y_pred)
mae= mean_absolute_error(y_test, y_pred)
print('MSE: ', mse)
print('MAE: ', mae)
print('SCORRE: ', score)
```
![image](https://user-images.githubusercontent.com/110837675/201606633-961d063e-0756-4367-b8cd-918bf62043fe.png)

- Bagging with DecisionTreeRegressor

```php
model = BaggingRegressor(DecisionTreeRegressor(max_depth=20), random_state=0)
model.fit(X_train,y_train)
score = model.score(X_test,y_test)
y_pred= model.predict(X_test)
mse= mean_squared_error(y_test,y_pred)
mae= mean_absolute_error(y_test, y_pred)
print('MSE: ', mse)
print('MAE: ', mae)
print('SCORRE: ', score)
```
![image](https://user-images.githubusercontent.com/110837675/201606793-d1266f09-be2f-413d-a768-1cf995149967.png)

I tried most of regression model. Now, I will synthetic result include: Mean MAE, Mean MSE and score into dataframe to evaluate models.

```php
cv_df= {
 'model_name':['PolynomialFeatures(degree=3)','BaggingRegressor','BaggingRegressor(LinearRegression)',
 'BaggingRegressor(RandomForestRegressor)','BaggingRegressor(DecisionTreeRegressor)'],
 'Mean MAE':[39.707817458729366, 28.954374305427592, 61.37902788738119, 29.84634918392471,29.43079687400924],   
 'Mean MSE':[73621.49870775724, 21799.12495604431, 55438.6410992574, 27310.201803991837, 15818.370687695025],
 'score':[0.15453006224360089, 0.7496586575493227,0.014684260264121507,0.6863694026252856,0.8183416920942959]
}
df2= pd.DataFrame(cv_df)
total= pd.concat([model_df, df2],ignore_index=True)
total.sort_values(by=['score'], ascending=False)
```
![image](https://user-images.githubusercontent.com/110837675/201607503-75dba7d6-e9b4-4778-bb01-8dc7ff69a803.png)

I will evaluate model base on criterion 'score'. The closer the score criterion is to 1, the better the model works. So, the models that work the best are DecisionTreeRegressor, BaggingRegressor(DecisionTreeRegressor), RandomForestRegressor, GradientBoostingRegressor.


# CLUSTERING
## I. Introduce Dataset

Dataset show average life expectancy of countries and relate variables(2020):

![image](https://user-images.githubusercontent.com/110837675/202417099-cd649697-1fdb-431f-85fe-ca947791a05b.png)

With this dataset, I will Subgroups of countries by GDP per capital.

## II. Data Processing And Visualization
   #### A. Data Processing
   
  Check columns to see if any have NAN values?
```php
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/110837675/202418356-61a2f6ee-b08a-4b4c-b2a8-f190e079981d.png)

Data doesn't have NAN values.

  #### B. Visualization
  Data visualization by scatter chart to see the correlation of the Life Expectancy (Year) column with other columns.
  ```php
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
sns.scatterplot(ax=axes[0, 0], data=df, x='CO2 emissions (Billion tons)', y='Life Expectancy (Year)', s=200, alpha=0.7,color='red')
axes[0,0].grid(True)
sns.scatterplot(ax=axes[0, 1], data=df, x='GDP per capita ($)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='green')
axes[0,1].grid(True)
sns.scatterplot(ax=axes[0, 2], data=df, x='Rate of using basic drinking water (%)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='black')
axes[0,2].grid(True)
sns.scatterplot(ax=axes[1, 0], data=df, x='Obesity among adults (%)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='pink')
axes[1,0].grid(True)
sns.scatterplot(ax=axes[1, 1], data=df, x='Beer consumption per capita (Liter)', y='Life Expectancy (Year)', s=200, alpha=0.7, color='brown');
axes[1,1].grid(True)
```
![image](https://user-images.githubusercontent.com/110837675/202599885-02d4a5f2-bd12-4660-b409-b985656d59d3.png)

```php
plt.figure(figsize=(20,10))
plt.title('CORRELATION ')
corr1= df.corr()
sns.heatmap(corr1, square= True, annot= True, fmt= '.2f', annot_kws= {'size':16}, cmap='Blues', linecolor='white', linewidths=0.5);
```
![image](https://user-images.githubusercontent.com/110837675/202419617-d2e6541c-452e-4abf-9d05-edb7a0e2ec02.png)

### III. Data Processing

I will drop Country columns. Because this columns contains word.

```php
data= df.drop(['Country'], axis='columns')
df1= data.values
```
Next, to make sure that the model have a good result, we must scaler data about the same range of values 0 and 1. I will use MinmaxScaler for data.
```php
from sklearn.preprocessing import StandardScaler
st= StandardScaler()
x= st.fit_transform(df1)
data_scaled= pd.DataFrame(x, columns=data.columns)
data_scaled
```
I will find the number of best match groups using elbow method and Silhouette_score.
 - ELBOW
 ```php
 k_war= {
'init':'random',
'n_init': 10,
'max_iter': 300,
'random_state': 42
}

sse= []
for i in range(1,11):
    kmeans= KMeans(n_clusters=i, **k_war)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE");
```
![image](https://user-images.githubusercontent.com/110837675/202424684-b82b278c-51ad-499f-9e23-ee4431dbb0f3.png)

The slope of the downward curve is very uniform, making it very difficult to choose the number of groups. I will draw Silhouette_score method.
- Silhouette_score.
```php
silhouette= []
for i in range(2,11):
    kmeans= KMeans(n_clusters=i, **k_war)
    kmeans.fit(data_scaled)
    silhouette_sc= silhouette_score(data_scaled, kmeans.labels_)
    silhouette.append(silhouette_sc)

plt.figure(figsize=(12,6))
plt.style.use('fivethirtyeight')
plt.plot(range(2,11), silhouette)
plt.xticks(range(2,11))
plt.xlabel("Number of Clusters")
plt.ylabel("silhouette_score");
```
![image](https://user-images.githubusercontent.com/110837675/202426019-43c3fdad-b33e-4762-8165-aaecb2d19318.png)

Theoretically, K=2 should be chosen because it has the highest value. But from k=2 to k=3, the slope is very large, so I choose k=3 and compare the elbow, k=3 is also in the downward slope of the elbow.

Besides, I wil use KneeLocator to determine elbow.

![image](https://user-images.githubusercontent.com/110837675/202427410-75094ca4-4caa-432c-ad19-cdde83aba1ec.png)

The results is k=3.

### IV. Build Model.

I will use K-MEANS, Hierarchical Clustering, Probabilistic Clustering to compare with each other.

  #### A. K-MEANS.
  
  ```php
  kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(data_scaled)
  
plt.figure(figsize=(12,6))
plt.scatter(data_scaled[y_kmeans==0,0], data_scaled[y_kmeans==0,1], s=100, c='yellow', label='cluster 1')
plt.scatter(data_scaled[y_kmeans==1,0], data_scaled[y_kmeans==1,1], s=100, c='green', label='cluster2')
plt.scatter(data_scaled[y_kmeans==2,0], data_scaled[y_kmeans==2,1], s=100, c='purple', label='cluster3')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,2], s=300, c='black', label='centroid')
plt.title('Clusters Country (K-MEANS)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202428409-7f4614d6-2d63-4130-afe9-7a9f34705492.png)

This is k=3, now I draw k=4 to compare with k=3.

```php
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data_scaled)
  
plt.figure(figsize=(12,6))
plt.scatter(data_scaled[y_kmeans==0,0], data_scaled[y_kmeans==0,1], s=100, c='yellow', label='cluster 1')
plt.scatter(data_scaled[y_kmeans==1,0], data_scaled[y_kmeans==1,1], s=100, c='green', label='cluster2')
plt.scatter(data_scaled[y_kmeans==2,0], data_scaled[y_kmeans==2,1], s=100, c='purple', label='cluster3')
plt.scatter(data_scaled[y_kmeans==3,0], data_scaled[y_kmeans==3,1], s=100, c='blue', label='cluster4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,2], s=300, c='black', label='centroid')
plt.title('Clusters Country (K-MEANS)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202428919-8c3b0daf-8d6c-4107-87be-a8b606a24f1a.png)

Comparing 2 drawings, k=3 is more reasonable.

I will add the clustering column to the data and plot them on the map.
```php
df['class']= kmeans.labels_

import plotly.express as px
fig = px.choropleth(df, locations="Country", locationmode='country names', color='class', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(K-MEANS)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202429948-77531a26-0d0b-40c3-b2e1-b6172c0d252b.png)

  #### B.Hierarchical Clustering.
  
  I will plot dendrogram to determine cluster.
  
  ```php
  plt.figure(figsize = (20,10))
mergings = sch.linkage(data_scaled, method="complete", metric='euclidean')
sch.dendrogram(mergings)
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202430470-ff68c8cc-7515-4419-9c0b-9733d137cbb1.png)

The dendrogram is divided into 3 clusters.

```
agg= AgglomerativeClustering(n_clusters=3, linkage='ward',affinity='euclidean')
agg.fit_predict(data_scaled)

plt.figure(figsize=(12,6))
plt.scatter(data_scaled['Life Expectancy (Year)'], data_scaled['GDP per capita ($)'], c=agg.labels_, s=200);
plt.title('Clusters Country (Hierarchical Clustering)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202430948-6ea9199e-e39b-4fe3-8ed3-f5aee6844aac.png)

```php
df['class']= agg.labels_

import plotly.express as px
fig = px.choropleth(df, locations="Country", locationmode='country names', color='class', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(Hierarchical Clustering)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202431141-ddeeae38-d48e-4e9a-aec6-244516470d67.png)

  #### C.Probabilistic Clustering
```php
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(data_scaled)

#predictions from gmm
labels = gmm.predict(data_scaled)
df['cluster'] = labels

plt.figure(figsize=(12,6))
plt.scatter(data_scaled['Life Expectancy (Year)'], data_scaled['GDP per capita ($)'], c=df.cluster, s=200);
plt.title('Clusters Country (Probabilistic Clustering)')
plt.xlabel('Life Expectancy (Year)')
plt.ylabel('GDP per capita ($)')
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/202431471-d73f56c6-41e8-4d55-ae3b-99138d4f5a23.png)

```php
import plotly.express as px
fig = px.choropleth(df, locations="Country", locationmode='country names', color='cluster', hover_name="Country", color_continuous_scale="tealrose")
fig.update_layout(title_text = 'Life Expectancy by Country(Probabilistic Clustering)', title_x = 0.5)
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202431596-53e8f5c4-9fa5-4a56-89ee-d3c9be57c033.png)

### V. Compare model.

![image](https://user-images.githubusercontent.com/110837675/202431821-225c239f-1a7d-4ffa-b2df-2ceb20d68206.png)

Grouping with K-means is the most reasonable.

![image](https://user-images.githubusercontent.com/110837675/202432313-372981fa-bebb-4d02-ae3f-473a24394dbf.png)

### VI. Analysis after grouped Country follow GDP.
```php
plt.figure(figsize=(12,6))
plt.grid(True)
sns.violinplot(data=df, x='class', y='Life Expectancy (Year)');
```
![image](https://user-images.githubusercontent.com/110837675/202433370-8724c666-20a5-4eb9-8fa4-9cfe55272e21.png)

Group 2 had the highest life expectancy at about 83 years. Group 1 accounts for a large amount but the average life expectancy is very low.

```php
import plotly.express as px
fig = px.scatter(df, x = 'CO2 emissions (Billion tons)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)', color = 'class', title='Correlations between CO2 emissions and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202435585-fbbc764a-8522-4892-809c-7e261bff89af.png)

Group 2 high-income countries (yellow) have a long life expectancy and high CO2 emissions. Poor countries have a lower life expectancy than rich countries, and emit less CO2.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Rate of using basic drinking water (%)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)' , color = 'class',title='Correlations between Rate of using basic drinking water (%) and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202436088-71800f4e-8165-496b-aead-67f564b2c2b7.png)

High-income countries have very high access to clean drinking water and a high life expectancy. Poor countries suffer from very low access to safe drinking water and lower life expectancy than rich countries.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Obesity among adults (%)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)' , color = 'class',title='Correlations between Obesity among adults (%) and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202436667-11336f8c-eb9b-4998-894b-c82a25e62115.png)

High and relative income countries (yellow and blue) have very high rates of adult obesity. Because high-income countries have a lot of excess food, the rate of obesity is very high. Compared to low-income countries, obesity rates are very low. Because low-income countries do not have much food and lack of food, obesity rates are lower, but in terms of life expectancy, Poor countries still have a lower average age than high-income countries.

```php
import plotly.express as px
fig = px.scatter(df, x = 'Beer consumption per capita (Liter)', y ='Life Expectancy (Year)',
                    size ='GDP per capita ($)' , color = 'class',title='Correlations between Beer consumption per capita (Liter) and life expectancy across different income groups')
fig.show()
```
![image](https://user-images.githubusercontent.com/110837675/202436905-8425d0c8-c3d5-4566-9ecb-9d2c32b39e49.png)


In terms of beer consumption per capita, Group 2 and Group 0 countries (yellow and blue) have a very high beer consumption rate. As for the low-income Group 1 (purple) countries, the amount of beer consumed per capita is lower than in the high-income countries.


Rich countries should share the economy and poorer countries, so that CO2 emissions into the environment are reduced. And poor countries, if they have economic conditions, their quality of life will be higher, the average life expectancy will be higher.





  




 
  
