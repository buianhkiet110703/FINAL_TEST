## REGRESSION

# I. Introduce Dataset

  This is company's dataset in USA show revenue and profit of company from 2014 to 2018. The dataset has 9994 rows and includes 21 columns such as:
    
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
    
 With this dataset will predict columns profit of company
    
# II. Data Processing And Visualization
   ## A. Data Processing
   
  Check columns to see if any have NAN values?
```php
df.isnull().sum()
```
 Output:
 
 ![](https://scontent.fsgn15-1.fna.fbcdn.net/v/t1.15752-9/308537140_658095595938222_5151017212716355642_n.png?_nc_cat=111&ccb=1-7&_nc_sid=ae9488&_nc_ohc=n9tRqz5qq_gAX8H6bDy&_nc_ht=scontent.fsgn15-1.fna&oh=03_AdTGwr4hpyPX0cb_mE4RDjp-Kqzh3C7LTuK7WytlqYAfJw&oe=6398473A)
 
 This dataset hasn't NAN values
 
In data has two columns time is Order Date and Ship Date, now will get number of days from Order Date to Ship Date see how long it take.
```php
df['order-ship'] = (pd.to_datetime(df['Ship Date']) - pd.to_datetime(df['Order Date'])).dt.days
```
Now, in the dataset will appear order-ship columns show number of days from Order Date to Ship Date.
![](https://scontent.fsgn15-1.fna.fbcdn.net/v/t1.15752-9/308498394_1170442966886788_8807345468326570683_n.png?_nc_cat=100&ccb=1-7&_nc_sid=ae9488&_nc_ohc=u7wWYmUOLmwAX85zybL&tn=-Fc4noKWOTfEC8FP&_nc_ht=scontent.fsgn15-1.fna&oh=03_AdSjjqxhuQrmtd7KZZhgl8ezzlWtDIVWhV5D-M3gjcUpyw&oe=639880AB)

Look at the standard deviation and mean of the dataset
```php
df.describe()
```
![image](https://user-images.githubusercontent.com/110837675/201741386-ef3477d5-01b6-4d34-99b2-2d745c507139.png)


  ## B. Visualization
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

![image](https://user-images.githubusercontent.com/110837675/201739895-3196e6f2-4c7a-4453-8efa-6978df5342ed.png)


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

![image](https://user-images.githubusercontent.com/110837675/201526091-106044a7-bfb2-4d96-9628-4672da9f59ab.png)

Visualize to see the concentration level of variables Sales and Profit compare with variable Segment.

```php
plt.figure(figsize=(12,6))
sns.scatterplot(x= 'Sales', y='Profit', hue = 'Segment', data = df );
```
![image](https://user-images.githubusercontent.com/110837675/201817875-f223dd65-82cb-417c-b8be-2cc325cdf7c2.png)


```php
plt.figure(figsize=(12,6))
sns.barplot(x= 'Segment', y='Profit', hue = 'Category', data = df );
```
![image](https://user-images.githubusercontent.com/110837675/201819138-455cac03-d002-4709-ba5e-cd8989d24760.png)


  ## III. Data processing with Scikit-learn
Remove columns such as: Row ID, Order ID, Order Date, Ship Date, Customer ID, Product ID , Customer Name , City, Postal Code, Country, Product Name, Sub-Category, State. These are columns unnecessary for training model.
```php
data= df.drop(['Row ID','Order ID','Order Date','Ship Date','Customer ID','Product ID','Customer Name','City', 'Postal Code','Country','Product Name','Sub-Category','State'], axis= 'columns')
```
![image](https://user-images.githubusercontent.com/110837675/201527068-4ffea0be-cbb1-49cc-b762-291a0095f325.png)

Next, Select feature for X are independent variables, Y is predict variable (dependent), and predict Y base on X. In this dataset, X keep the columns and remove profit column, Y is Profit column.

```php
df1= data.drop(['Profit'], axis='columns')
x= df1.values
y= data['Profit']
```
Let the model have a good result, we must scaler data about the same range of values 0 and 1. I will use MinmaxScaler to  return data X from fourth columnback to the end (from Sales column to order-ship column)  about the same  range of values 0 and 1.

```php
from sklearn.preprocessing import MinMaxScaler
mn= MinMaxScaler(feature_range=(0,1))
x[:, 4:]= mn.fit_transform(x[:,4:])
```
Then, for columns with word, we have to encode them as numbers because computer only understand numbers, itn't understander word. I will use method OnehotEncoder to encode columns have word (Ship Mode, Segment, Region, Category) return number 0 and 1.
```php
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
ohe= make_column_transformer((OneHotEncoder(),[0,1,2,3]), remainder= 'passthrough')
x= ohe.fit_transform(x)
```
Next, I will perform split X and Y into 2 sets of train and test( X_train,  X_test,  y_train,  y_test). 80% data use for training model and 20% data use for testing evaluate result work of model.
```php
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)
```
# III. Build And Evaluate Model.
   I will call the model out and transfer the training set into model to it learn and i will evaluate model  on testing set based on criterias 'Mean_square_error, mean_squared_error and score'. For  Mean_square_error, mean_squared_error, The smaller the index, the better the model works. Criteria 'score' ,the closer the index to 1, the better the model work.

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

Look at the result, we can see  the model has the best work and the model isn't work good. To make the model work even better, the parameters passed to the model must be reasonable. In order to do that, I will use GridSearchCV to find the best parameters for each model.

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

Now, I will see the best parameter for RandomForestRegressor model. Next, try with the rest of the models.

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

After I found best parameters for each model, I will replace it into the model again, to see the best result of model.
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

After I found best parameters for each model, I will replace it into the model again, to see the best result of model.

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

I tried most of regression model. Now, I will synthetic result inclue: Mean MAE, Mean MSE and score into dataframe to evaluate model.

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


## CLUSTERING

 
  
