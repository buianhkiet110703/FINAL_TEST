# FINAL_TEST

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

  ## B. Visualization
  Data visualization by scatter chart to see the correlation of the profit column with other columns
```php
fig, ((ax1,ax2),(ax3,ax4))= plt.subplots(nrows=2, ncols=2,figsize=(20,10))
ax1.scatter(df['Profit'], df['Sales'], color='gray')
ax1.set(title='Biểu đồ tương quan giữa Profit và Sales', xlabel='Profit', ylabel='Sales');
ax2.scatter(df['Profit'], df['Quantity'], color='black')
ax2.set(title='Biểu đồ tương quan giữa Profit và Quantity', xlabel='Profit', ylabel='Quantity');
ax3.scatter(df['Profit'], df['Discount'], color='green')
ax3.set(title='Biểu đồ tương quan giữa Profit và Discount', xlabel='Profit', ylabel='Discount');
ax4.scatter(df['Profit'], df['order-ship'], color='brown')
ax4.set(title='Biểu đồ tương quan giữa Profit và order-ship', xlabel='Profit', ylabel='order-ship');
```
Output:

![image](https://user-images.githubusercontent.com/110837675/201525599-7c994e1e-6686-4597-aa61-c04722162284.png)

 
 
 
  
