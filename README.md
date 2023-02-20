# S&P 500 Sectors Vs USD

## Introduction

Introduction 

In this project we wanted to see the correlation between the USD and all the other sectors within S&P 500. There were 11 sectors within S&P 500, but we dropped 2 of the sectors due to insufficient data, ‘XLC’ & ‘XLRE’.

The initial finings showed very strong correlation among the various sectors. This is shown in details in the file S&P_Sectors_Correlation.ipnyp.

We then used Fibonacci and Stochastic models along with Machine Learning Model SVM, in order to see how each of the model performs. For Fibonacci we used 6 different combinations in order to find the combinations which works best in terms of returns.

- 1-day close with 3-Day moving average
- 1-day close with Mid-point values
- 60-minute closing prices
- 60-minute close with 3-Day moving average
- 60-minute close with Mid-point values


## Using Fibonacci Ratios as an Indicator:

## Import Libraries:

Used various libraries to retrieve and analyze financial data from the Alpaca API and Yahoo Finance. It applies technical indicators and calculations to the data and creates new columns in a Pandas DataFrame to represent the results of the analysis. Here's a breakdown:

Imports the following libraries:

-alpaca_trade_api: This is the library used to access the Alpaca API to retrieve financial data.
-pandas: This is a library used for data manipulation and analysis.
-matplotlib.pyplot: This is a library used for data visualization.
-seaborn: This is a library used for data visualization.
-yfinance: This is a library used to access Yahoo Finance to retrieve financial data.
-hvplot.pandas: This is a library used for interactive data visualization.
-Connect to Alpaca API: The code connects to the Alpaca API using the user's API key and secret.

### Set Ticker and Time Frame:

Sets the ticker symbol and time frame for the data to be retrieved.

### Set Start and End Dates:

Sets the start and end dates for the data to be retrieved.

### Retrieve Data: 

Retrieves the financial data from the Alpaca API using the ticker symbol, time frame, start date, and end date.

### Convert Data to Pandas DataFrame:

Converts the retrieved data to a Pandas DataFrame.

## Calculate Technical Indicators: 

Calculated several technical indicators for the data and creates new columns in the DataFrame to represent the results of the calculations. The technical indicators calculated include:

-HH21: This is the highest high of the last 21 days.
-LL21: This is the lowest low of the last 21 days.
-50Pct: This is 50% of the difference between HH21 and LL21.
-61Pct: This is 61.5% of the difference between HH21 and LL21.
-50R: This is the ratio of the close price to 50Pct.
-61R: This is the ratio of the close price to 61Pct.
-Mid: This is the average of the high and low prices.
-3dH: This is the difference between the highest high of the next 3 days and the close price (shifted by 3 days).
-3dL: This is the difference between the close price and the lowest low of the next 3 days (shifted by 3 days).
-RewardRisk: This is the ratio of 3dH to 3dL.
-CPctile: This is the percentile of the close price within the range of the high and low prices.
-OPctile: This is the percentile of the open price within the range of the high and low prices.
-Range: This is the difference between the high and low prices.
-RangeT: This is the standardized difference between the high and low prices over the last 30 days.

### Data Preprocessing:

Performed several preprocessing steps on the DataFrame, including:

-Computing the percent change of the close price and the mid price.
-Setting the target close price to the percent change of the close price.
-Removing columns for the open price, high price, low price, volume, trade count, volume weighted average price (vwap), 3dH, 3dL, and Range.

### Add Signal Column:

Added a "Signal" column to the DataFrame and sets its values to 0.0.

It reads in a pandas DataFrame df and iterates through each row using iterrows(). It checks the value of the "target_close" column in each row and sets the "Signal" column to 1.0 if the value is greater than or equal to 0, and to 0.0 if the value is less than 0.

Next, shifts the "Signal" column up by one row using the shift() function and assigns the result to the "Signal" column itself. It also creates a new column called "FutureSlope" by shifting the "target_close" column up by one row.

Finally, the DataFrame df is returned with the updated columns.



### Data exploration and analysis:

-The first line of the code gets the column names of a DataFrame 'df' and assigns them to a list variable 'lstColumns'.
-The second line displays the list of column names using the 'display()' function.
-The third line creates an empty DataFrame called 'dfopt'.

The next block of code is a 'for' loop that iterates through each column in 'lstColumns'.
Within the loop, the 'hvplot.scatter()' function is used to create a scatter plot of the column against the 'FutureSlope' column.
The 'dfopt' DataFrame is then assigned with two columns - 'FutureSlope' and the current column from 'lstColumns'.
The correlation between these two columns is calculated using the 'corr()' method and is displayed using the 'display()' function.
The code essentially generates scatter plots and correlation values for each column in the DataFrame against the 'FutureSlope' column. This is useful for identifying relationships between variables and identifying potential predictors of the 'FutureSlope' column. The code can be modified as per the requirements of the user. Dropped NAs, set y as the Signal, then review the value counts.
 
 ## Machine Learning-Support Vector Machine:
 
It defines a machine learning pipeline for a Support Vector Machine (SVM) classifier. The pipeline is designed to train the classifier on a set of input features (50Pct, 61Pct, RangeT, OPctile, and CPctile) using a historical dataset, and then test the performance of the classifier on a separate test dataset.

The pipeline consists of the following steps:

### Data preparation:

The input features are loaded from a Pandas DataFrame named 'df', which contains historical data for the assets being analyzed. The features are stored in a new DataFrame named 'X', and any rows with missing data are dropped from the DataFrame.

### Training and testing data splitting: 

The historical data is split into a training set and a testing set. The start date of the training set is defined as the minimum index date in the X DataFrame, and the end date is set to 60 months after the start date. The testing set includes all data after the end of the training set.

### Feature scaling: 

The input features are standardized using a StandardScaler instance. This scaling is done separately on the training and testing sets to avoid data leakage.

### Model training: 

An SVM classifier is trained on the scaled training data.

### Model testing: 

The trained SVM classifier is used to predict the labels of the scaled testing data, and the performance of the model is evaluated using a classification report.

It outputs the training begin and end dates, displays the first and last 5 rows of the training dataset, and the first 5 rows of the testing dataset. The scaled training and testing datasets are also generated, but not displayed.

 Performed the following operations:

Instantiates an SVM classifier model instance by calling the SVC function from the svm module and setting the probability parameter to True.
Fits the model to the training data by calling the fit method on the instantiated model object and passing the scaled training data X_train_scaled and y_train as arguments.
Uses the trained model to predict the target values for the testing data by calling the predict method on the model object and passing the scaled testing data X_test_scaled as an argument. The predicted values are stored in the svm_pred variable.
Generates a classification report for the model evaluation by calling the classification_report function from the sklearn.metrics module and passing the true target values y_test and the predicted values svm_pred as arguments. The resulting report is stored in the svm_testing_report variable.
Prints the classification report by calling the print function and passing the svm_testing_report variable as an argument.
The classification report provides a summary of the model's performance, including metrics such as precision, recall, and F1 score for each target class, as well as the overall accuracy and the macro-averaged metrics. This allows for a thorough evaluation of the model's ability to correctly classify the testing data.

### instantiate SVC classifier model instance:

Created a predictions_df DataFrame which contains the predicted signal, actual returns and trading algorithm returns. The index of the DataFrame is set to X_test.index.

The predicted_signal column is populated with the predicted signals generated by the SVM model which are stored in the svm_pred variable.

The actual_returns column is set to the closing price of the stock which is retrieved from the df DataFrame.

The trading_algorithm_returns column is calculated by multiplying the actual_returns by the predicted_signal.

Finally, the cumulative returns for both the actual_returns and the trading_algorithm_returns are calculated using the cumprod() function and plotted.

The (1 + predictions_df[["actual_returns", "trading_algorithm_returns"]]).cumprod().plot() line of code adds 1 to both columns and takes the cumulative product using the cumprod() function. The resulting DataFrame is then plotted to show the cumulative returns of the actual returns and trading algorithm returns.

Used the same process for:

- 1-day close with 3-Day moving average

- 1-day close with Mid-point values

- 60-minute closing prices

- 60-minute close with 3-Day moving average

- 60-minute close with Mid-point values

Notebook is attached for your reference.


## Dollar Images:

### 1 Day Close

![Dollar 1 Day Close](/Fiboimages/Dollar1DayClose.png)

### 1 Day 3 MA:

![Dollar 1 Day 3 MA](/Fiboimages/Dollar1Day3MA.png)

### 1 Day Mid:

![Dollar 1 Day Mid](/Fiboimages/Dollar1DayMid.png)

### 60 Min Close:

![Dollar  60 Min Close](/Fiboimages/Dollar60MinClose.png)

### 60 Min 3 MA:

![Dollar 60 Min 3 MA](/Fiboimages/Dollar60Min3MA.png)

### 60 min Mid:

![Dollar 60 Min Mid](/Fiboimages/Dollar60MinMid.png)



## XLB Images:

### 1 Day Close:

![XLB 1 Day Close](/Fiboimages/XLB1DayClose.png)

### 1 Day 3 MA:

![XLB 1 Day 3 MA](/Fiboimages/XLB1Day3MA.png)

### 1 Day Mid:

![XLB 1 Day Mid](/Fiboimages/XLB1DayMid.png)

### 60 Min Close:

![XLB  60 Min Close](/Fiboimages/XLB60MinClose.png)

### 60 Min 3 MA:

![XLB 60 Min 3 MA](/Fiboimages/XLB60Min3MA.png)

### 60 Min Mid:

![XLB 60 Min Mid](/Fiboimages/XLB60MinMid.png)


## XLE Images:

### 1 Day Close:

![XLE 1 Day Close](/Fiboimages/XLE1DayClose.png)

### 1 Day 3 MA:

![XLE 1 Day 3 MA](/Fiboimages/XLE1Day3MA.png)

### 1 Day Mid:

![XLE 1 Day Mid](/Fiboimages/XLE1DayMid.png)

### 60 Min Close:

![XLE  60 Min Close](/Fiboimages/XLE60MinClose.png)

### 60 Min 3 MA:

![XLE 60 Min 3 MA](/Fiboimages/XLE60Min3MA.png)

### 60 Min Mid:

![XLE 60 Min Mid](/Fiboimages/XLE60MinMid.png)


## XLF Images:

### 1 Day Close:

![XLF 1 Day Close](/Fiboimages/XLF1DayClose.png)

### 1 Day Mid:

![XLF 1 Day Mid](/Fiboimages/XLF1DayMid.png)

### 60 Min Close:

![XLF  60 Min Close](/Fiboimages/XLF60MinClose.png)

### 60 Min 3 MA:

![XLF 60 Min 3 MA](/Fiboimages/XLF60Min3MA.png)

### 60 Min Mid:

![XLF 60 Min Mid](/Fiboimages/XLF60MinMid.png)


## XLI Images:

### 1 Day Close:

![XLI 1 Day Close](/Fiboimages/XLI1DayClose.png)

### 1 Day 3 MA:

![XLI 1 Day 3 MA](/Fiboimages/XLI1Day3MA.png)

### 1 Day Mid:

![XLI 1 Day Mid](/Fiboimages/XLI1DayMid.png)

### 60 Min Close:

![XLI  60 Min Close](/Fiboimages/XLI60MinClose.png)

### 60 Min 3 MA:

![XLI 60 Min 3 MA](/Fiboimages/XLI60MinMA.png)

### 60 Min Mid:

![XLI 60 Min Mid](/Fiboimages/XLI60MinMid.png)


## XLK Images:

### 1 Day Close:

![XLK 1 Day Close](/Fiboimages/XLK1DayClose.png)

### 1 Day 3 MA:

![XLK 1 Day 3 MA](/Fiboimages/XLK1Day3MA.png)

### 1 Day Mid:

![XLK 1 Day Mid](/Fiboimages/XLK1DayMid.png)

### 60 Min Close:

![XLK  60 Min Close](/Fiboimages/XLK60MinClose.png)

### 60 Min 3 MA:

![XLK 60 Min 3 MA](/Fiboimages/XLK60Min3MA.png)

### 60 Min Mid:

![XLK 60 Min Mid](/Fiboimages/XLK60MinMid.png)


## XLP Images:

### 1 Day Close:

![XLP 1 Day Close](/Fiboimages/XLP1DayClose.png)

### 1 Day 3 MA:

![XLP 1 Day 3 MA](/Fiboimages/XLP1Day3MA.png)

### 1 Day Mid:

![XLP 1 Day Mid](/Fiboimages/XLP1DayMid.png)

### 60 Min Close:

![XLP  60 Min Close](/Fiboimages/XLP60MinClose.png)

### 60 Min 3 MA:

![XLP 60 Min 3 MA](/Fiboimages/XLP60Min3MA.png)

### 60 Min Mid:

![XLP 60 Min Mid](/Fiboimages/XLP60MinMid.png)


## XLU Images:

### 1 Day Close:

![XLU 1 Day Close](/Fiboimages/XLU1DayClose.png)

### 1 Day 3 MA:

![XLU 1 Day 3 MA](/Fiboimages/XLU1Day3MA.png)

### 1 Day Mid:

![XLU 1 Day Mid](/Fiboimages/XLU1DayMid.png)

### 60 Min Close:

![XLU  60 Min Close](/Fiboimages/XLU60MinClose.png)

### 60 Min 3 MA:

![XLU 60 Min 3 MA](/Fiboimages/XLU60Min3MA.png)

### 60 Min Mid:

![XLU 60 Min Mid](/Fiboimages/XLU60MinMid.png)


## XLV Images:

### 1 Day Close:

![XLV 1 Day Close](/Fiboimages/XLV1DayClose.png)

### 1 Day 3 MA:

![XLV 1 Day 3 MA](/Fiboimages/XLV1Day3MA.png)

### 1 Day Mid:

![XLV 1 Day Mid](/Fiboimages/XLV1DayMid.png)

### 60 Min Close:

![XLV  60 Min Close](/Fiboimages/XLV60MinClose.png)

### 60 Min 3 MA:

![XLV 60 Min 3 MA](/Fiboimages/XLV60Min3MA.png)

### 60 Min Mid:

![XLV 60 Min Mid](/Fiboimages/XLV60MinMid.png)


## XLY Images:

### 1 Day Close:

![XLY 1 Day Close](/Fiboimages/XLY1DayClose.png)

### 1 Day 3 MA:

![XLY 1 Day 3 MA](/Fiboimages/XLY1Day3MA.png)

### 1 Day Mid:

![XLY 1 Day Mid](/Fiboimages/XLY1DayMid.png)

### 60 Min Close:

![XLY  60 Min Close](/Fiboimages/XLY60MinClose.png)

### 60 Min 3 MA:

![XLY 60 Min 3 MA](/Fiboimages/XLY60Min3MA.png)

### 60 Min Mid:

![XLY 60 Min Mid](/Fiboimages/XLY60MinMid.png)




# Using Stochastic as an Indicator:


Imports necessary libraries such as finta, pandas, yfinance, hvplot.pandas, talib, sklearn, seaborn, matplotlib, and others.I
The code defines a list of S&P 500 sector ETFs, downloads the ETF data from Yahoo Finance for the specified time period (from "2008-01-01" to "2022-12-31") with daily frequency, and creates a pandas DataFrame "Close_prices" with the "Close" prices of all the ETFs. The DataFrame is also renamed with column names representing each ETF category.

Then, the code uses hvplot library to plot the closing prices of all the ETFs against the year.

The purpose of the code is to visualize the historical closing prices of S&P 500 sector ETFs and to analyze the trends and patterns in the ETFs. The code can be useful for investment purposes, for instance, by identifying which sector ETFs perform better than others over time.

![Sector Etfs](/StocImages/sectorEtfs.png)


### Calculate Stochastic Indicator for S&P 500 ETF Ticker XLK:

Download S&P 500 ETF data for the "XLK" ticker from Yahoo Finance using the yfinance library. It then generates buy and sell signals based on the Stochastic Oscillator indicator and calculates the moving average of the closing prices using a window of 50 periods.

The code also adds the entry and exit signals to the dataset based on the Stochastic Oscillator signals, and calculates the difference between consecutive signals to obtain the points in time at which a position should be taken, 1 or -1. It then visualizes the entry and exit points and moving averages relative to the close price using the hvplot library.

The resulting plot shows the entry and exit points as purple and orange arrows, respectively, overlaid on the close price and moving average plot for XLK. The plot provides a visualization of how the Stochastic Oscillator indicator can be used to generate buy and sell signals for an investment in XLK. Finally, sets the index of the DataFrame to the "Date" column.

The XLK stock, which is the Technology Select Sector SPDR ETF. The code uses a dual moving average crossover strategy to determine when to buy or sell a 500 share position. When the crossover signal equals 1, the algorithm buys the position. Otherwise, it sells the position. The initial capital for the trading is set to 100,000.

The algorithm tracks the portfolio holdings, portfolio cash, and portfolio total, and calculates the portfolio daily returns and portfolio cumulative returns. It also visualizes the exit position and entry position relative to the total portfolio value and displays the total portfolio value.

The code also calculates the annualized volatility of the option returns for XLK and calculates the pivot and future pivot values for the XLK stock.


The strategy is a dual moving average crossover , calculates various portfolio metrics, and visualizes the portfolio value. It also calculates various other technical indicators for the XLK stock.

![XLK Signals](/StocImages/XLKSignals.png)

![XLK Equity](/StocImages/XLKEquity.png)

The code also calculates the Y value for each row in the dataframe. The Y value is based on the mid-price of the high and low price of the stock. If the Y value is greater than 0, it is set to 1, otherwise, it is set to 0. The code also calculates the MA14Slope and ZScore14 values for the stock.

The code assigns a copy of selected columns from a DataFrame df to a new DataFrame called X. The columns selected are "slowk", "slowd", "MA14Slope", and "ZScore14". The shift() function is used to shift the index by one and drops the resulting NaN values. This is done to align the data with the target variable for training a machine learning model.

The copy() function is used to create a new copy of the DataFrame so that any changes made to the new DataFrame X do not affect the original DataFrame df.

Finally, the code displays the first five and last five rows of the X DataFrame using the head() and tail() functions respectively, and then saves the X DataFrame to a CSV file named X.csv. The saved CSV file will contain the shifted and selected columns data that can be used for further analysis and machine learning modeling.


 ## Machine Learning-Support Vector Machine:

The code performs time-series data preparation for a machine learning model. It first creates a new Pandas Series called y by copying the Y column from a DataFrame called df.

Next, the code imports the DateOffset function from the pandas.tseries.offsets module, which will be used to set the training and testing periods. It then selects the start of the training period as the minimum index value of the X DataFrame and displays the training begin date.

After that, it sets the ending period for the training data with an offset of 115 months, which corresponds to approximately 9.5 years. The code then displays the training end date.

Subsequently, the it generates the X_train and y_train DataFrames by selecting rows from X and y that fall within the training period, which is defined by training_begin and training_end.

Finally, the it generates the X_test and y_test DataFrames by selecting rows from X and y that fall after the training period, which is defined by training_end. These DataFrames will be used to test the machine learning model.

The machine learning modeling using the Support Vector Machine (SVM) algorithm to predict trading signals based on time-series data.

First, the code imports the required StandardScaler class from the sklearn.preprocessing module, which will be used to scale the input data. It creates an instance of the StandardScaler class called scaler.

Next, it fits the scaler to the training data using the fit() method of the scaler instance, and then transforms both the training and testing data using the transform() method of the scaler instance. The transformed data is stored in X_train_scaled and X_test_scaled DataFrames.

After that, the code imports the SVM model from the sklearn module, along with the classification_report function from the sklearn.metrics module. It creates an instance of the svm.SVC class called svm_model.

Then, it fits the model to the training data using the fit() method of the svm_model instance with the scaled training data X_train_scaled and y_train. It uses the trained model to predict the trading signals for the training data and stores the predicted signals in training_signal_predictions.

Next, it evaluates the model's performance on the training data using the classification_report() function and stores the report in training_report. It displays the report.

After that, it uses the trained model to predict the trading signals for the testing data and stores the predicted signals in testing_signal_predictions.

Finally, it evaluates the model's ability to predict the trading signal for the testing data using the classification_report() function and stores the report in testing_report. It displays the report. The reports contain metrics such as precision, recall, and F1-score for each class. These metrics help to evaluate the model's performance.

Then, it creates a DataFrame named predictions_df which will store the predicted signal, actual returns, and trading algorithm returns. The predicted_signal is the output of a machine learning model for a testing dataset named X_test. The actual_returns represent the actual returns of the trading dataset stored in the DataFrame df.

Then calculates the trading_algorithm_returns which are the product of the actual_returns and the predicted_signal. This product represents the returns of the trading algorithm using the predicted signals.

Finally, it calculates and plots the cumulative returns for both actual_returns and trading_algorithm_returns using the cumprod() function. The resulting plot shows the cumulative returns of the trading algorithm compared to the actual returns for the trading dataset.


![XLK SVM](/StocImages/XLKSVM.png)


## Dollar Images:

![Dollar Signals](/StocImages/DOLLARSignal.png)

![Dollar Equity](/StocImages/DOLLAREquity.png)

![Dollar SVM](/StocImages/DOLLARSVM.png)

## XLB Images:

![XLB Signals](/StocImages/XLB Signal.png)

![XLB Equity](/StocImages/XLBEquity.png)

![XLB SVM](/StocImages/XLBSVM.png)

## XLE Images:

![XLE Signals](/StocImages/XLESignal.png)

![XLE Equity](/StocImages/XLEEquity.png)

![XLE SVM](/StocImages/XLESVM.png)

## XLF Images:

![XLF Signals](/StocImages/XLF Signal.png)

![XLF Equity](/StocImages/XLFEquity.png)

![XLF SVM](/StocImages/XLFSVM.png)


## XLI Images:

![XLI Signals](/StocImages/XLISignal.png)

![XLI Equity](/StocImages/XLIEquity.png)

![XLI SVM](/StocImages/XLISVM.png)


## XLP Images:

![XLP Signals](/StocImages/XLPSignal.png)

![XLP Equity](/StocImages/XLPEquity.png)

![XLP SVM](/StocImages/XLPSVM.png)


## XLU Images:

![XLU Signals](/StocImages/XLUSignal.png)

![XLU Equity](/StocImages/XLUEquity.png)

![XLU SVM](/StocImages/XLUSVM.png)


## XLV Images:

![XLV Signals](/StocImages/XLVSignal.png)

![XLV Equity](/StocImages/XLVEquity.png)

![XLV SVM](/StocImages/XLVSVM.png)


## XLY Images:

![XLY Signals](/StocImages/XLYSignal.png)

![XLY Equity](/StocImages/XLYEquity.png)

![XLY SVM](/StocImages/XLYSVM.png)



