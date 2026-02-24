import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import json
import os
import sys

def load_config():
    public_defaults = {
        "sma_window": 20, "volume_window": 100, "volatility_window": 10,
        "rsi_window": 31, "target_window": 3, "target_pct": 1.05
    }
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            return json.load(f)
    return public_defaults 

cfg = load_config()

ticker = input("Enter the ticker for the stock (e.g. NVDA, RELIANCE.NS) : ").strip().lower()

#Downloaded the data for Nifty too because we want to compare the stock's move with nifty's move to find out its Relative Strength.
df = yf.download(ticker, period="10y", progress=False)
nifty = yf.download("^NSEI", period= "10y", progress= False)

print(f"\nFetching data for {ticker.upper()} from Yahoo Finance.....")

#As YFinance sometimes give us a MultiIndex Data, out model gets confused in choosing a specifin index so we drop our first level of index
#and then pass on the data for further steps.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.droplevel(1)

#As Nifty and our Stock both trade at different scales so we can't compare there daily returns in 'numbers' instead we need to 
# compare their Percentage Returns
df_return = df['Close'].pct_change()
nifty_return = nifty['Close'].pct_change()

return_diff = (df_return - nifty_return)

"""
GOAL: To code very commonly used indicators in order to use them in predicting the next move of our stock.
WHY: We cant just guess wether the stock is going to show an upside move or other, so we need something backing our claim and that
     backing is provided by Indicators which improve the odds on a side.
METHOD: Using provided Data from yfinance and moulding it according to our needs for moving averages, RSI windows, and volume thresholds.
"""
#INDICATOR - 1 : Relative Strength
#We added a new column to our data frame giving us the Relative Strength of Stock as compared to Nifty
df['Relative_Strength'] = return_diff

#INDICATOR - 2 : Moving Average Distance
df['Dist_from_SMA'] = (df['Close'] / df['Close'].rolling(window= cfg['sma_window']).mean()) - 1

#INDICATOR - 3: Surge in Volume based on lookback rolling window.
df['Volume_Surge'] = (df['Volume'] / df['Volume'].rolling(window=cfg['volume_window']).mean())

#INDICATOR - 4 : Volatility (Change in avg standard deviation of stock based on rolling window.)
df['Volatility'] = (df['Close'].rolling(window=cfg['volatility_window']).std())

#INDICATOR - 5 : RSI (Basically N0. of winning days / No. of losing days) in past rolling window period.
delta = (df['Close']).diff()
gains = delta.where(delta > 0, 0).rolling(window=cfg['rsi_window']).mean()
loses = abs(delta.where(delta < 0, 0).rolling(window=cfg['rsi_window']).mean())
rel_Str = gains / loses

df['RSI_Metric'] = (100 - (100 / (1+rel_Str)))

"""
The Supervised Machine Learning (Building the Answer Key)

The Time Machine: What it does is, it looks into the future and grabs the absolute HIGHEST price the stock reaches at any point over a specific forward-looking window. 
                  Then, it checks if that future peak hits a predetermined proprietary profit target relative to today's closing price.
                  If YES (meaning a real-world Take-Profit limit order would have successfully triggered), it marks the setup as a WIN (1) and studies the indicators. 
                  If NO, it marks it as a LOSS (0). 
                  This is how the AI builds its rulebook—learning exactly which present-day indicator patterns have the highest probability of triggering a quick momentum pop.
"""
future_price = df['High'].rolling(window=cfg['target_window']).max().shift(-cfg['target_window'])
target_price = df['Close'] * cfg['target_pct']

df['Target'] = (future_price > target_price).astype(int)
df = df.dropna()

features = ['Relative_Strength','Dist_from_SMA', 'Volume_Surge', 'Volatility', 'RSI_Metric']

X = df[features]
Y = df['Target']

#If we are using Time-Period of 5y then training will be done on first 4 years of data and noting that it wont shuffle the data
# so that the model doesnt peak into the future data and cheat. It will test on Year-5.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, shuffle=False)

#We created a base Random Forest Classifier and didn't use Regressor because in this we are predicting (True/False)
# basically 1's and 0's and for that we need a Classifier.
base_rf = RandomForestClassifier(oob_score=True, random_state=42, class_weight= 'balanced')

#We defined a search space to feed into our model so that it can tweak and find many random commbinations, feed that again into the
#model and repeat it again and again until it finds the model with the highest score and then uses those parameters for predictions.
search_space = {
    'n_estimators' : [50,100,200,500],
    'max_depth' : [5,10,20,None],
    'min_samples_leaf' : [1,2,5]
}

#We used Time Series Split because we dont want our model to cheat by peaking into the future data so this library helps us by
#completing cutting off the last 2yrs of data from the model's visibility and only gives him data of first 8 yrs to look into and train
#It divides those 8 yrs in chunks, First it will train of 1st year then test on 2nd yr, next it will train on 1st and 2nd yr and test on 3rd yr and so on.
time_Series_Split_no = TimeSeriesSplit(n_splits=8)

#We use Randomized Search CV and pass on out parameters, we use n_iter 20 as we want our model to choose 20 random combinations from our search space and then train on it.
rf_search = RandomizedSearchCV(estimator=base_rf, param_distributions= search_space, n_iter= 20, cv = time_Series_Split_no, random_state= 42)

#Computer trains 20 different Random Forest Models based on the 20 different combinations it got from Randomized Search Cv.
#Then it scores them all, and when it finds the model with best result it, it takes the setting of that model and builds a new
#model and trains it on our entire X_train dataset and then locks that final project model inside rf_search.
rf_search.fit(X_train,Y_train)

#It goes inside the rf_search machine, grabs the fully trained Random Forest Model that won amongst the 19 others, and assigns it to the variable best_rf.
#best_rf is now our functional predictive model ( best estimator it gives us the model.)
best_rf = rf_search.best_estimator_

# 1. Getting the raw probability scores (Confidence)
# column 0 is 'Probability of 0' and column 1 is 'Probability of 1'
probabilities = best_rf.predict_proba(X_test)

# 2. THRESHOLD OPTIMIZER: Testing different confidence levels to find the sweet spot
print("\n" + "="*55)
print(" 🔍 THRESHOLD OPTIMIZATION SCANNER")
print("="*55)
print("Thresh | Precision (Win%) | Recall (Caught%) | F1-Score")
print("-" * 55)

for t in np.arange(0.50, 0.76, 0.05):
    test_preds = (probabilities[:, 1] >= t).astype(int)
    report = classification_report(Y_test, test_preds, output_dict=True, zero_division=0)
    
    if '1' in report:
        prec = report['1']['precision']
        rec = report['1']['recall']
        f1 = report['1']['f1-score']
        print(f" {t:.2f}   |      {prec*100:>6.2f}%   |     {rec*100:>6.2f}%     |  {f1:.2f}")
    else:
        print(f" {t:.2f}   |       0.00%   |       0.00%     |  0.00")
print("="*55 + "\n")

#After looking at the table of different thresholds we choose one according to out need and pass it on.
user_input = input("Enter your chosen Threshold from the table (e.g., 0.55, 0.60): ")
threshold = float(user_input)

# Now we run the final predictions based on the threshold chosen by us
predictions = (probabilities[:, 1] >= threshold).astype(int)

#3. Predicting accuracy of our model to predict the move 
    # Highes accuracy doesnt mean better model, we need to look for more to confirm.
accuracy = accuracy_score(Y_test, predictions)
print(f"Accuracy of our model to predict the move : {accuracy * 100 :.2f}% ")

#best_params gives us the settings of the winning model
print("\n--- THE WINNING ENGINE SETTINGS ---")
print(rf_search.best_params_)

report_dict = classification_report(Y_test, predictions, output_dict=True)
buy_stats = report_dict['1']

print(f"==========================================")
print(f"   TRADING PERFORMANCE: {ticker.upper()}")
print(f"==========================================")
print(f"Overall Accuracy          : {accuracy * 100:.2f}%")

#PRECISION : Out of all the times our model screamed "BUY", how many times was it actually right? A high precision means when it
#            fires a signal you can trust it.
print(f"Win Rate (Precision)      : {buy_stats['precision']:.2%}")

#RECALL : Out of all the winning moves that occured in the market, how many of them did our model catch? 
#         A low recall means the model is too scared and missing out on winning trades
print(f"Opportunity Caught (Recall): {buy_stats['recall']:.2%}")

#F1 SCORE : It is the mathematical balance between Precision and Recall, it penalizes the models that just screams "BUY" everyday,
#           or models that are too terrified to take a trade
print(f"Consistency Score (F1)    : {buy_stats['f1-score']:.2f}")

#SUPPORT : The raw number of actual winning moves that existed in our data based on specified targets.
print(f"Total Market Opportunities: {int(buy_stats['support'])}")
print(f"==========================================\n")

#Getting the percentage contributed by each indicator in predicting the move so that we know on which our stock depends the most on.
feature_importance_scores = best_rf.feature_importances_
sorted_scores = pd.Series(feature_importance_scores, index=features).sort_values(ascending=False)
print("\n--- FEATURE IMPORTANCE SCORES ---")

for name, score in sorted_scores.items():
    print(f"{name} : {score * 100:.2f}%")


#Plotting the feature importance (Importance of the Indicators used on a bar chart)
plt.figure(figsize= (12,6))
sorted_scores.plot(kind = 'bar', color = 'green')
plt.title(f"Feature Importance scores for : {ticker.upper()}") 
plt.xlabel("Predictive Indicators")
plt.ylabel("Importance Score %")
plt.xticks(rotation = 45)
plt.grid(linestyle = '--', axis='y')
plt.tight_layout()
plt.show()

#LIVE SIGNAL SCANNER 
print(f"==========================================")
print(f"   LIVE SIGNAL FOR {ticker.upper()}")
print(f"==========================================")

# 1. Grab the very last row of data (Today's Data)
latest_data = X.tail(1)

# 2. Get the probability for today
today_proba = best_rf.predict_proba(latest_data)
buy_probability = today_proba[0][1]

# 3. Print the Verdict
print(f"Model Confidence: {buy_probability:.2%}")

if buy_probability >= threshold:
    print(f"VERDICT: 🟢 BUY SIGNAL (Passes your {threshold*100}% requirement)")
elif buy_probability >= (threshold - 0.10):
    print(f"VERDICT: 🟡 WEAK BUY (Close to your threshold, check charts)")
else:
    print("VERDICT: 🔴 NO TRADE (Wait for better setup)")

print("="*40 + "\n")