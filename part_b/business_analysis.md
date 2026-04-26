###Part B: Business Case Analysis


B1. Problem Formulation


(Part-a) Machine Learning Problem

This problem is about predicting how many items will be sold based on different factors. Target variable: items_sold Input 
features: store size, location type, promotion type, competition density, customer behavior, month, whether it is a weekend 
or festival, etc. This is a supervised learning regression problem because: We already have past data (inputs + output) 
The output (items_sold) is a number (continuous value) So, the goal is to learn from past data and predict future sales.


(Part-b) Why use items_sold instead of revenue

Using items_sold is better than revenue because: Revenue can be affected by price changes,
discounts, or expensive products But items_sold directly shows how well a promotion is working in terms of quantity sold 
For example: A store may have high revenue just because products are expensive, not because the promotion is effective. 
This shows an important principle: The target variable should directly match the business goal. In this case, the goal is 
to increase sales volume, so items_sold is a better choice.


(Part-c) Better modelling strategy

Instead of using one single model for all stores, we can: Use separate models for different store types (urban, semi-urban, 
rural) OR Add store-specific features or group-based models This is better because: Different locations behave differently 
The same promotion may work in urban areas but not in rural areas So, this approach makes the model more accurate and 
realistic.



B2. Data and EDA Strategy


(Part-a) Data Joining and Structure

We have 4 tables: Transactions Store attributes Promotion details Calendar
We can join them using: store_id → to combine store information date / transaction_date → to combine calendar and promotion 
data Final dataset:
Each row represents: One store for one month Before modeling, we can aggregate: Total items sold per store per month Average 
basket size Total visits This makes the dataset clean and suitable for modeling.


(Part b) EDA Strategy

Before building the model, I would perform: Sales distribution plot To see how items_sold is spread Helps detect outliers 
Promotion vs sales comparison (bar chart) To see which promotion performs better Helps in feature importance Time trend 
(line chart) To see how sales change over months Helps capture seasonality Correlation heatmap To check relationships between 
variables Helps in feature selection These analyses help understand patterns and improve feature engineering.


(Part-c) Handling imbalance (80% no promotion)

This imbalance can cause the model to: Focus more on “no promotion” cases Ignore the impact of promotions To fix this, 
we can: Use balanced sampling Give more importance (weight) to promotion cases Analyze promotions separately This ensures 
the model learns the real impact of promotions.




B3. Model Evaluation and Deployment



(Part-a) Train-test split and metrics

Since the data is time-based: Use: First 2.5 years → training Last 6 months → testing A random split is not 
suitable because: It mixes past and future data Gives unrealistic performance
Metrics to use: RMSE → shows overall error (penalizes large mistakes) MAE → shows average error
These help understand how far predictions are from actual sales.



(Part-b) Explaining different recommendations

If the model gives different promotions for different months: We check feature importance
For example:
December may have festivals → Loyalty Points works better March may be normal → Discount works better
We can explain to the team: “The model is choosing different promotions because customer behavior and conditions change over time.
This builds trust in the model.


(Part-c) Deployment process

To use the model in real life:
Save the model Use tools like joblib or pickle Prepare new data every month Collect latest store, promotion, and calendar data Apply same preprocessing steps Make predictions Predict best promotion for each store Monitoring Track prediction error over time If performance drops, retrain the model
This ensures the model stays useful and accurate.
