#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.diagnostic as dg
from pandas.tools.plotting import autocorrelation_plot

df = pd.read_csv("subs.txt", delimiter="\t")

#typy danych
print(df.dtypes)

#head
print(df.head())

#czyszczenie
df["stop_type"] = df["stop_type"].fillna(value="N")
df["stop_date"] = df["stop_date"].fillna(value="2020-01-01 00:00:00")


#wstepne grupowanie
print(df.groupby(["stop_type"]).size())

#nowe typy danych
df["rate_plan"] = df["rate_plan"].astype("category")
df["market"] = df["market"].astype("category")
df["channel"] = df["channel"].astype("category")
df["stop_type"] = df["stop_type"].astype("category")
df["start_date"] = pd.to_datetime(df["start_date"])
df["stop_date"] = pd.to_datetime(df["stop_date"])
#remove na data, only 181
df.dropna(subset=["start_date"], inplace=True)


df["stop_year"] = df["stop_date"].dt.year
df["stop_month"] = df["stop_date"].dt.month
df["stop_day"] = df["stop_date"].dt.dayofyear


df["start_year"] = df["start_date"].dt.year
df["start_month"] = df["start_date"].dt.month
df["start_day"] = df["start_date"].dt.dayofyear

print(df.head())

group_parameters = ["rate_plan", "market", "channel"]

customers_stats = dict()

for param in group_parameters:
    customers_stats[param] = dict()
    new_customers = df.groupby([param, "start_date"]).size()
    retired_customers = df.groupby([param, "stop_date"]).size()
    new_df = pd.DataFrame(new_customers)
    retired_df = pd.DataFrame(retired_customers)
    new_df.columns = ["new customers"]
    #new_df.index.name = "date"
    retired_df.columns = ["retired customers"]
    #retired_df.index.name = "date"
    for category in df[param].cat.categories:
        new_decomposed = new_df.loc[category]
        retired_decomposed = retired_df.loc[category]
        customer_stats_by_year = new_decomposed.join(retired_decomposed, how='outer')
        customer_stats_by_year.fillna(value=0, inplace=True)
        customer_stats_by_year["active customers"] = customer_stats_by_year["new customers"] - customer_stats_by_year["retired customers"]
        customer_stats_by_year["total"] = customer_stats_by_year["active customers"].cumsum()
        customer = customer_stats_by_year[1:-1]
        not_truncated = customer.loc['2004-01-01':]
        customers_stats[param][category] = not_truncated
        

#teraz mozey wyświetlić statystyki dla poszczególnych elementów agregacji
        


#ogolne
new_customers = df.groupby(["start_date"]).size()
retired_customers = df.groupby(["stop_date"]).size()
new_df = pd.DataFrame(new_customers)
retired_df = pd.DataFrame(retired_customers)
new_df.columns = ["new customers"]
new_df.index.name = "date"
retired_df.columns = ["retired customers"]
retired_df.index.name = "date"
customer_stats_by_year = new_df.join(retired_df, how='outer')
customer_stats_by_year.fillna(value=0, inplace=True)
customer_stats_by_year["active customers"] = customer_stats_by_year["new customers"] - customer_stats_by_year["retired customers"]
customer_stats_by_year["total"] = customer_stats_by_year["active customers"].cumsum()
customer = customer_stats_by_year[1:-1]

not_truncated = customer.loc['2004-01-01':]

res = sm.tsa.seasonal_decompose(customer["total"], model='aditive', freq=1)
res.plot()

onehot = OneHotEncoder(categorical_features=[3])
le = LabelEncoder()


test_fe = df.head()

b = test_fe.loc[:,["rate_plan", "market" , "channel", "stop_type"]]

c = onehot.fit_transform(b).toarray()
c = b.values
b["rate_plan"] = le.fit_transform(b["rate_plan"])

b["market"] = le.fit_transform(b["market"])
b["channel"] = le.fit_transform(b["channel"])
b["stop_type"] = le.fit_transform(b["stop_type"])