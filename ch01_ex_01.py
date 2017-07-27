import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import util
from sklearn.linear_model import LinearRegression

#collecting the data
data_dir = "../data/"
oecd_bli = pd.read_csv(data_dir+ "life_satisfaction.csv", thousands=',')
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
# print(oecd_bli.head(2))
# print(oecd_bli["Life satisfaction"].head())

gdp_per_capita = pd.read_csv(data_dir + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
gdp_per_capita.set_index("Country", inplace=True)
# print(gdp_per_capita.head(2))

#prepare the data
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
full_country_stats.sort_values(by="GDP per capita", inplace=True)
# print(full_country_stats.head(2))

# print(full_country_stats[['GDP per capita', 'Life satisfaction']].loc['United States'])

#preparing Test and Train data
remove_indices = [0, 1, 6, 8, 33, 35]
# print(full_country_stats[['GDP per capita', 'Life satisfaction']].iloc[[0]])
keep_indices = list(set(range(36)) - set(remove_indices))

sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]
# print(full_country_stats[['GDP per capita', 'Life satisfaction']].loc['United States'])
# print(sample_data[["GDP per capita", 'Life satisfaction']].loc['United States'])
# print(missing_data[["GDP per capita", 'Life satisfaction']].loc['United States'])
# print(full_country_stats[['GDP per capita']]).iloc[[34]]

#plot
sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
# plt.axis([0, 60000, 0, 10])

# #test sample
# position_text = {
#     "Hungary": (5000, 1),
#     "Korea": (18000, 1.7),
#     "France": (29000, 2.4),
#     "Australia": (40000, 3.0),
#     "United States": (52000, 3.8),
# }

# for country, pos_text in position_text.items():
# 	pos_data_x, pos_data_y = sample_data.loc[country]
# 	country = "U.S" if country == "United States" else country
# 	plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text, arrowprops=dict(facecolor="black", width=0.5, shrink=0.1, headwidth=5))
# 	plt.plot(pos_data_x, pos_data_y, "ro")

# #save fig
# plt=util.save_fig(plt, 'money_happy_scatterplot')
# plt.show()

model = LinearRegression()
X = np.c_[full_country_stats['GDP per capita']]
y = np.c_[full_country_stats['Life satisfaction']]
model.fit(X, y)

# plt.axis([0, 60000, 0, 10])
# X=np.linspace(0, 60000, 1000)
# t0, t1 = model.intercept_[0], model.coef_[0][0]
# plt.plot(X, t0 + t1*X, "b")

# print(t0, t1)

test_x = full_country_stats.iloc[[1]]['GDP per capita']
test_y = model.predict(test_x)

# plt.plot([test_x, test_x], [0, test_y], "r--")

# plt = util.save_fig(plt, 'pred_with_best_fit')
# plt.show()
print(test_y, full_country_stats.iloc[[1]]['Life satisfaction'])
model1 = LinearRegression()
print( len(sample_data['GDP per capita']), len(sample_data['Life satisfaction']))
model1.fit(np.c_[sample_data['GDP per capita']], np.c_[sample_data['Life satisfaction']])
print("finished")
test_x1 = sample_data.iloc[[1]]['GDP per capita']
test_y1 = model1.predict(test_x1)
print(model.intercept_[0], model.coef_[0][0], model1.intercept_[0], model1.coef_[0][0])
print(test_y1, sample_data.iloc[[1]]['Life satisfaction'])