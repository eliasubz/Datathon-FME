import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

date_cols = ["phase_in", "phase_out"]

# Load data
train = pd.read_csv("data/train_subset.csv", delimiter=';', parse_dates=date_cols,
    infer_datetime_format=True,)
test = pd.read_csv("data/test_subset.csv", delimiter=';', parse_dates=date_cols,
    infer_datetime_format=True,)

# Create unique weeks 
train['date_week'] = pd.to_datetime(train['year'].astype(str) + '-W' + train['num_week_iso'].astype(str) + '-1', format='%G-W%V-%u')
print(train['date_week'].head())

# Assume `agg` from previous step: weekly sales per week
agg = train.groupby('date_week')['weekly_sales'].sum().reset_index()

ts = agg.set_index('date_week')['weekly_sales']

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_acf(ts, ax=axes[0], lags=30)
axes[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(ts, ax=axes[1], lags=30, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()