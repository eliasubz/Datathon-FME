import pandas as pd
date_cols = ["phase_in", "phase_out"]

# Load data
train = pd.read_csv("data/train_subset.csv", delimiter=';', parse_dates=date_cols,
    infer_datetime_format=True,)
test = pd.read_csv("data/test_subset.csv", delimiter=';', parse_dates=date_cols,
    infer_datetime_format=True,)

import matplotlib.pyplot as plt
import pandas as pd

# Example: make sure your columns match your dataset
# Columns: ['piece_id', 'production', 'phase_in', 'phase_out']


# Aggregate weekly sales per week (you can also group by piece_id if needed)
agg = train.groupby('date_week')['weekly_sales'].sum().reset_index()
agg.rename(columns={'weekly_sales': 'wsales'}, inplace=True)

# Plot total weekly sales over time
# plt.figure(figsize=(12, 6))
# plt.plot(agg['date_week'], agg['wsales'], marker='o')
# plt.xlabel('Week')
# plt.ylabel('Weekly Sales')
# plt.title('Total Weekly Sales Over Time')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
