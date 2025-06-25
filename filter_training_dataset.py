import pandas as pd

f = 16
df = pd.read_csv('out/f_16/train_sftmx.csv')
# df = pd.read_csv('out/f_16/test.csv')

tau = 0.90
alpha = 0.4
filtered_bothside = True

filtered_df = pd.DataFrame()

for i in range(1, 11):
    class_group = df[df['label'] == i-1]
    column_idx = f+i-1
    mean = class_group.iloc[:, column_idx].mean()
    
    if filtered_bothside:
        lower_bound = mean - (tau/2) * mean
        upper_bound = mean + (tau/2) * mean
        filtered_class_group = class_group[
            (class_group.iloc[:, column_idx] >= lower_bound) &
            (class_group.iloc[:, column_idx] <= upper_bound)
        ]
        # save_path='out/f_16/filtered_train.csv'
        save_path='out/f_16/filtered_train_sftmx.csv'
    else:  
        lower_bound = mean - (0.5-alpha) * mean
        filtered_class_group = class_group[
            (class_group.iloc[:, column_idx] >= lower_bound)
        ]
        save_path='out/f_16/filtered_lower_train.csv'

    filtered_df = pd.concat([filtered_df, filtered_class_group])

filtered_df.to_csv(save_path, index=False)
print("Done")

