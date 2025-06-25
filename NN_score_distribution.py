import pandas as pd

f = 16
# df = pd.read_csv('out/f_16/filtered_train.csv')
df = pd.read_csv('out/f_16/train_sftmx.csv')
# df = pd.read_csv('out/f_16/test.csv')


for i in range(1,11) :
    classGroup = df[df['label'] == i-1]

    mean = classGroup.iloc[:, f+i-1].mean()
    sd = classGroup.iloc[:, f+i-1].std()

    print("Mean for class label ", i-1, " is ", mean)
    print("SD for class label ", i-1, " is ", sd)


