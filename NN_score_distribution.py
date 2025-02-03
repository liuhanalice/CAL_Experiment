import pandas as pd

f = 16
df = pd.read_csv('out/f_16/train.csv')

for i in range(1,11) :
    classGroup = df[df['label'] == i-1]

    mean = classGroup.iloc[:, f+i-1].mean()
    variance = classGroup.iloc[:, f+i-1].var()

    print("Mean for class label ", i-1, " is ", mean)
    print("Variance for class label ", i-1, " is ", variance)


