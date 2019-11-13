import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nanDict = {}
df = pd.read_excel('creditcard_data.xls', header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)# Features and targets 

df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder = OneHotEncoder(categories = "auto")

X = ColumnTransformer(
    #[("", onehotencoder, [3],)], 
    [("", onehotencoder, [1, 2, 3, 4]),],
    remainder="passthrough"
).fit_transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 1-0.5)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()#(with_mean = False)
X_train = sc.fit_transform(X_train.toarray(), y_train)
X_test = sc.transform(X_test.toarray())

# One-hot's of the target vector
#Y_train_onehot, Y_test_onehot = onehotencoder.fit_transform(y_train), onehotencoder.fit_transform(y_test)