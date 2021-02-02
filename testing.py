import pandas as pd


def load_data():
    df = pd.read_csv('./train.csv')
    return df


def data_binomailization(degree, df):
    df_set = [df]
    for i in range(2, degree + 1):
        df_temp = df**i
        df_set.append(df_temp)
    return pd.concat(df_set, axis=1)


da = load_data()
da = da['SalePrice']
da =data_binomailization(3, da)
print(da)