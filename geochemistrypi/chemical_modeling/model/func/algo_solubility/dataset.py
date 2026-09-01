from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler


def train_test_dfs(df, target="SCSS", test_size=0.25, rand_state=100, scale=True):
    df_x = df.drop(target, axis=1)
    df_y = df[target]
    x_train, x_test, y_train, y_test = tts(df_x, df_y, test_size=test_size, random_state=rand_state)
    if scale:
        scaler = StandardScaler()
        x_train_norm = scaler.fit_transform(x_train)
        x_test_norm = scaler.transform(x_test)
        return x_train_norm, x_test_norm, y_train, y_test, scaler
    else:
        return x_train, x_test, y_train, y_test


def preprocess(df):
    df.drop([0], axis=0, inplace=True)
    df.dropna(inplace=True)
    return df
