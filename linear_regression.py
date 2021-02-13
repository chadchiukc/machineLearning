import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LEARNING_RATE = 0.01
ITERATION = 1000
EPSILON = 0.01
PERCENTAGE_OF_TRAINING = 0.9
FEATURES_NUM = 10


def load_data():
    df = pd.read_csv('./train.csv')
    return df


def shuffle_data(df):
    df = df.sample(frac=1)
    train_data = df[: int(len(df) * PERCENTAGE_OF_TRAINING)]
    test_data = df[int(len(df) * PERCENTAGE_OF_TRAINING):]
    return train_data, test_data


def data_selection_numeric_only(train_data, test_data, params):
    x_train = train_data[params]
    y_train = train_data['SalePrice']

    x_test = test_data[params]
    y_test = test_data['SalePrice']
    return x_train, y_train, x_test, y_test


def data_normalization(df):
    df = (df - df.mean()) / df.std()
    df = np.c_[np.ones(df.shape[0]), df]
    return df


##only for dataframe
def data_nonbasic_expression(degree, df):
    df_set = [df]
    for i in range(2, degree + 1):
        df_temp = df ** i
        df_set.append(df_temp)
    return pd.concat(df_set, axis=1)


def gradient_descent(x, y):
    n = y.size
    rng = np.random.RandomState(128)
    w = rng.rand(x.shape[1])
    all_avg_err = []
    all_w = [w]

    for i in range(ITERATION):
        predication = np.dot(w, x.T)
        error = predication - y
        avg_err = 1 / n * np.dot(error.T, error)
        all_avg_err.append(avg_err)

        w = w - LEARNING_RATE * (2 / n) * np.dot(x.T, error)
        all_w.append(w)
    return all_w, all_avg_err


def validation(w, x, y):
    n = y.size
    final_w = w[-1]
    predication = np.dot(final_w, x.T)
    error = predication - y
    avg_err = 1 / n * np.dot(error.T, error)
    return avg_err


def show_x(x, y, all_w, all_avg_err):
    fig = plt.figure()
    ax = plt.axes()
    plt.title('Sale Price vs Living Area')
    plt.xlabel('Liv')
    plt.ylabel('Sales Price')
    plt.scatter(x[:, 1], y, color='red')
    line, = ax.plot([], [], lw=2)
    annotation = ax.text(-1, 700000, '')
    annotation.set_animated(True)
    plt.close()

    def init():
        line.set_data([], [])
        annotation.set_text('')
        return line, annotation

    def animate(i):
        x = np.linspace(-5, 20, 1000)
        y = all_w[i][1] * x + all_w[i][0]
        line.set_data(x, y)
        annotation.set_text('err = %.2f e10' % (all_avg_err[i] / 100000000))
        return line, annotation

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=0, blit=True)
    anim.save('./animation.gif', writer='imagemagick', fps=30)
    print('saved')


def show_graph(title, xlabel, ylabel, data, label=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(data)):
        plt.plot(data[i], label=label[i])
    plt.legend()
    plt.show()


def find_corr_features(df, param_num):
    numeric_features = df.select_dtypes(include=[np.number])  ##only select numeric feature
    corr = numeric_features.corr()
    corr = corr['SalePrice'].sort_values(ascending=False)[1:param_num + 1]
    return corr.index


def test():
    df = load_data()
    param_select = find_corr_features(df, FEATURES_NUM)
    overall_train_err_set = []
    overall_val_err_set = []
    epoch = 0

    while epoch < 15:
        error_set = []
        validation_error_set = []
        train_data, test_data = shuffle_data(df)
        param_set = []
        for param in param_select:
            param_set.append(param)
            x_train, y_train, x_test, y_test = data_selection_numeric_only(train_data, test_data, param_set)

            x_train = data_nonbasic_expression(2, x_train)
            x_test = data_nonbasic_expression(2, x_test)
            x_train = data_normalization(x_train)
            x_test = data_normalization(x_test)

            w, err = gradient_descent(x_train, y_train)

            error_set.append(err)
            validation_error_set.append(validation(w, x_test, y_test))

        train_err_final = [error[-1] for error in error_set]
        epoch += 1
        overall_train_err_set.append(train_err_final)
        overall_val_err_set.append(validation_error_set)

    overall_train_err = pd.DataFrame(overall_train_err_set)
    overall_val_err = pd.DataFrame(overall_val_err_set)
    train_mean = overall_train_err.mean(axis=0)
    val_mean = overall_val_err.mean(axis=0)
    train_mean.index += 1
    val_mean.index += 1
    # show_learning_curve(train_mean, val_mean)
    show_graph('Learning Curve', 'Complexity', 'Error', [train_mean, val_mean], ['Train', 'Validation'])


if __name__ == "__main__":
    test()
