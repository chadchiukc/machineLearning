import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LEARNING_RATE = 0.05
ITERATION = 500
EPSILON = 0.01
PERCENTAGE_OF_TRAINING = 0.9


def load_data():
    df = pd.read_csv('./train.csv')
    # test_data = pd.read_csv('./test.csv')
    # print(df.info())
    df = df.sample(frac=1)
    train_data = df[: int(len(df) * PERCENTAGE_OF_TRAINING)]
    test_data = df[int(len(df) * PERCENTAGE_OF_TRAINING):]

    # numeric_features = df.select_dtypes(include=[np.number]) ##only select numeric feature
    # corr = numeric_features.corr()
    # print(corr['SalePrice'].sort_values(ascending=False))
    return train_data, test_data


def data_process_linear_numeric_only(train_data, test_data, params):
    x_train = train_data[params]
    y_train = train_data['SalePrice']

    x_test = test_data[params]
    # x_test = test_data['GrLivArea']
    y_test = test_data['SalePrice']

    # Normalization with bias added
    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]

    return x_train, y_train, x_test, y_test


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
        print(avg_err)
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


def show_err(all_avg_err):
    plt.title('Error')
    plt.xlabel('No. of iterations')
    plt.ylabel('Error')
    for err in all_avg_err:
        plt.plot(err)
    plt.show()


def test():
    param_select = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
    overall_train_err_set = []
    overall_val_err_set = []
    epoch = 0
    while epoch < 15:
        error_set = []
        validation_error_set = []
        train_data, test_data = load_data()
        param_set = []
        for param in param_select:
            param_set.append(param)
            x_train, y_train, x_test, y_test = data_process_linear_numeric_only(train_data, test_data, param_set)
            w, err = gradient_descent(x_train, y_train)
            error_set.append(err)
            validation_error_set.append(validation(w, x_test, y_test))
        # show_err(error_set)
        train_err_final = [error[-1] for error in error_set]
        epoch += 1
        overall_train_err_set.append(train_err_final)
        overall_val_err_set.append(validation_error_set)
    overall_train_err = pd.DataFrame(overall_train_err_set)
    overall_val_err = pd.DataFrame(overall_val_err_set)
    print(overall_train_err.mean(axis=0))
    print(overall_val_err.mean(axis=0))


    # w, err = gradient_descent(x_train, y_train)
    # print(err[-1])

    # show_x(x, y, w, err)


if __name__ == "__main__":
    test()
