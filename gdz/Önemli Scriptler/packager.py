def packager(train_scaled, val_scaled, test_scaled, window_size=24):
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    for i in range(window_size, len(train_scaled)):
        X_train.append(train_scaled[i-window_size:i, 0])
        y_train.append(train_scaled[i, 0])

    for i in range(window_size, len(val_scaled)):
        X_val.append(val_scaled[i-window_size:i, 0])
        y_val.append(val_scaled[i, 0])

    for i in range(window_size, len(test_scaled)):
        X_test.append(test_scaled[i-window_size:i, 0])
        y_test.append(test_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    X = np.concatenate((X_train, X_val, X_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)

    return X_train, y_train, X_val, y_val, X_test, y_test, X, y