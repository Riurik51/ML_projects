import time
import numpy as np
from scipy.special import expit


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой,
        необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода
        по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        if loss_function == 'binary_logistic':
            self.Loss = BinaryLogistic(**kwargs)
        else:
            raise NotImplementedError('Loss function is not implemented')

    def fit(self, X, y, w_0=None, trace=False, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history,
        содержащий информацию о поведении метода.
        Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени
        между двумя итерациями метода history['func']: list of floats,
        содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        w = np.copy(w_0)
        self.w = w
        pred_loss = self.Loss.func(X, y, w)
        if trace is True:
            history = {}
            history['time'] = []
            history['func'] = []
            time_pred = time.time()
            history['time'].append(0)
            history['func'].append(pred_loss)
            if X_test is not None and y_test is not None:
                history['test_acc'] = []
                history['train_acc'] = []
                history['test_acc'].append(float((self.predict(X_test) == y_test).sum()) / y_test.shape[0])
                history['train_acc'].append(float((self.predict(X) == y).sum()) / y.shape[0])
        for i in range(1, self.max_iter + 1):
            w_next = w - self.Loss.grad(X, y, w) * ((self.step_alpha) /
                                                    (i ** self.step_beta))
            new_loss = self.Loss.func(X, y, w_next)
            if np.abs(new_loss - pred_loss) < self.tolerance:
                self.w = w_next
                if trace is True:
                    time_next = time.time()
                    history['time'].append(time_next - time_pred)
                    history['func'].append(new_loss)
                    if X_test is not None and y_test is not None:
                        history['test_acc'].append(float((self.predict(X_test) == y_test).sum()) / y_test.shape[0])
                        history['train_acc'].append(float((self.predict(X) == y).sum()) / y.shape[0])
                    return history
                break
            w = w_next
            self.w = w
            if trace is True:
                time_next = time.time()
                history['time'].append(time_next - time_pred)
                time_pred = time_next
                history['func'].append(new_loss)
                if X_test is not None and y_test is not None:
                    history['test_acc'].append(float((self.predict(X_test) == y_test).sum()) / y_test.shape[0])
                    history['train_acc'].append(float((self.predict(X) == y).sum()) / y.shape[0])
            pred_loss = new_loss
        if trace is True:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        return np.sign(X.dot(self.w))

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array,
        [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        answer = np.zeros((X.shape[0], 2), float)
        answer[:, 1] = expit(X.dot(self.w))
        answer[:, 0] = -answer[:, 1] + 1
        return answer

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        return self.Loss.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        return self.Loss.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить
        оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних
        значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать
        np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов
        на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.batch_size = batch_size
        if loss_function == 'binary_logistic':
            self.Loss = BinaryLogistic(**kwargs)
        else:
            raise NotImplementedError('Loss function is not implemented')

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, X_test=None, y_test=None):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        w = np.copy(w_0)
        w_next = w
        pred_loss = self.Loss.func(X, y, w)
        epoch_num_pred = 0.0
        epoch_num = epoch_num_pred
        self.w = w_next
        if trace is True:
            history = {}
            history['epoch_num'] = []
            history['time'] = []
            history['func'] = []
            history['weights_diff'] = []
            time_pred = time.time()
            history['time'].append(0)
            history['func'].append(pred_loss)
            history['epoch_num'].append(epoch_num_pred)
            history['weights_diff'].append(0.0)
            if X_test is not None and y_test is not None:
                history['test_acc'] = []
                history['train_acc'] = []
                history['test_acc'].append(float((self.predict(X_test) == y_test).sum()) / y_test.shape[0])
                history['train_acc'].append(float((self.predict(X) == y).sum()) / y.shape[0])
        for i in range(1, self.max_iter + 1):
            diff = float(self.batch_size) / X.shape[0]
            permutation = np.random.permutation(X.shape[0])
            w_pred = w
            for j in range(self.batch_size, X.shape[0], self.batch_size):
                w_next = w - self.Loss.grad(X[j - self.batch_size:j], y[j - self.batch_size:j], w) * \
                        ((self.step_alpha) / (i ** self.step_beta))
                epoch_num += diff
                w = w_next
                if epoch_num - epoch_num_pred > log_freq:
                    break
            epoch_num_pred = epoch_num
            new_loss = self.Loss.func(X, y, w_next)
            self.w = w
            if abs(new_loss - pred_loss) < self.tolerance:
                if trace is True:
                    time_new = time.time()
                    history['time'].append(time_new - time_pred)
                    history['func'].append(new_loss)
                    history['epoch_num'].append(epoch_num)
                    history['weights_diff'].append(np.linalg.norm(w - w_pred))
                    if X_test is not None and y_test is not None:
                        history['test_acc'].append(float((self.predict(X_test) == y_test).sum()) / y_test.shape[0])
                        history['train_acc'].append(float((self.predict(X) == y).sum()) / y.shape[0])
                    return history
                break
            pred_loss = new_loss
            if trace is True:
                    time_new = time.time()
                    history['time'].append(time_new - time_pred)
                    time_pred = time_new
                    history['func'].append(new_loss)
                    history['epoch_num'].append(epoch_num)
                    history['weights_diff'].append(np.linalg.norm(w - w_pred))
                    if X_test is not None and y_test is not None:
                        history['test_acc'].append(float((self.predict(X_test) == y_test).sum()) / y_test.shape[0])
                        history['train_acc'].append(float((self.predict(X) == y).sum()) / y.shape[0])
        self.w = w
        if trace is True:
            return history
