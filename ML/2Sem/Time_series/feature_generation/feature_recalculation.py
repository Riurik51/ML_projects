import numpy as np

class RecursivePredictor():

    def __init__(self, header, model, label_codes_dict, sales_start_day, features_by_predict):
        """

        Parameters
        ----------
        header : list
            Список заголовков колонок
        model : Optional
            Модель для предсказания с методом predict
        label_codes_dict : dict
            Словарь кодирования категориальных признаков: {'column_name': {'label': code}}
        sales_start_day : int
            День начала продаж - нужен для подсчета экспоненциального среднего
        features_by_predict: int array
            Индексы признаков по которым предсказывает модель
        """     
        self.header = header  
        self.model = model
        self.label_codes_dict = label_codes_dict
        self.sales_start_day = sales_start_day
        self.features_by_predict = features_by_predict

    def get_recursive_prediction(self, X, day_range=28):
        """
        
        Итеративное получение предсказаний по дням

        Parameters
        ----------
        X : numpy ndarray
            Матрица признаков (кол-во элементов, кол-во признаков)
        day_range : int, optional
            Количество дней, на которое надо делать прогноз, by default 28

        Returns
        -------
        numpy ndarray
            Матрица предсказаний (кол-во элементов, )
            
        """
        predictions = np.zeros(X.shape[0])
        starting_day = X[:, self.header.index('day_as_int')].min()
        starting_idx = np.array(np.nonzero(X[:, self.header.index('day_as_int')] == starting_day)[0], dtype=int)
        for col in self.header:
            if 'shift' in col:
                shift_value = int(col.split('_')[1])
                if shift_value < 28:
                    X[:, self.header.index(col)].reshape(-1, 28)[:, shift_value:] = np.nan
            if 'mean' in col:
                shift_value = int(col.split('_')[1])
                if shift_value < 28:
                    X[:, self.header.index(col)].reshape(-1, 28)[:, shift_value:] = np.nan
        for day in range(day_range):
            dict_of_shifts = {}
            for prev_day in range(day):
                dict_of_shifts[prev_day + 1] = (self.header.index('demand'), starting_idx + prev_day)

            for col in self.header:
                if 'shift' in col:
                    shift_value = int(col.split('_')[1])
                    for prev_day in range(day + 1):
                        if shift_value + prev_day not in dict_of_shifts.keys():
                            dict_of_shifts[shift_value + prev_day] = (self.header.index(col), starting_idx + day - prev_day)
                    if np.isnan(X[:, self.header.index(col)][starting_idx + day]).sum() > 0:
                        if day >= shift_value:
                            X[:, self.header.index(col)][starting_idx + day] = X[:, self.header.index('demand')][starting_idx + day - shift_value]
                        else:
                            raise NameError('Shift cannot be calculated in day', day, shift_value)

            for col in self.header:
                if 'mean' in col:
                    if np.isnan(X[:, self.header.index(col)][starting_idx + day]).sum() > 0:
                        split = col.split('_')
                        shift_value = int(split[1])
                        if len(split[2].split('.')) < 2:
                            window_size = int(split[2])

                            if day == 0:
                                raise NameError('Mean cannot be calculated')
                            if len(split) > 3:
                                period_size = int(split[4])
                            else:
                                period_size = 1

                            if day < period_size:
                                prev_mean_idx = -1
                                for col_2 in header:
                                    if f'mean_{shift_value + period_size}_{window_size}_{period_size}' in col_2:
                                      	prev_mean_idx = self.header.index(col_2)
                                if prev_mean_idx < 0:
                                    raise NameError('Mean cannot be calculated, not found previos period in features')
                                prev_mean = X[:, prev_mean_idx][starting_idx + day]
                            else:
                                prev_mean = X[:, self.header.index(col)][starting_idx + day - period_size]

                            if window_size * period_size + shift_value in dict_of_shifts.keys():
                                prev_y = X[:, dict_of_shifts[window_size * period_size + shift_value][0]][dict_of_shifts[window_size * period_size + shift_value][1]]
                            else:
                                raise NameError('Mean cannot be calculated, not found featue shift', window_size * period_size + shift_value)
                            if shift_value in dict_of_shifts.keys():
                                new_y = X[:, dict_of_shifts[shift_value][0]][dict_of_shifts[shift_value][1]]
                            else:
                                raise NameError('Mean cannot be calculated, not found featue shift', shift_value)
                                
                            X[:, self.header.index(col)][starting_idx + day] = (prev_mean * window_size - prev_y + new_y) / window_size
                        else:
                            shift_value = int(split[1])
                            beta = float(split[2])
                            beta_sum = sum(beta ** (np.arange(int((starting_day + day - shift_value - self.sales_start_day) / period_size)))) * beta
                            if len(split) < 4:
                                period_size = 1
                            else:
                                period_size = int(split[4])
                            if day < period_size:
                                    prev_exp_idx = -1
                                    for col_2 in header:
                                        if f'mean_{shift_value + period_size}_{beta}_{period_size}' in col_2:
                                            prev_mean_idx = self.header.index(col_2)
                                    if prev_mean_idx < 0:
                                        raise NameError('Mean cannot be calculated, not found previos period in features')
                                    prev_exp = X[:, prev_mean_idx][starting_idx + day]
                            else:
                                    prev_exp = X[:, self.header.index(col)][starting_idx + day - period_size]
                            if shift_value in dict_of_shifts.keys():
                                new_y = X[:, dict_of_shifts[shift_value][0]][dict_of_shifts[shift_value][1]]
                            X[:, self.header.index(col)][starting_idx + day] = (prev_exp * beta_sum * beta + new_y) / (beta_sum * beta + 1)
            predictions[starting_idx + day] = self.model.predict(X[starting_idx + day][:, self.features_by_predict])
            X[starting_idx + day, self.header.index('demand')] = predictions[starting_idx + day]                    
        return predictions

    def fill_final_submission(self, sample_submission, X, predictions=None, day_range=28):
        """
        
        Заполнение шаблона посылки

        Parameters
        ----------
        sample_submission : DataFrame
            шаблон посылки для соревнования
        X : numpy ndarray
            Матрица признаков (кол-во элементов, кол-во признаков)
        predictions : numpy ndarray, optional
            Массив предсказаний, если он получен заранее, by default None
        day_range : int, optional
            Количество дней, на которое надо делать прогноз, by default 28
        """        
        if predictions is not None:
            predictions = self.get_recursive_prediction(X)
        
        index_to_item_id = {value: key for key, value in self.label_codes_dict['item_id'].items()}
        X_test_item_ids = [index_to_item_id[elem] for elem in X[:, self.header.index('item_id')]]
        index_to_store_id = {value: key for key, value in self.label_codes_dict['store_id'].items()}
        X_test_store_ids = [index_to_store_id[elem] for elem in X[:, self.header.index('store_id')]]
        X_test_submission_index = [
            f'{item_id}_{store_id}_validation'
            for item_id, store_id in zip(X_test_item_ids, X_test_store_ids)
        ]