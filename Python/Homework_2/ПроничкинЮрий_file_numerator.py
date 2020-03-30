"""
Модуль содержит функцию file_numerator
для нумерации фаулов в директории,
используется библиотека os
"""
import os


def file_numerator(path):
    """
    Функция рекурсивно обходит все директории в директории path,
    в каждой последовательно обходит все файлы
    и печатает их номер в порядке открытия
    при отсутствии такой директории происходит FileNotFoundError
    """
    iterator = 0
    for next_path in os.listdir(path):
        full_path = os.path.join(path, next_path)
        if os.path.isdir(full_path):
            file_numerator(full_path)
        if os.path.isfile(full_path):
            print("%d, %s" % (iterator, next_path))
            iterator += 1
