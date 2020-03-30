"""
Модуль содержит функцию merge_sort,
реализующую сортировку слиянием,
а так же служебную функцию merge
"""


def merge(arr1, arr2):
    """
    Функция присоединяет отсортированные
    массивы arr1 arr2 , сохраняя свойство
    отсортированности
    Вход : arr1, arr2 - отсортированные массивы
    массивы удалятся в процессе слияния
    Выход: отсортированный массив arr
    """
    arr = []
    while len(arr1) != 0 and len(arr2) != 0:
        if arr1[0] < arr2[0]:
            arr.append(arr1.pop(0))
        else:
            arr.append(arr2.pop(0))
    if len(arr1) != 0:
        arr += arr1
    else:
        arr += arr2
    return arr


def merge_sort(arr):
    """
    Функция релизует рекурсивно сортировку слиянием
    Вход: arr - сортируемый массив
    Выход: arr - отсортированный массив
    """
    if len(arr) <= 1:
        return arr
    else:
        k = int(len(arr) / 2)
        return merge(merge_sort(arr[:k]), merge_sort(arr[k:]))


