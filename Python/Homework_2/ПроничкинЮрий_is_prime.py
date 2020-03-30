"""
Модуль содержит функцию is_prime,
которая проверяет число на делимость
используется библиотека math
"""
import math


def is_prime(number, divider=2):
    """
	Функция рекурсивно проверяет,
	делится ли число number
	на числа от divider до корня из number

	Вход: number - число для проверки
	divider - начало проверяемых делителей
	по умолчанию 2

	Выход: True - число не делится на все
	числа от divider до корня
	(по умолчанию проверка на простоту)
	False - в обратном случаее
    """
    if divider > math.sqrt(number):
        return True
    if number % divider == 0:
        return False
    return is_prime(number, divider + 1)
