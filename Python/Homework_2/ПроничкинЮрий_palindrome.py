"""
Модуль содержит фукнцию, проверяющую является ли
строка полиндромом
"""


def palindrome(string):
    """
    Функция проверяет является ли строка полиндромом
    Вход: string - строка
    Выход: True - является False - нет
    """
    if len(string) < 2:
        return True
    if string[0] != string[-1]:
        return False
    return palindrome(string[1:-1])
