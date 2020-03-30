'''
Домашняя работа № 3
Проничкин Юрий
'''
from math import sqrt, pi, inf
from random import randint, choice
EPS = 1e-14


class Triangle:
    '''
    Класс треугольник
    '''
    def __init__(self, length1, length2, length3):
        '''
        Инициализация треугольник по трем сторонам -
        length1, length2, length3 - если такого треугольника не существует
        бросается Assert Exception
        '''
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3
        assert length1 + length2 > length3 and \
            length2 + length3 > length1 and \
            length1 + length3 > length2, "No triangle with such sides"

    def perimeter(self):
        '''
        Периметр треугольника
        '''
        return self.length1 + self.length2 + self.length3

    def area(self):
        '''
        Площадь треугольника
        '''
        half_perimetr = self.perimeter() / 2
        return sqrt(half_perimetr *
                    (half_perimetr - self.length1) *
                    (half_perimetr - self.length2) *
                    (half_perimetr - self.length3))

    def isosceles(self):
        '''
        Проверка является ли треугольник равнобедренным
        '''
        return abs(self.length1 - self.length2) < EPS \
            or abs(self.length1 - self.length3) < EPS \
            or abs(self.length2 - self.length3) < EPS

    def equilateral(self):
        '''
        Проверка является ли треугольник равносторонним
        '''
        return abs(self.length1 - self.length2) < EPS and \
            abs(self.length2 - self.length3) < EPS

    def __eq__(self, obj):
        '''
        Перегрузка ==
        Проверка на равенство треугольнику obj
        '''
        return abs(self.length1 - obj.length1) < EPS \
            and abs(self.length2 - obj.length2) < EPS \
            and abs(self.length3 - obj.length3) < EPS \
            or abs(self.length1 - obj.length2) < EPS \
            and abs(self.length2 - obj.length3) < EPS \
            and abs(self.length3 - obj.length1) < EPS \
            or abs(self.length1 - obj.length3) < EPS \
            and abs(self.length2 - obj.length1) < EPS \
            and abs(self.length3 - obj.length2) < EPS \
            or abs(self.length1 - obj.length1) < EPS \
            and abs(self.length2 - obj.length3) < EPS \
            and abs(self.length3 - obj.length2) < EPS \
            or abs(self.length1 - obj.length3) < EPS \
            and abs(self.length2 - obj.length2) < EPS \
            and abs(self.length3 - obj.length1) < EPS \
            or abs(self.length1 - obj.length2) < EPS \
            and abs(self.length2 - obj.length1) < EPS \
            and abs(self.length3 - obj.length3) < EPS


class Circle:
    '''
    Класс окружность
    '''
    def __init__(self, coords, r):
        '''
        Инициализация окружности по
        координатам ее центра - coords
        и радиусу r
        '''
        self.coordinate_x = coords[0]
        self.coordinate_y = coords[1]
        self.radius = r

    def area(self):
        '''
        Площадь Круга
        '''
        return pi * self.radius * self.radius

    def perimeter(self):
        '''
        Периметр окружности
        '''
        return pi * self.radius * 2

    def move_to_point(self, coords):
        '''
        Сдвиг центра окружности в
        точку с координатами coords
        '''
        self.coordinate_x = coords[0]
        self.coordinate_y = coords[1]

    def set_radius(self, r):
        '''
        Изменение радиуса окружности на r
        '''
        self.radius = r

    def __lt__(self, obj):
        '''
        Перегрузка <
        Проверка на то лежит ли окружность
        внтури окружности obj
        '''
        distance = sqrt(
            (self.coordinate_x - obj.coordinate_x) *
            (self.coordinate_x - obj.coordinate_x) +
            (self.coordinate_y - obj.coordinate_y) *
            (self.coordinate_y - obj.coordinate_y)
        )
        return distance + self.radius < obj.radius

    def __gt__(self, obj):
        '''
        Перегрузка >
        Проверка на то лежит ли окружность obj
        внутри окружности
        '''
        distance = sqrt(
            (self.coordinate_x - obj.coordinate_x) *
            (self.coordinate_x - obj.coordinate_x) +
            (self.coordinate_y - obj.coordinate_y) *
            (self.coordinate_y - obj.coordinate_y)
        )
        return distance + obj.radius < self.radius

    def __eq__(self, obj):
        '''
        Перегрузка ==
        Провеяет совпадает ли окружность
        с окружностью obj
        '''
        return abs(self.coordinate_x - obj.coordinate_x) < EPS \
            and abs(self.coordinate_y - obj.coordinate_y) < EPS \
            and abs(self.radius - obj.radius) < EPS

    def __and__(self, obj):
        '''
        Перегрузка &
        Проверка на то пересекает ли окружность
        окружность obj
        '''
        distance = sqrt(
            (self.coordinate_x - obj.coordinate_x) *
            (self.coordinate_x - obj.coordinate_x) +
            (self.coordinate_y - obj.coordinate_y) *
            (self.coordinate_y - obj.coordinate_y)
        )
        return self.radius + obj.radius >= distance \
            and not self > obj and not self < obj


class Item:
    '''
    Класс Товар
    '''
    def __init__(self, Name, Price, Category):
        '''
        Инициализация класса
        Name - название
        Price - цена
        Category - категория товара
        '''
        self.Name = Name
        self.Price = Price
        self.Category = Category


class Shop:
    '''
    Класс Магазин
    '''
    def __init__(self):
        '''
        Инициализация класса -
        создание пустого списка товаров
        в этом магазине
        '''
        self.list_of_items = []

    def add(self, item):
        '''
        Добавление товара Item в список
        товаров магазина
        '''
        self.list_of_items.append(item)

    def count_category(self, category):
        '''
        Возвращает количество товаров
        категории category
        '''
        count = 0
        for item in self.list_of_items:
            if category == item.Category:
                count += 1
        return count

    def find_items(self,
                   price_bounds=[None, None],
                   categories=None):
        '''
        Поиск товаров по заданным фильтрам -
        диапазон цен список price_bounds, при передачи None
        проверяется весь диапозон цен
        категории - список categories,
        при передаче None выдаются все категории
        '''
        items_to_find = []
        if price_bounds[0] is None:
            price_bounds[0] = -inf
        if price_bounds[1] is None:
            price_bounds[1] = inf
        if categories is None:
            for item in self.list_of_items:
                if not item.Price > price_bounds[1] \
                        and not item.Price < price_bounds[0]:
                    items_to_find.append(item)
        else:
            for item in self.list_of_items:
                if not item.Price > price_bounds[1] \
                and not item.Price < price_bounds[0] \
                and item.Category in categories:
                    items_to_find.append(item)
        return items_to_find

    def get_full_price(self, categories=None):
        '''
        Сумма всех товаров категорий списка categories
        при передаче None сумма цен всех товаров
        '''
        if categories is None:
            return sum(x.Price for x in self.list_of_items)
        else:
            return sum(x.Price for x in self.list_of_items
                       if x.Category in categories)

    def __contains__(self, item):
        '''
        Перегрузка in
        Проверяет есть ли товар с данным названием
        среди товаров в магазине
        '''
        return item.Name in [x.Name for x in self.list_of_items]

    def is_available(self, item):
        '''
        Проверяет есть ли товар item в магазине
        '''
        return item in self


def log_hit_decorator(funk):
    '''
    Декоратор для функции hit -
    логгирует удар
    '''
    def wrapper(*args):
        result = funk(*args)
        print(f'Робот {args[0].name}' +
              f' ударил робота {args[1].name}' +
              f' и нанес {result} урона здоровью')
        print(f'Здоровье робота {args[0].name}:' +
              f' {args[0].health},' +
              f' здоровье робота {args[1].name}:' +
              f' {args[1].health}')
        print()
        return result
    return wrapper


class Robot:
    '''
    Класс робот
    count - количество роботов
    в данный момент
    '''
    count = 0

    @classmethod
    def update_cls(cls):
        '''
        Метод класса,
        Увеличивает count на единицу
        '''
        cls.count += 1

    def __init__(self, name=None, power=10, health=100):
        '''
        Инициализация робота
        name - название, если None -
        будет присвоено 'Robot count'
        power - сила
        health - здоровье
        '''
        if name is None:
            self.name = f'Robot {self.count}'
        else:
            self.name = name
        self.power = power
        self.health = health
        self.update_cls()

    @log_hit_decorator
    def hit(self, robot):
        '''
        Удар робота по роботу robot
        '''
        damage = randint(1, self.power)
        robot.health -= damage
        return damage

    def fightWith(self, robot):
        '''
        Поединок роботов
        '''
        while True:
            self.hit(robot)
            if robot.health <= 0:
                print(f'Робот {self.name} побеждает!')
                return True
            robot.hit(self)
            if self.health <= 0:
                print(f'Робот {robot.name} побеждает!')
                return False


class Node:
    '''
    Класс - узел графа с не более
    чем 4 соседями
    '''
    def __init__(self, nodetype):
        '''
        Инициализация
        nodetype - тип узла
        '''
        self.nodetype = nodetype
        self.top = None
        self.left = None
        self.right = None
        self.bottom = None
        self.dist = 0
        self.pred = None
        self.isshortway = False

    def __str__(self):
        '''
        Перегрузка str
        возвращает nodetype
        '''
        return str(self.nodetype)


def check_and_update(curr, node, queue):
    '''
    Дополнительная функция для удобства -
    проверка нужно ли добавить вершину node
    дочернюю для вершины  curr в очередь queue
    '''
    if node is not None:
        if str(node) != 'X' and node not in queue:
            node.dist = curr.dist + 1
            node.pred = curr
            queue.append(node)


class Board:
    '''
    Класс лабиринт
    '''
    def __init__(self):
        '''
        Инициализация лабиринта 10 на 110
        finish - конечный узел
        start - начальный узел
        '''
        self.finish = None
        self.start = None
        self.lefttop = None

    def read(self, string):
        '''
        Cчитывание лабиринта 10 на 10
        из строки string
        '''
        string = string.split('\n')
        self.lefttop = Node(string[0][0])
        if string[0][0] == 'S':
            self.start = self.lefttop
        curr_node = self.lefttop
        curr_bottom = Node(string[1][0])
        if string[1][0] == 'S':
            self.start = curr_bottom
        curr_right = Node(string[0][1])
        if string[0][1] == 'S':
            self.start = curr_right
        curr_node.right = curr_right
        curr_right.left = curr_node
        curr_node.bottom = curr_bottom
        curr_bottom.top = curr_node
        next_line_start = curr_bottom
        for j in range(1, 9):
            curr_node = curr_node.right
            curr_bottom.right = Node(string[1][j])
            if string[1][j] == 'S':
                self.start = curr_bottom.right
            curr_bottom.right.left = curr_bottom
            curr_bottom = curr_bottom.right
            curr_node.bottom = curr_bottom
            curr_bottom.top = curr_node
            curr_right = Node(string[0][j + 1])
            if string[0][j + 1] == 'S':
                self.start = curr_right
            curr_right.left = curr_node
            curr_node.right = curr_right
        curr_node = curr_node.right
        curr_bottom.right = Node(string[1][9])
        if string[0][9] == 'S':
            elf.start = curr_bottom.right
        curr_bottom.right.left = curr_bottom
        curr_bottom = curr_bottom.right
        curr_node.bottom = curr_bottom
        curr_bottom.top = curr_node
        for i in range(1, 9):
            curr_node = next_line_start
            curr_bottom = Node(string[i + 1][0])
            if string[i + 1][0] == 'S':
                self.start = curr_bottom
            curr_node.bottom = curr_bottom
            curr_bottom.top = curr_node
            next_line_start = curr_bottom
            for j in range(1, 9):
                curr_node = curr_node.right
                curr_bottom.right = Node(string[i + 1][j])
                if string[i + 1][j] == 'S':
                    self.start = curr_bottom.right
                curr_bottom.right.left = curr_bottom
                curr_bottom = curr_bottom.right
                curr_node.bottom = curr_bottom
                curr_bottom.top = curr_node
            curr_node = curr_node.right
            curr_bottom.right = Node(string[i + 1][9])
            if string[i + 1][9] == 'S':
                self.start = curr_bottom.right
            curr_bottom.right.left = curr_bottom
            curr_bottom = curr_bottom.right
            curr_node.bottom = curr_bottom
            curr_bottom.top = curr_node

    def print(self):
        '''
        Печать результата поиска пути от
        start до finish
        '''
        if self.finish is None:
            print('No way')
        else:
            print(self.finish.dist)
            print()
            RED = '\033[31m'
            END = '\033[0m'
            curr = self.lefttop
            next_line = curr.bottom
            string = ''
            for i in range(10):
                for j in range(10):
                    if curr.isshortway:
                        string += RED
                        string += str(curr)
                        string += END
                    else:
                        string += str(curr)
                    curr = curr.right
                curr = next_line
                if next_line is not None:
                    next_line = curr.bottom
                string += '\n'
            print(string)

    def solve(self):
        '''
        Поиск кратчайшего пути от start до finfish
        '''
        curr = self.start
        curr.dist += 1
        queue = [curr]
        for curr in queue:
            if str(curr) == 'F':
                self.finish = curr
                break
            check_and_update(curr, curr.top, queue)
            check_and_update(curr, curr.left, queue)
            check_and_update(curr, curr.bottom, queue)
            check_and_update(curr, curr.right, queue)
        curr = self.finish
        while curr is not None:
            curr.isshortway = True
            curr = curr.pred

    def generate(self):
        '''
        Отладочная функция - генерация случайного лабиринта
        '''
        string = ''
        for i in range(10):
            for j in range(10):
                string += choice(['X', 'O'])
        start = randint(0, 99)
        if start == 0:
            finish = randint(1, 99)
        elif start == 99:
            finish = randint(0, 98)
        else:
            if randint(0, 1) == 0:
                finish = randint(0, start - 1)
            else:
                finish = randint(start + 1, 99)
        string = string[:start] + 'S' + string[start + 1:]
        string = string[:finish] + 'F' + string[finish + 1:]
        for i in range(1, 10):
            string = string[:i * 10 + i - 1] + '\n' + string[i * 10 + i - 1:]
        self.read(string)
