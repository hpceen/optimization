import numpy
import matplotlib.pyplot as plt

hjroot = []
nmroot = []


class HookeJeeves:
    @staticmethod
    def __exploring_search(basis_point: numpy.ndarray, step: numpy.ndarray, target_function) -> numpy.ndarray:
        """
        Исследующий поиск.
        Поиск точек по каждой оси вокруг базисной точки.
        Возвращает точку с минимальным значением целевой функции.
        """
        for i in step:
            if target_function(basis_point + i) < target_function(basis_point):
                basis_point = basis_point + i
            elif target_function(basis_point - i) < target_function(basis_point):
                basis_point = basis_point - i
        return basis_point

    @staticmethod
    def __pattern_search(basis_point, minimum, target_function, alpha: float) -> numpy.ndarray:
        """
        Поиск по образцу, пока целевая функция уменьшается.
        Возвращает точку с минимально возможным значением целевой функции.
        """
        d = (minimum - basis_point) * alpha
        next_point = minimum + d
        while target_function(next_point) < target_function(minimum):
            next_point, minimum = next_point + d, next_point
        return minimum

    @staticmethod
    def __stop_criteria(delta_x: numpy.ndarray, epsilon: float) -> bool:
        """
        Критерий останова.
        True когда длина каждого шага (по каждой оси) стала меньше epsilon
        """
        for i in delta_x:
            if numpy.linalg.norm(i) > epsilon:
                return False
        return True

    @staticmethod
    def optimization(basis_point: numpy.ndarray, delta_x: numpy.ndarray, target_function, alpha: float,
                     epsilon: float, decrement_parameter: float):
        """
        Главная функция оптимизации.
        Возвращает координаты точки и значение целевой функции в этой точке.
        """

        while not HookeJeeves.__stop_criteria(delta_x, epsilon):
            minimum = HookeJeeves.__exploring_search(basis_point, delta_x, target_function)
            # print(f"minimum= {minimum}\ntarget_function(minimum) = {target_function(minimum)}")
            # print(f"basis_point {basis_point}\ntarget_function(basis_point) = {target_function(basis_point)}")
            if numpy.array_equal(minimum, basis_point):
                delta_x *= decrement_parameter
                continue
            # Добавление точки в список точек (путь) (для графиков)
            hjroot.append(minimum)
            temp_basis_point = HookeJeeves.__pattern_search(basis_point, minimum, target_function, alpha)
            if numpy.array_equal(temp_basis_point, basis_point):
                delta_x *= decrement_parameter
                continue
            # Добавление точки в список точек (путь) (для графиков)
            hjroot.append(temp_basis_point)
            basis_point = temp_basis_point
        return basis_point, target_function(basis_point)


class NelderMead:

    @staticmethod
    def __find_points(point: numpy.ndarray, deltax: numpy.ndarray, point_coefficients: list = None) -> numpy.ndarray:
        """
        Нахождение точек.
        Выполнено как шаг deltax по оси относительно начальной точки.
        """
        return numpy.array([point, *[point + x for x in deltax]])

    @staticmethod
    def __points_sorting(points: numpy.ndarray, target_function) -> numpy.ndarray:
        """Сортировка точек"""
        return numpy.array(sorted(points, key=target_function))

    @staticmethod
    def __centroid(points: numpy.ndarray, n: int) -> numpy.ndarray:
        """Нахождение центроиды (центра масс)"""
        return 1 / n * sum(points[0:n])

    @staticmethod
    def __reflection(x_high: numpy.ndarray, centroid: numpy.ndarray, coefficients: list) -> numpy.ndarray:
        """Нахождение точки отражения"""
        return centroid + coefficients[0] * (centroid - x_high)

    @staticmethod
    def __expansion(reflection: numpy.ndarray, centroid: numpy.ndarray, coefficients: list) -> numpy.ndarray:
        """Нахождение точки расширения"""
        return centroid + coefficients[1] * (reflection - centroid)

    @staticmethod
    def __stop_criteria(points: numpy.ndarray, x_centroid, target_function, n) -> float:
        """
        Критерий останова.
        """
        return numpy.sqrt(sum([((target_function(x) - target_function(x_centroid)) ** 2) / n for x in points]))

    @staticmethod
    def optimization(point: numpy.ndarray, delta_x: numpy.ndarray, target_function, coefficients: list, epsilon: float,
                     point_coefficients: list = None):
        n = len(point)
        points = NelderMead.__find_points(point, delta_x, point_coefficients)
        while NelderMead.__stop_criteria(points, NelderMead.__centroid(points, n), target_function, n) > epsilon:
            # Добавление точки в список точек (путь) (для графиков)
            nmroot.append(points)
            # Сортировка
            points = NelderMead.__points_sorting(points, target_function)

            # Отрисовка точек на каждой итерации
            # print(points[0], points[1], points[2], end="\n")

            # Поиск центроиды
            x_centroid = NelderMead.__centroid(points, n)

            # Отражение
            x_reflection = NelderMead.__reflection(points[n], x_centroid, coefficients)
            if target_function(points[0]) <= target_function(x_reflection) < target_function(points[n - 1]):
                points[n] = x_reflection
                continue

            # Растяжение
            if target_function(x_reflection) < target_function(points[0]):
                x_expansion = NelderMead.__expansion(x_reflection, x_centroid, coefficients)
                if target_function(x_expansion) < target_function(x_reflection):
                    points[n] = x_expansion
                else:
                    points[n] = x_reflection
                continue

            # Сужение
            if target_function(x_reflection) < target_function(points[n]):
                x_contraction = x_centroid + coefficients[2] * (x_reflection - x_centroid)
                if target_function(x_contraction) < target_function(x_reflection):
                    points[n] = x_contraction
                    continue

            if target_function(x_reflection) >= target_function(points[n]):
                x_contraction = x_centroid + coefficients[2] * (points[n] - x_centroid)
                if target_function(x_contraction) < target_function(points[n]):
                    points[n] = x_contraction
                    continue

            # Сжатие
            for i in range(1, len(points)):
                points[i] = points[0] + coefficients[3] * (points[i] - points[0])

        return points[0], target_function(points[0])


def function(point: numpy.ndarray) -> float:
    """Целевая функция"""
    # 0 at (4, -4)
    # return (point[0] - 4) ** 2 + (point[1] + 4) ** 2

    # 0 at (3, -3)
    # return (point[0] - 1) ** 2 + (point[1] + 1) ** 2

    # 0 at (1, 1)
    return 100 * (point[0] ** 2 - point[1]) ** 2 + (point[0] - 1) ** 2

    # 10.4265 at (0.731404, -0.365702)
    # return (point[0] + point[1]) ** 2 + (numpy.sin(point[0] + 2)) ** 2 + point[1] ** 2 + 10

    # 0 at (4, 2)
    # return (4 - point[0]) ** 2 + (2 - point[1]) ** 2


if __name__ == "__main__":
    """Графики"""
    x = numpy.arange(-20, 20, 0.5)
    y = numpy.arange(-20, 20, 0.5)
    X, Y = numpy.meshgrid(x, y)
    plt.contourf(X, Y, function(numpy.array([X, Y])), cmap='Reds', levels=20, alpha=1)

    """Метод Хука-Дживса"""
    # Приращения по каждой оси
    deltax = numpy.array([[1., 0.], [0., 1.]])
    # Начальная точка
    x0 = numpy.array([5., 12.])
    # alpha - коэффициент шага в поиске по образцу
    # epsilon - точность
    result = HookeJeeves.optimization(basis_point=x0, delta_x=deltax, target_function=function,
                                      alpha=1, epsilon=0.00001, decrement_parameter=0.5)
    print(f"Метод Хука-Дживса\t{result}")

    # метод Хука-Дживса отрисовка точек
    plt.scatter(x0[0], x0[1], color='blue', s=5)
    plt.plot([i[0] for i in hjroot], [i[1] for i in hjroot], color='black')

    """Метод Нелдера-Мида"""
    # Приращения по каждой оси
    dx = numpy.array([[1., 0.], [0., 1.]])
    # Начальная точка
    x0 = numpy.array([5., 12.])
    result = NelderMead.optimization(point=x0, delta_x=dx, target_function=function, coefficients=[1, 2, 0.5, 0.5],
                                     epsilon=0.001)
    print(f"Метод Нелдера-Мида\t{result}")

    # метод Нелдера-Мида отрисовка точек
    plt.scatter(x0[0], x0[1], color='blue', s=5)
    for i in nmroot:
        plt.plot([j[0] for j in i], [j[1] for j in i], color='green')

    plt.show()
