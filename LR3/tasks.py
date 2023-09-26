import cv2
import numpy as np

def main():
    #задание 4- применение размытия Гаусса

    img = cv2.imread("test4.jpg", cv2.IMREAD_GRAYSCALE)
    #стандартное отклонение
    standard_deviation = 100
    # standard_deviation1 = 200
    # standard_deviation2 = 50

    #размер ядра(матрицы)
    kernel_size = 5
    # kernel_size1 = 3
    # kernel_size2 = 7

    #размытие Гаусса (усредняет значения пикселей в соответствии с их расположением относительно центрального пикселя и весами)
    imgBlur_1 = AnotherGaussianBlur(img, kernel_size, standard_deviation)
    # imgBlur_2 = AnotherGaussianBlur(img, kernel_size1, standard_deviation1)
    # imgBlur_3 = AnotherGaussianBlur(img, kernel_size2, standard_deviation2)

    cv2.imshow('Original_image', img)
    cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur_1)
    # cv2.imshow(str(kernel_size1) + 'x' + str(kernel_size1) + ' and deviation ' + str(standard_deviation1), imgBlur_2)
    # cv2.imshow(str(kernel_size2) + 'x' + str(kernel_size2) + ' and deviation ' + str(standard_deviation2), imgBlur_3)



 # Задание 5 - Реализация размытие Гаусса встроенным методом OpenCV
    imgBlur_CV2 = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), standard_deviation)
    cv2.imshow('Blur_by_CV2', imgBlur_CV2)
    cv2.waitKey(0)



def AnotherGaussianBlur(img, kernel_size, standard_deviation):
    #задание 1- построение матрицы гаусса

    #построение начальной матрицы ядра свертки из единиц.
    kernel = np.ones((kernel_size, kernel_size))
    a = b = (kernel_size+1) // 2 #координаты центрального элемента матрицы

    # построение матрицы свёртки
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b) # вычисление функции Гаусса для каждого элемента матрицы
    print("Матрица ядра свертки:")
    print(kernel)

    # Задание 2 - Нормализация полученной матрицы
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum += kernel[i, j]
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum
    print("Нормализованная матрица ядра свертки:")
    print(kernel)

    # применение операции свёртки
    imgBlur = Convolution(img, kernel)
    return imgBlur


# реализация операции свёртки изображения
def Convolution(img, kernel):
    kernel_size = len(kernel)
    imgBlur = img.copy()

    # Вычисляем начальные координаты для итераций по пикселям
    x_start = kernel_size // 2
    y_start = kernel_size // 2

    # Начинаем проход по каждому пикселю изображения, исключая края, чтобы не "выходить за границы" изображения
    for i in range(x_start, imgBlur.shape[0] - x_start):
        for j in range(y_start, imgBlur.shape[1] - y_start):
            val = 0

            # Проходимся по каждому элементу ядра свертки(центр+-kernel_size // 2)
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    # Умножаем значение текущего пикселя(центральный+смещение) изображения на соответствующий элемент ядра свертки и суммируем
                    val += img[i + k, j + l] * kernel[k + (kernel_size // 2), l + (kernel_size // 2)]

            # Значение val становится новым значением пикселя в результирующем изображении
            imgBlur[i, j] = val

    # Возвращаем результирующее изображение после свертки
    return imgBlur


# реализация функции Гаусса
def gauss(x, y, omega, a, b):
    omegaIn2 = 2 * omega ** 2
    m1 = 1/(np.pi * omegaIn2)
    m2 = np.exp(-((x-a) ** 2 + (y-b) ** 2)/omegaIn2)
    return m1*m2


main()
