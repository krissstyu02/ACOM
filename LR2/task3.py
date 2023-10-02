import cv2
import numpy as np


# Операция эрозии - уменьшение изображения для удаления мелких деталей и шумов
def erode(image, kernel):
    m, n, _ = image.shape  # Получаем высоту и ширину изображения
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    # Проходимся по каждому пикселю изображения, начиная с пикселей, где размер ядра помещается полностью
    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            # Вычисляем минимум среди пикселей внутри ядра, только если соответствующий элемент ядра равен 1
            eroded[i, j] = np.min(
              #создания подматрицы (подобласти) изображения вокруг текущего пикселя
                image[i - hkm :i + hkm + 1, j - hkn :j + hkn + 1][kernel == 1])

    return eroded

# Операция дилатации - увеличение изображения для заполнения недостающих частей
def dilate(image, kernel):
    m, n, _ = image.shape  # Получаем высоту и ширину изображения
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(image)

    # Проходимся по каждому пикселю изображения, начиная с пикселей, где размер ядра помещается полностью
    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            # Вычисляем максимум среди пикселей внутри ядра, только если соответствующий элемент ядра равен 1
            dilated[i, j] = np.max(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated



# объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # определение диапазона красного цвета в HSV
    lower_red = np.array([0, 0, 100])  # минимальные значения оттенка, насыщенности и яркости
    upper_red = np.array([60, 255, 255])  # максимальные значения оттенка, насыщенности и яркости

    # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета, имеют значение 255 (белый), а остальные пиксели имеют значение 0 (черный).
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # применение маски на изображение
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # структурирующий элемент (ядро) размером 5x5, который представляет собой матрицу, где все элементы установлены в 1
    # (определяет размер и форму области)
    kernel = np.ones((5, 5), np.uint8)

    # Операция открытия - 1)эрозия, 2)дилатация.
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    # Операция закрытия-1)дилатации, 2) операция эрозии.
    closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    # Применение операции эрозии
    # eroded = erode(res, kernel)
    #
    # # Применение операции дилатации
    # dilated = dilate(res, kernel)
    #
    # # Отображение изображений с применением эрозии и дилатации
    # cv2.imshow('Erosion', eroded)
    # cv2.imshow('Dilation', dilated)


    # нажатие клавиши esc для выхода из цикла
    if cv2.waitKey(1) & 0xFF == 27:
        break

# освобождение ресурсов окна
cap.release()
cv2.destroyAllWindows()

