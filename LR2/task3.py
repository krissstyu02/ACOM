import cv2
import numpy as np

# объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # определение диапазона красного цвета в HSV
    lower_red = np.array([0, 0, 100])  # минимальные значения оттенка, насыщенности и значения(яркости)
    upper_red = np.array([100, 100, 255])  # максимальные значения оттенка, насыщенности и значения(яркости)

    # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета, имеют значение 255 (белый), а остальные пиксели имеют значение 0 (черный).
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # применение маски на изображение
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # структурирующий элемент(определяет размер и форму области)
    kernel = np.ones((5, 5), np.uint8)

    # применение операции открытия - позволяет удалить шумы и мелкие объекты на изображении(удаление нежелательных пикселей или деталей)
    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    # применение операции закрытия - позволяет заполнить маленькие пробелы и разрывы в объектах на изображении
    closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    # нажатие клавиши esc для выхода из цикла
    if cv2.waitKey(1) & 0xFF == 27:
        break

# освобождение ресурсов окна
cap.release()
cv2.destroyAllWindows()

def erode(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return eroded


def dilate(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated