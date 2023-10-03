import cv2
import numpy as np

# объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # определение диапазона красного цвета в HSV
    lower_red = np.array([0, 100, 100])  # минимальные значения оттенка, насыщенности и значения (яркости)
    upper_red = np.array([8, 255, 255])  # максимальные значения оттенка, насыщенности и значения (яркости)

    # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета, имеют значение 255 (белый), а остальные пиксели имеют значение 0 (черный).
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Выполнить морфологические операции для разделения близких объектов
    kernel = np.ones((5, 5), np.uint8)  # Создать ядро (прямоугольник) размером 5x5
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Применить операцию открытия (убрать мелкие шумы)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Применить операцию закрытия (заполнить дыры)

    # поиск контуров в бинарном изображении.
    # Контур - это набор точек, представляющих границу объекта на изображении
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL, #выделить только наружные границы объектов.
                                   cv2.CHAIN_APPROX_SIMPLE) #контуры будут аппроксимироваться с минимальным количеством точек, чтобы сохранить память

    # Отрисовать контуры на исходном кадре и вычислить моменты
    for contour in contours:
        area = cv2.contourArea(contour)  # Вычислить площадь контура
        if area > 100:  # Отфильтровать маленькие контуры (по площади)
            moments = cv2.moments(contour)  # Вычислить моменты контура
            # Найти координаты центра объекта
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])
            width = height = int(np.sqrt(area))
            cv2.rectangle(frame,
                          # верхний левый угол
                          (c_x - (width ), c_y - (height)),
                          # нижний правый угол
                          (c_x + (width), c_y + (height)),
                          (0, 0, 0), 2)

    cv2.imshow('HSV_frame', hsv)
    cv2.imshow('Result_frame', frame)

    # нажатие клавиши esc для выхода из цикла
    if cv2.waitKey(1) & 0xFF == 27:
        break

# освобождение ресурсов окна
cap.release()
cv2.destroyAllWindows()
