import cv2
import numpy as np

# объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # преобразование кадра в формат HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # определение диапазона красного цвета в HSV
    lower_red = np.array([0, 0, 100])  # минимальные значения оттенка, насыщенности и яркости
    upper_red = np.array([60, 255, 255]) # максимальные значения оттенка, насыщенности и яркости

    # Маска - бинарное изображение, где пиксели, соответствующие заданному диапазону цвета,
    # имеют значение 255 (белый), а остальные пиксели имеют значение 0 (черный).
    mask = cv2.inRange(hsv, lower_red, upper_red)

    #  выполняет операцию "И" между пикселями исходного изображения (frame) и маской (mask).
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('Original_video', frame)
    cv2.imshow('HSV_video', hsv)
    cv2.imshow('Result_video', res)

    # нажатие клавиши esc для выхода из цикла
    if cv2.waitKey(1) & 0xFF == 27:
        break

# освобождение ресурсов окна
cap.release()
cv2.destroyAllWindows()