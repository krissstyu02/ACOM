import cv2
import numpy as np


img = cv2.imread('logo.png')
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)

# Установка параметров для прямоугольников и линии
color = (0, 0, 255)  # Красный цвет в формате BGR (синий, зеленый, красный)
thickness = 2  # Толщина линии

# Получение размеров изображения (ширина, высота)
height, width, _ = img.shape

# Вертикальный прямоугольник
rect_width_1 = 50
rect_height_1 = 400
x1_1 = width // 2 - rect_width_1 // 2
y1_1 = height // 2 - rect_height_1 // 2
x2_1 = width // 2 + rect_width_1 // 2
y2_1 = height // 2 + rect_height_1 // 2

# Горизонтальный прямоугольник
rect_width_2 = 50
rect_height_2 = 350
x1_2 = width // 2 - rect_height_2 // 2
y1_2 = height // 2 - rect_width_2 // 2
x2_2 = width // 2 + rect_height_2 // 2
y2_2 = height // 2 + rect_width_2 // 2

# Отрисовка прямоугольников на изображении
cv2.rectangle(img, (x1_1, y1_1), (x2_1, y2_1), color, thickness)
cv2.rectangle(img, (x1_2, y1_2), (x2_2, y2_2), color, thickness)

# Размер ядра для размытия
kernel_size = (71, 11)

# Выделение части изображения, соответствующей горизонтальному прямоугольнику
img_part = img[y1_2:y2_2, x1_2:x2_2]

# Применение размытия к выделенной части изображения
img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 30)

# Замена исходной части изображения размытой версией
img[y1_2:y2_2, x1_2:x2_2] = img_part_blur

# Определение цвета центрального пикселя
cx = width // 2
cy = height // 2
r, g, b = img[cy][cx]

# Список возможных цветов в формате RGB
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Расчет расстояний до каждого цвета и выбор ближайшего цвета
distances = [np.sqrt((r - color[0])**2 + (g - color[1])**2 + (b - color[2])**2) for color in colors]
min_index = distances.index(min(distances))
nearest_color = colors[min_index]

# Закрашивание прямоугольников ближайшим цветом
cv2.rectangle(img, (x1_1, y1_1), (x2_1, y2_1), nearest_color, -1)
cv2.rectangle(img, (x1_2, y1_2), (x2_2, y2_2), nearest_color, -1)

# Отображение изображения
cv2.imshow('Display window', img)

# Ожидание нажатия клавиши и закрытие окна по нажатию
cv2.waitKey(0)
cv2.destroyAllWindows()
