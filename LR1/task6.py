import cv2

img = cv2.imread('cam_photo.jpg')
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)

# цвет и толщина прямоугольников и линии
color = (0, 0, 255) # красный цвет
thickness = 2 # толщина

# размеры изображения
height, width, _, = img.shape

# ширина и высота вертикального прямоугольника
rect_width_1 = 50
rect_height_1 = 400

# координаты углов для вертикального прямоугольника
x1_1 = width // 2 - rect_width_1 // 2 # левый верхний угол по оси x
y1_1 = height // 2 - rect_height_1 // 2 # левый верхний угол по оси y
x2_1 = width // 2 + rect_width_1 // 2 # правый нижний угол по оси x
y2_1 = height // 2 + rect_height_1 // 2 # правый нижний угол по оси y

# ширина и высота горизонтального прямоугольника
rect_width_2 = 50
rect_height_2 = 350

# координаты углов горизонательного прямоугольника
x1_2 = width // 2 - rect_height_2 // 2 # левый верхний угол по оси x
y1_2 = height // 2 - rect_width_2 // 2 # левый верхний угол по оси y
x2_2 = width // 2 + rect_height_2 // 2 # правый нижний угол по оси x
y2_2 = height // 2 + rect_width_2 // 2 # правый нижний угол по оси y

# отрисовка
cv2.rectangle(img, (x1_1, y1_1), (x2_1, y2_1), color, thickness)
cv2.rectangle(img, (x1_2, y1_2), (x2_2, y2_2), color, thickness)

# размер ядра для размытия
kernel_size = (71, 11) # ширина и высота ядра в пикселях

# часть изображения, соответствующая горизонтальному прямоугольнику
img_part = img[y1_2:y2_2, x1_2:x2_2]

img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 30)

# замена части изображения размытой версией
img[y1_2:y2_2, x1_2:x2_2] = img_part_blur

cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
