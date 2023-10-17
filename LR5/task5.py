import cv2

i = 0

# Основная функция для обработки видео
def main(kernel_size, standard_deviation, delta_thresh, min_area):
    global i
    i += 1

    # Открываем видеофайл для чтения
    video = cv2.VideoCapture(r'ЛР5_main_video.mov', cv2.CAP_ANY)

    # Читаем первый кадр и преобразуем его в оттенки серого
    #ret-Это булевая переменная, которая указывает, удалось ли успешно прочитать кадр из видеопотока.
    #frame: Это переменная, которая содержит сам кадр как изображение.
    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применяем гауссово размытие для сглаживания шума
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    # Получаем ширину и высоту видеофайла в пикселях из видеопотока
    # используем метод get объекта видеозахвата video, чтобы получить значение конкретного свойства видеопотока
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Устанавливаем кодек и создаем объект для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(r'.\LR5_output ' + str(i) + '.mp4', fourcc, 144, (w, h))

    while True:
        # Сохраняем старый кадр, чтобы вычислить разницу между кадрами
        old_img = img.copy()
        ok, frame = video.read()
        if not ok:
            break

        # Преобразуем текущий кадр в оттенки серого и применяем гауссово размытие
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        # Вычисляем разницу между текущим и предыдущим кадрами
        diff = cv2.absdiff(img, old_img)

        # Применяем бинаризацию, чтобы выделить измененные области
        thresh = cv2.threshold(diff, delta_thresh, 255, cv2.THRESH_BINARY)[1]

        # Находим контуры в бинарном изображении
        (contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если на кадре есть хотя бы один контур, площадь которого больше min_area, записываем кадр
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            video_writer.write(frame)

    # Закрываем объект записи видео
    video_writer.release()


# Первый набор параметров
# умеренное размытие и порог детекции для изменений, а также минимальную площадь объекта для сохранения.
# замедленные кадры
kernel_size = 3
standard_deviation = 80
delta_thresh = 60
min_area = 20
main(kernel_size, standard_deviation, delta_thresh, min_area)

# Второй набор параметров
#увеличение размера ядра и уменьшение стандартного отклонения-лучший вариант
kernel_size = 11
standard_deviation = 60
delta_thresh = 60
min_area = 20
main(kernel_size, standard_deviation, delta_thresh, min_area)

# Третий набор параметров
#слишком маленькаое значение дельты-попали кадры с минимальными изменениями, которые нам не нужны
kernel_size = 9
standard_deviation = 30
delta_thresh = 2
min_area = 20
main(kernel_size, standard_deviation, delta_thresh, min_area)

# Четвертый набор параметров
#очень большое значение min_area-зафиксированы не все кадры изменений
kernel_size = 3
standard_deviation = 50
delta_thresh = 60
min_area = 1000
main(kernel_size, standard_deviation, delta_thresh, min_area)

