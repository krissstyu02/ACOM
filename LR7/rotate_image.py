import os
import cv2
import numpy as np

#метод для программного аугментирования датасета
def augment_dataset(input_folder, output_folder, rotation_range=(-20, 21, 1), scale_factor=0.8,
                    background_color=(255, 255, 255)):
    # Создаем папку для сохранения аугментированных изображений
    os.makedirs(output_folder, exist_ok=True)

    # Получаем список файлов в папке с исходными изображениями
    image_files = [file for file in os.listdir(input_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        #считываем изображение
        img = cv2.imread(image_path)

        # Поворачиваем и уменьшаем изображение на углы в заданном диапазоне
        for angle in range(rotation_range[0], rotation_range[1], rotation_range[2]):
            rotated_img = rotate_and_resize_image(img, angle, background_color)
            output_file = f"{os.path.splitext(image_file)[0]}_{angle}.png"
            output_path = os.path.join(output_folder, output_file)
            #сохраняем
            cv2.imwrite(output_path, rotated_img)


def rotate_and_resize_image(image, angle,  background_color):
    # Получаем центр изображения(ширина высота /2)
    center = tuple(np.array(image.shape[1::-1]) / 2)

    # Вычисляем коэффициент масштабирования в зависимости от размера поворота
    scale_factor = 1.0 - 0.03 * abs(angle)

    # Получаем матрицу преобразования и поворачиваем изображение
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_img = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)

    return rotated_img


# входная и выходная папки
input_folder = 'dataset_1'
output_folder = 'dataset_2'

augment_dataset(input_folder, output_folder)
