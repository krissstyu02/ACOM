import os
import json
import time
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr

class ImageRecognitionTester:
    def __init__(self):
        # Устанавливаем путь к исполняемому файлу Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def read_annotations(self, annotation_file):
        # Читаем аннотации из файла и возвращаем их в виде списка кортежей
        with open(annotation_file, 'r', encoding='utf-8') as file:
            annotations = [line.strip().split(maxsplit=1) for line in file.readlines()]
        return annotations

    def straight_recognition(self, img):
        # Выполняем прямое распознавание с использованием Tesseract для предоставленного изображения
        start_time = time.time()
        text = pytesseract.image_to_string(img, lang='rus+eng')
        end_time = time.time()
        elapsed_time = end_time - start_time
        return text.strip(), elapsed_time

    def easyocr_recognition(self, image_path):
        # Выполняем распознавание с использованием EasyOCR для предоставленного изображения
        start_time = time.time()
        reader = easyocr.Reader(['en', 'ru'])
        result = reader.readtext(image_path)
        text = ' '.join([item[1] for item in result])
        end_time = time.time()
        elapsed_time = end_time - start_time
        return text.strip(), elapsed_time

    def evaluate_accuracy(self, expected_text, recognized_text):
        # Оцениваем точность сравнивая ожидаемый и распознанный тексты
        # разделяем текст на слова-множества
        expected_words = set(expected_text.lower().split())
        recognized_words = set(recognized_text.lower().split())
        # Полное совпадение текста
        full_match = expected_text.lower() == recognized_text.lower()

        # Подсчет процента правильно распознанных слов
        # находим пересечения множеств expected_words и recognized_words
        # Затем мы делим размер пересечения на размер множества expected_words
        correct_word_percentage = (len(expected_words.intersection(recognized_words)) / len(expected_words)) * 100

        return full_match, correct_word_percentage

    def rotate_and_resize_image(self, image, angle, background_color):
        # Поворачиваем и изменяем размер изображения на основе предоставленного угла и цвета фона
        #находим центр изображения-кортеж ширина и высота/2
        center = tuple(np.array(image.shape[1::-1]) / 2)
        #коэффициент уменьшения изображения-чем больше угол, тем больше уменьшаем картинку
        scale_factor = 1.0 - 0.03 * abs(angle)
        #создает матрицу преобразования для поворота изображения вокруг его центра.
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
        #применяем матрицу преобразования к изображению Это создает повернутое изображение.
        # flags определяет метод интерполяции, borderMode и borderValue определяют поведение в случае, если после поворота появляются
        # пустые области, background_color используется для заполнения этих областей.
        rotated_img = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=background_color)
        return rotated_img

#метод для аугментации и распознавания изображения
    def augment_and_recognize(self, rec_type, image_path, expected_text):
        # Аугментируем изображение, поворачивая и изменяя размер, затем выполняем распознавание для каждого аугментированного изображения
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        background_color = (255, 255, 255)

        results = []

        #для всех углов
        for angle in range(-20, 21, 1):
            #поворачиваем изображение
            rotated_img = self.rotate_and_resize_image(img, angle, background_color=background_color)
            #преобразуем повернутое изображение в объект изображения библиотеки Pillow
            pil_image = Image.fromarray(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
            #применяем распознование
            if rec_type == 'straight_recognition':
                recognized_text, elapsed_time = self.straight_recognition(pil_image)
            elif rec_type == 'easyocr_recognition':
                recognized_text, elapsed_time = self.easyocr_recognition(rotated_img)
            else:
                raise ValueError("Неподдерживаемый тип распознавания")

            full_match, correct_word_percentage = self.evaluate_accuracy(expected_text, recognized_text)

            results.append({
                'Изображение': image_path,
                'Угол поворота': angle,
                'Ожидаемый текст': expected_text,
                'Распознанный текст': recognized_text,
                'Полное совпадение': full_match,
                'Процент правильных слов': correct_word_percentage,
                'Время распознавания': elapsed_time
            })

        return results

    def test_recognition_with_augmentation(self, rec_type, annotation_file, images_folder):
        # Тестируем распознавание с аугментацией для каждого изображения в наборе данных
        annotations = self.read_annotations(annotation_file)
        all_full = 0
        all_correct_word = 0
        start_time = time.time()

        results = []

        for annotation in annotations:
            image_name, expected_text = annotation
            image_path = os.path.join(images_folder, image_name)

            # Аугментируем и распознаем текущее изображение
            augmentation_results = self.augment_and_recognize(rec_type, image_path, expected_text)
            results.extend(augmentation_results)

            # Вычисляем средние метрики для всех аугментированных изображений
            avg_full_match = sum(result['Полное совпадение'] for result in augmentation_results) / len(augmentation_results)
            avg_correct_word = sum(result['Процент правильных слов'] for result in augmentation_results) / len(augmentation_results)

            all_full += avg_full_match
            all_correct_word += avg_correct_word

        # Сохраняем результаты в файл
        result_file_path = f'результаты_{rec_type}_augmented.json'
        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            json.dump(results, result_file, indent=4, ensure_ascii=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        all_full /= len(annotations)
        all_correct_word /= len(annotations)

        print(f"Метод распознавания: {rec_type} с аугментацией\n"
              f"Процент полного совпадения: {all_full * 100}%\n"
              f"Процент правильно распознанных слов: {all_correct_word}%\n"
              f"Время распознавания: {elapsed_time * 100}%\n")


if __name__ == "__main__":
    tester = ImageRecognitionTester()

    tester.test_recognition_with_augmentation(rec_type='straight_recognition', annotation_file='annotations_1.txt', images_folder='dataset_1')
    tester.test_recognition_with_augmentation(rec_type='easyocr_recognition', annotation_file='annotations_1.txt', images_folder='dataset_1')
