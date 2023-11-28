import os
import json
import time
from PIL import Image
import pytesseract
import easyocr
import re  # модуль re для работы с регулярными выражениями
import nltk

# nltk.download('punkt')

# задание 1,2,8
class ImageRecognitionTester:
    # инициализируем Тессеракт
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # считываем аннотации-расшифровки из файла
    def read_annotations(self, annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as file:
            annotations = [line.strip().split(maxsplit=1) for line in file.readlines()]
        return annotations

    def newlines(self,text):
        # Токенизация текста и вставка пробелов между словами
        tokens = nltk.word_tokenize(text)
        fixed_text = ' '.join(tokens)
        return fixed_text

    # распознавание Тессеракт
    def straight_recognition(self, img):
        start_time = time.time()
        text = pytesseract.image_to_string(img, lang='rus+eng')
        fixed_text = self.newlines(text)
        end_time = time.time()
        return fixed_text.strip(), end_time - start_time

    # распознавание easyorc
    def easyocr_recognition(self, image_path):
        start_time = time.time()
        reader = easyocr.Reader(['en', 'ru'])
        result = reader.readtext(image_path)
        text = ' '.join([item[1] for item in result])
        end_time = time.time()
        return text.strip(), end_time - start_time

    # оценка качества распознавания
    def evaluate_accuracy(self, expected_text, recognized_text):
        expected_words = set(expected_text.lower().split())
        recognized_words = set(recognized_text.lower().split())

        full_match = expected_text.lower() == recognized_text.lower()

        correct_word_percentage = (len(expected_words.intersection(recognized_words)) / len(expected_words)) * 100

        return full_match, correct_word_percentage


    # постобработка текста - удаление спецсимволов и приведение к нижнему регистру
    def postprocess_text(self, text):
        # Используем регулярное выражение для удаления спецсимволов и приведения к нижнему регистру
        processed_text = re.sub(r'[^a-zA-Zа-яА-Я0-9.—,№:\s-]', '', text)
        processed_text = processed_text.replace('\n','')
        return processed_text

    # главная функция распознавания изображений
    def test_recognition(self, rec_type, annotation_file, images_folder):
        annotations = self.read_annotations(annotation_file)
        results = []
        all_full = 0
        all_correct_word = 0
        start_time = time.time()

        for annotation in annotations:
            image_name, expected_text = annotation
            image_path = os.path.join(images_folder, image_name)
            img = Image.open(image_path)

            # в зависимости от метода вызываем необходимую функцию
            if rec_type == 'straight_recognition':
                recognized_text, elapsed_time = self.straight_recognition(img)
            elif rec_type == 'easyocr_recognition':
                recognized_text, elapsed_time = self.easyocr_recognition(image_path)
            else:
                raise ValueError("Неподдерживаемый тип распознавания")

            # выполняем постобработку текста
            processed_text = self.postprocess_text(recognized_text)

            # получаем оценку качества распознавания
            full_match, correct_word_percentage = self.evaluate_accuracy(expected_text, processed_text)

            results.append({
                'Изображение': image_name,
                'Ожидаемый текст': expected_text,
                'Распознанный текст': processed_text,
                'Полное совпадение': full_match,
                'Процент правильных слов': correct_word_percentage,
                'Время распознавания': elapsed_time
            })
            all_full += int(full_match)
            all_correct_word += correct_word_percentage

        # записываем результаты для каждой картинки в файл
        result_file_path = f'результаты_{rec_type}_form_2.json'
        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            json.dump(results, result_file, indent=4, ensure_ascii=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        all_full /= len(annotations)
        all_correct_word /= len(annotations)

        # выводим общую характеристику
        print(f"Метод разпознавания:{rec_type}\n"
              f"Процент полного совпадения:{all_full * 100}%\n"
              f"Процент правильно распознанных слов:{all_correct_word}%\n"
              f"Время распознования:{elapsed_time * 100}%\n")

if __name__ == "__main__":
    tester = ImageRecognitionTester()

    tester.test_recognition(rec_type='straight_recognition', annotation_file='annotations_2.txt', images_folder='dataset_2')
    tester.test_recognition(rec_type='easyocr_recognition', annotation_file='annotations_2 _2.txt', images_folder='dataset_2')
