import os
import json
import time
from PIL import Image
import pytesseract
import easyocr
import string
import nltk

# nltk.download('punkt')

#задание 4

def newlines(text):
    # Токенизация текста и вставка пробелов между словами
    tokens = nltk.word_tokenize(text)
    fixed_text = ' '.join(tokens)
    return fixed_text

class ImageRecognitionTester:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def read_annotations(self, annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as file:
            annotations = [line.strip().split(maxsplit=1) for line in file.readlines()]
        return annotations

    # Пример использования в функции straight_recognition
    def straight_recognition(self, img):
        start_time = time.time()
        text = pytesseract.image_to_string(img, lang='rus+eng')
        #сделать пробелы между словами после удаления знаков препинания
        fixed_text = newlines(text)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return fixed_text.strip(), elapsed_time

    def easyocr_recognition(self, image_path):
        start_time = time.time()
        reader = easyocr.Reader(['en', 'ru'])
        result = reader.readtext(image_path)
        text = ' '.join([item[1] for item in result])
        end_time = time.time()
        elapsed_time = end_time - start_time
        return text.strip(), elapsed_time


    def evaluate_accuracy(self, expected_text, recognized_text):
        #убрать знаки препинания
        translator = str.maketrans('', '', string.punctuation)
        expected_text_new=expected_text.translate(translator)
        recognized_text_new=recognized_text.translate(translator)

        expected_words = set(expected_text_new.lower().split())
        recognized_words = set(recognized_text_new.lower().split())

        # Полное совпадение текста
        full_match = expected_text.lower() == recognized_text.lower()

        # Подсчет процента правильно распознанных слов
        correct_word_percentage = (len(expected_words.intersection(recognized_words)) / len(expected_words)) * 100

        return full_match, correct_word_percentage


    def test_recognition(self, rec_type, annotation_file, images_folder):
        annotations = self.read_annotations(annotation_file)
        results = []
        all_full=0
        all_correct_word=0
        start_time = time.time()

        for annotation in annotations:
            image_name, expected_text = annotation
            image_path = os.path.join(images_folder, image_name)
            img = Image.open(image_path)

            if rec_type == 'straight_recognition':
                recognized_text, elapsed_time = self.straight_recognition(img)
            elif rec_type == 'easyocr_recognition':
                recognized_text, elapsed_time = self.easyocr_recognition(image_path)
            else:
                raise ValueError("Неподдерживаемый тип распознавания")

            #убрать табуляцию
            recognized_text = recognized_text.replace('\n', '')
            full_match, correct_word_percentage = self.evaluate_accuracy(expected_text, recognized_text)

            results.append({
                'Изображение': image_name,
                'Ожидаемый текст': expected_text,
                'Распознанный текст': recognized_text,
                'Полное совпадение': full_match,
                'Процент правильных слов': correct_word_percentage,
                'Время распознавания': elapsed_time
            })
            all_full+=int(full_match)
            all_correct_word+=correct_word_percentage

        result_file_path = f'результаты_{rec_type}_form.json'
        with open(result_file_path, 'w', encoding='utf-8') as result_file:
            json.dump(results, result_file, indent=4, ensure_ascii=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        all_full/=len(annotations)
        all_correct_word/=len(annotations)
        print(f"Метод разпознавания:{rec_type}\n"
              f"Процент полного совпадения:{all_full*100}%\n"
              f"Процент правильно распознанных слов:{all_correct_word}%\n"
              f"Время распознования:{elapsed_time * 100}%\n")

if __name__ == "__main__":
    tester = ImageRecognitionTester()
    tester.test_recognition(rec_type='straight_recognition', annotation_file='annotations_1.txt', images_folder='dataset_1')
    tester.test_recognition(rec_type='easyocr_recognition', annotation_file='annotations_1.txt', images_folder='dataset_1')
