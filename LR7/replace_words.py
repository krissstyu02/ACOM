#метод использовался для ускоренного создания аннотаций
with open('annotations_2.txt', 'r',encoding='utf-8') as file:
    filedata = file.read()

filedata = filedata.replace('« Иногда очень хочется полной свободы » . Драма Евгения Стычкина — с детективной интригой и молодыми звездами',
                            'На работу утром хожу очень быстро, чтобы не передумать.')

with open('ann3.txt', 'w') as file:
    file.write(filedata)
