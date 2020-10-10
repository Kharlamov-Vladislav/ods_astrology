# Предсказание подходящего заголовка статьи в Nature по астрологическим данным с последующим предсказанием цитирования. (ODS hackaton)

### [telegram: @HirschMagicbot](https://t.me/HirschMagicbot)


*OpenDataScience, 2020*

*Авторы: <b> [V.D. Kharlamov](https://t.me/justfairy), [Vladislav](https://t.me/quantum_forest), [Vadim](https://t.me/AndersA), [Vladislava](https://t.me/@awniar)  </b>*

*Ментор:* [Roman Romadin](https://t.me/RomanRomadin)


## Цели проекта
Используя данные о заголовках статей журнала Nature, раздела Computer Science, мы предсказываем по дате рождения автора и его предполагаемой дате публикации подходящий для него заголовок используя астрологические данные (такие как положение планет, ...). Помимо предсказывания подходящего заголовка, обучен линейный классификатор, который предсказывает квартиль успешности цитирования статьи используя разреженную матрицу от 1 до 5 нграмм с фичами по TfIdf реализации Sklearn'а.    

## Методы
Все файлы связанные с использованной моделью лежат в `/model`. Использована нейросеть с архитектурой LTSM (подробнее в `model/model.py`), веса `classic.h5`.    
Все файлы связанные с обработкой и сбором данных лежат в `/scripts`. Для сбора данных использован файл `scripts/parser.py`.      
