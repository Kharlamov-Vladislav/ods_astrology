import logging
from datetime import datetime
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from config import TOKEN

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.utils as ku
from keras.models import load_model
from collections import defaultdict
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn import linear_model
import random
import pandas as pd
import numpy as np
from pathlib import Path
from nltk import pos_tag

# str: dates -> keywords
from get_keywords import get_keywords

gachi_tags = ['ass', 'fucking', 'slaves', 'oh', 'fuck', 'mad', 'semen', 'master', 'fantasies',
              'ahhhhhhh', 'atteeention', 'fat cock', 'cock', 'cumming', 'cum', 'gay sex', 'gay', 'stick finger',
              'penetration', 'woo-woo', 'suction', 'fucking slaves', 'yeaaaah', 'spanking', 'anal']

dict_gachi = defaultdict(list)

#####################################
# Gachi block
def all_tags(text):
    ans = []
    for word in text:
        ans.append(pos_tag([word]))

    return ans


for tag in all_tags(gachi_tags):
    word, tag = tag[0]
    dict_gachi[tag].append(word)


def replace_text(text, gachi):
    p = 5 / len(text)
    flag = True
    answer_text = []
    text = all_tags(text.split())
    for word in reversed(text):
        word, tag = word[0]
        if flag:
            if gachi[tag]:
                flag = False
                word = random.choice(gachi[tag])
        if p > random.random():
            if gachi[tag]:
                word = random.choice(gachi[tag])
        answer_text.append(word)
    return ' '.join(answer_text)

#####################################
# LTSM Predict

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


inp_sequences, total_words = get_sequence_of_tokens(corpus)


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'), dtype='int16')

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words, dtype='int16')
    return predictors, label, max_sequence_len


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()


predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

#####################################
#Preprocessing funcs

def multiclass(x):
    if x < quantiles[0]:
        return 0
    elif quantiles[0] < x < quantiles[1]:
        return 1
    else:
        return 2

def predict_quantile(text):
    text = vectorizer.transform([text])
    predict_quantile_ = int(clf.predict(text))

    mean_citation = means_quantiles[predict_quantile_] * 365
    std_citation = stds_quantiles[predict_quantile_] * 365

    describe = f"Статьи с похожим заголовком, чаще всего находятся между {predict_quantile_} и {predict_quantile_ + 1} " \
               f"квартиле всех работ опубликованных в Nature в категории Computer Science. \n \n" \
               f"В среднем работы в этом промежутке набирают {int(mean_citation)} цитирований в год." \
               f" По нашему предсказанию, вы получите {int(np.random.normal(mean_citation, std_citation))} цитирований за год."

    return describe

def predict_quantile_ddf(text):
    text = vectorizer.transform([text])
    predict_quantile_ = int(clf.predict(text))

    mean_citation = means_quantiles[predict_quantile_] * 365
    std_citation = stds_quantiles[predict_quantile_] * 365

    describe = f"Статьи с похожим заголовком, чаще всего находятся между {predict_quantile_} и {predict_quantile_ + 1} " \
               f"квартиле всех работ опубликованных в 'Gachi Science' в категории Fucking Slaves. \n \n" \
               f"В среднем работы в этом промежутке набирают {int(mean_citation)} penetration в год." \
               f" По нашему предсказанию, вы получите {int(np.random.normal(mean_citation, std_citation))} deep penetration за год. \n"

    return describe

#####################################


app_dir: Path = Path(__file__).parent
model_path = app_dir / "../model/classic.h5"
ncs_path = app_dir / "../datasets nature/nature_computer_science.csv"

model = load_model(model_path)
tokenizer = Tokenizer()

df = pd.read_csv(ncs_path)
corpus = df['title']
tokenizer = Tokenizer()

########################################################
# Predict citation
df['submit_date'] = [(datetime.now() - datetime.strptime(date, '%Y-%m-%d')).days for date in df['submit_date']]
df['citations'] = [c / d for c, d in df[['citations', 'submit_date']].values]

means_quantiles = [df['citations'][df['citations'] < df['citations'].quantile(0.33 * i)].mean() for i in range(1, 4)]
stds_quantiles = [df['citations'][df['citations'] < df['citations'].quantile(0.33 * i)].std() for i in range(1, 4)]
quantiles = [df['citations'].quantile(0.33 * i) for i in range(1, 4)]


vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]{1,}\b',
                                     min_df=0.0005, max_df=0.9, ngram_range=(1, 5), max_features=10000)

shuffle_df = df.sample(frac=1)
X, y = vectorizer.fit_transform(shuffle_df['abstract']), shuffle_df['citations'].apply(lambda x: multiclass(x))


clf = linear_model.LogisticRegression(class_weight='balanced', max_iter=10000)
clf.fit(X, y)

######################################3


API_TOKEN = TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


# States
class Form(StatesGroup):
    Classic = State()  # Will be represented in storage as 'Form:Classic'
    Classic1 = State()  # Will be represented in storage as 'Form:Classic1'
    ClassicB = State()  # Will be represented in storage as 'Form:ClassicB'
    ClassicB1 = State()  # Will be represented in storage as 'Form:ClassicB'
    DDF = State()  # Will be represented in storage as 'Form:DDF'
    DDF1 = State()  # Will be represented in storage as 'Form:DDF'


@dp.message_handler(commands=['start'], state='*')
async def send_welcome(message: types.Message):
    button_start = KeyboardButton('Начать')
    button_cancel = KeyboardButton('Отменить')
    greet_kb = ReplyKeyboardMarkup(resize_keyboard=True)
    greet_kb.add(button_start).add(button_cancel)
    await message.reply('Добро пожаловать! это ХиршМэйджикбот. С помощью нейросетей и магии он способен па дате '
                        'рождения автора и дате публикации предсказать абстракт и цитируемость статьи!')
    await message.reply("Если хотите начать работу - жмём кнопку!", reply_markup=greet_kb)


@dp.message_handler(commands=['help'], state="*")
async def process_help_command(message: types.Message):
    await message.reply("Раздел Help находится в разработке xD")


@dp.message_handler(Text(equals='Отменить', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info('Cancelling state %r', current_state)
    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    button_start = KeyboardButton('Начать')
    button_cancel = KeyboardButton('Отменить')
    greet_kb = ReplyKeyboardMarkup(resize_keyboard=True)
    greet_kb.add(button_start).add(button_cancel)
    await message.reply('Отменено', reply_markup=greet_kb)


@dp.message_handler(text=['Начать'], state="*")
async def process_begin_command(message: types.Message):
    button_model_1 = InlineKeyboardButton('Classic', callback_data='a')
    button_model_2 = InlineKeyboardButton('Classic B', callback_data='b')
    button_model_3 = InlineKeyboardButton('Deep dark fantasies', callback_data='c')
    modeling_kb = InlineKeyboardMarkup(resize_keyboard=True).add(button_model_1).add(button_model_2).add(
        button_model_3)
    await message.reply("Итак, пожалуйста выберите модель, с которой будем работать", reply_markup=modeling_kb)
    button_start = KeyboardButton('Начать')
    button_cancel = KeyboardButton('Отменить')
    greet_kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    greet_kb.add(button_start).add(button_cancel)


@dp.callback_query_handler(text='a')
async def process_callback_button_classic(callback_query: types.CallbackQuery):
    await bot.send_message(callback_query.from_user.id, 'Выбрана модель Classic')
    await callback_query.answer()
    await Form.Classic.set()
    await bot.send_message(callback_query.from_user.id,
                           'Приступим к вводу данных: пожалуйста введите дату рождения автора в формате dd.mm.yyyy')


@dp.callback_query_handler(text='b')
async def process_callback_button_classic_b(callback_query: types.CallbackQuery):
    await bot.send_message(callback_query.from_user.id, 'Выбрана модель Classic B')
    await callback_query.answer()
    await Form.ClassicB.set()
    await bot.send_message(callback_query.from_user.id,
                           'Приступим к вводу данных: пожалуйста введите дату рождения автора в формате dd.mm.yyyy')


@dp.callback_query_handler(text='c')
async def process_callback_button_DDF(callback_query: types.CallbackQuery):
    await bot.send_message(callback_query.from_user.id, 'Выбрана модель Deep dark fantasies')
    await callback_query.answer()
    await Form.DDF.set()
    await bot.send_message(callback_query.from_user.id,
                           'Приступим к вводу данных: пожалуйста введите дату рождения автора в формате dd.mm.yyyy')


# Следующие три блока работают для выбранной модели - Classic
@dp.message_handler(state=Form.Classic, content_types=types.ContentTypes.TEXT)
async def process_enter_b_date(message: types.Message, state: FSMContext):
    try:
        b_date = datetime.strptime(message.text, '%d.%m.%Y')
    except ValueError:
        await message.reply('дата введена в неправильном формате, попробуйте снова')
        return
    # Логирование можно убрать
    logging.info('получена дата %r', b_date)
    await Form.Classic1.set()   # Смена состояния
    await message.reply('Введена дата рождения автора: %r' % b_date.strftime("%d.%m.%Y"))
    await message.answer('Теперь введите дату публикации, в том же формате')
    await state.update_data(b_date=b_date)  # Запоминание введенной переменной b_date в состояние


@dp.message_handler(state=Form.Classic1, content_types=types.ContentTypes.TEXT)
async def process_enter_pub_date(message: types.Message, state: FSMContext):
    try:
        pub_date = datetime.strptime(message.text, '%d.%m.%Y')
    except ValueError:
        await message.reply('дата введена в неправильном формате, попробуйте снова')
        return
    logging.info('получена дата публикации %r', pub_date)
    await message.reply('Введена дата публикации %r' % pub_date.strftime("%d.%m.%Y"))
    button_magic = InlineKeyboardButton('Да', callback_data='x')
    magic_kb = InlineKeyboardMarkup(resize_keyboard=True).add(button_magic)
    await message.reply('Все готово! Начинаем колдовать?', reply_markup=magic_kb)
    await state.update_data(pub_date=pub_date)


@dp.callback_query_handler(state=Form.Classic1, text='x')
async def process_send_to_magic(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    await state.get_data()
    my_dict = await state.get_data()   # Вытаскиваем значения переменных b_date и pub_date в словарь
    magic_dates = [my_dict['b_date'], my_dict['pub_date']]  # Передаем их в список
    # Заменить на отправку в модель Classic:
    keywords = get_keywords(*magic_dates)
    title = generate_text(keywords[0], 10, model, random.randint(8, max_sequence_len - 1))
    describe = predict_quantile(title)
    response = f"Подходящий для вас заголовок для публикации: \n{title.title()} \n \n" + describe
    await bot.send_message(callback_query.from_user.id, response)
    # сообщением


# Следующие три блока работают для выбранной модели - ClassicB
@dp.message_handler(state=Form.ClassicB, content_types=types.ContentTypes.TEXT)
async def process_enter_b_date(message: types.Message, state: FSMContext):
    try:
        b_date = datetime.strptime(message.text, '%d.%m.%Y')
    except ValueError:
        await message.reply('дата введена в неправильном формате, попробуйте снова')
        return
    # Логирование можно убрать
    logging.info('получена дата %r', b_date)
    await Form.ClassicB1.set()   # Смена состояния
    await message.reply('Введена дата рождения автора: %r' % b_date.strftime("%d.%m.%Y"))
    await message.answer('Теперь введите дату публикации, в том же формате')
    await state.update_data(b_date=b_date)  # Запоминание введенной переменной b_date в состояние


@dp.message_handler(state=Form.ClassicB1, content_types=types.ContentTypes.TEXT)
async def process_enter_pub_date(message: types.Message, state: FSMContext):
    try:
        pub_date = datetime.strptime(message.text, '%d.%m.%Y')
    except ValueError:
        await message.reply('дата введена в неправильном формате, попробуйте снова')
        return
    logging.info('получена дата публикации %r', pub_date)
    await message.reply('Введена дата публикации %r' % pub_date.strftime("%d.%m.%Y"))
    button_magic = InlineKeyboardButton('Да', callback_data='x')
    magic_kb = InlineKeyboardMarkup(resize_keyboard=True).add(button_magic)
    await message.reply('Все готово! Начинаем колдовать?', reply_markup=magic_kb)
    await state.update_data(pub_date=pub_date)


@dp.callback_query_handler(state=Form.ClassicB1, text='x')
async def process_send_to_magic(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    await state.get_data()
    my_dict = await state.get_data()   # Вытаскиваем значения переменных b_date и pub_date в словарь
    magic_dates = [my_dict['b_date'], my_dict['pub_date']]  # Передаем их в список
    # Заменить на отправку в модель ClassicB:
    await bot.send_message(callback_query.from_user.id, '%r' % magic_dates)    # Сейчас бот отправляет полученный список
    # сообщением


# Следующие три блока работают для выбранной модели - DDF
@dp.message_handler(state=Form.DDF, content_types=types.ContentTypes.TEXT)
async def process_enter_b_date(message: types.Message, state: FSMContext):
    try:
        b_date = datetime.strptime(message.text, '%d.%m.%Y')
    except ValueError:
        await message.reply('дата введена в неправильном формате, попробуйте снова')
        return
    # Логирование можно убрать
    logging.info('получена дата %r', b_date)
    await Form.DDF1.set()   # Смена состояния
    await message.reply('Введена дата рождения автора: %r' % b_date.strftime("%d.%m.%Y"))
    await message.answer('Теперь введите дату публикации, в том же формате')
    await state.update_data(b_date=b_date)  # Запоминание введенной переменной b_date в состояние


@dp.message_handler(state=Form.DDF1, content_types=types.ContentTypes.TEXT)
async def process_enter_pub_date(message: types.Message, state: FSMContext):
    try:
        pub_date = datetime.strptime(message.text, '%d.%m.%Y')
    except ValueError:
        await message.reply('дата введена в неправильном формате, попробуйте снова')
        return
    logging.info('получена дата публикации %r', pub_date)
    await message.reply('Введена дата публикации %r' % pub_date.strftime("%d.%m.%Y"))
    button_magic = InlineKeyboardButton('Да', callback_data='x')
    magic_kb = InlineKeyboardMarkup(resize_keyboard=True).add(button_magic)
    await message.reply('Все готово! Начинаем колдовать?', reply_markup=magic_kb)
    await state.update_data(pub_date=pub_date)


@dp.callback_query_handler(state=Form.DDF1, text='x')
async def process_send_to_magic(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()
    await state.get_data()
    my_dict = await state.get_data()   # Вытаскиваем значения переменных b_date и pub_date в словарь
    magic_dates = [my_dict['b_date'], my_dict['pub_date']]  # Передаем их в список
    # Заменить на отправку в модель DDF:
    keywords = get_keywords(*magic_dates)
    # state = random.randint(1, 2)
    # if state == 1:
    #     keywords = keywords[0] + ' ' + random.choice(gachi_tags)
    # else:
    #     keywords = keywords[1] + ' ' + random.choice(gachi_tags)
    title = generate_text(keywords[0], 10, model, random.randint(8, max_sequence_len - 1))
    title = replace_text(title, dict_gachi)
    describe = predict_quantile_ddf(title)
    response = f"Подходящий для вас заголовок для публикации: \n{title.title()} \n \n" + describe
    await bot.send_message(callback_query.from_user.id, response)    # Сейчас бот отправляет полученный список
    # сообщением

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
