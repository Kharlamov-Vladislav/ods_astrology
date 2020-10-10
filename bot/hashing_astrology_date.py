import hashlib

keys_value = 30
# принимаем дату, возвращаем байт хеша в виде строки
def date_go_hash(date):
    hash_bdate = hashlib.md5(bytes(date, 'utf-8'))
    return hash_bdate.hexdigest()[keys_value:]

