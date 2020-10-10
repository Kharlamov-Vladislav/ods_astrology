import hashing_astrology_date
import keywords_dict_for_model


def get_keywords(*dates):
    keywords_from_hashes = []
    for date in dates:
        date = date.strftime('%d.%m.%Y')
        hashed_date = hashing_astrology_date.date_go_hash(date)
        keywords_from_hashes.append(keywords_dict_for_model.keywords_dict[hashed_date])
    return keywords_from_hashes





#print(get_keywords('01.02.1998', '03.05.1997'))
#['accuracy']
