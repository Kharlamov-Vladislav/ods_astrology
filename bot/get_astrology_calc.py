from flatlib import const
from flatlib.chart import Chart
from flatlib.datetime import Datetime
from flatlib.geopos import GeoPos

import keywords_dict_for_model as kdfm

pos = GeoPos('51n52', '0w11')

def get_datastring(publish_date_text):
    publish_date = Datetime(publish_date_text, '00:00', '+00:00')
    list_obj = [publish_date_text]
    for obj in const.LIST_OBJECTS_TRADITIONAL:
        calculator = Chart(publish_date, pos).objects
        list_obj.append(kdfm.sign_dict[calculator.get(obj).sign])
        list_obj.append(calculator.get(obj).lon)
        list_obj.append(calculator.get(obj).lat)
    return list_obj

