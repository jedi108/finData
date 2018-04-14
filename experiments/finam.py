from urllib import urlencode
from datetime import datetime
periods = {'tick': 1, 'min': 2, '5min': 3, '10min': 4, '15min': 5, '30min': 6, 'hour': 7, 'daily': 8, 'week': 9, 'month': 10}
FINAM_URL = "http://195.128.78.52/table.csv?"
market = 5
start_date_str = "01.12.2014"
end_date_str = "06.12.2014"
start_date = datetime.strptime(start_date_str, "%d.%m.%Y").date()
end_date = datetime.strptime(end_date_str, "%d.%m.%Y").date()
period = 2
symbol_code = 86
symbol = "GBPUSD"
# Формируем строку с параметрами запроса:
params = urlencode([('market', 5), ('em', symbol_code), ('code', symbol),
                   ('df', start_date.day), ('mf', start_date.month - 1), ('yf', start_date.year),
                   ('from', start_date_str),
                   ('dt', end_date.day), ('mt', end_date.month - 1), ('yt', end_date.year),
                   ('to', end_date_str),
                   ('p', period), ('f', "table"), ('e', ".csv"), ('cn', symbol),
                   ('dtf', 1), ('tmf', 3), ('MSOR', 1), ('mstime', "on"), ('mstimever', 1),
                   ('sep', 3), ('sep2', 1), ('datf', 5), ('at', 1)])

url = FINAM_URL + params # Полная строка адреса со всеми параметрами.
# Соединяемся с сервером, получаем данные и выполняем их разбор:
data = read_csv(url, header=0, index_col=0, parse_dates={'Date&Time': [0, 1]}, sep=';').sort_index()
data.columns = ['' + i for i in ['Open', 'High', 'Low', 'Close', 'Volume']] # Заголовки столбцов
print(data.head())