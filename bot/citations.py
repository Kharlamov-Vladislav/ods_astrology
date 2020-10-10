from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import numpy as np
from pathlib import Path

app_dir: Path = Path(__file__).parent
nature_computer = app_dir / "nature_computer_science.csv"
magic_nature = app_dir / "magic_nature_computer_science.csv"


df = pd.read_csv(nature_computer)
df['publish_date'] = pd.to_datetime(df['publish_date'])
df['article_age'] = (datetime.now()-df['publish_date']).dt.days
df['citations'] = df['citations']/df['article_age']

df1 = pd.read_csv(magic_nature, sep=';')
citations = df['citations']
star_features = df1.drop(['citations', 'submit_date'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(star_features, citations, test_size=0.1, random_state=0)

iso = IsolationForest(contamination=0.1)
iso_res = iso.fit_predict(X_train)
indices = np.argwhere(iso_res==-1)
indices = indices.reshape((len(indices),))

X_train = X_train.drop(X_train.index[indices], axis=0)
y_train = y_train.drop(y_train.index[indices], axis=0)
X_train = X_train.filter(regex='sign_number')
X_test = X_test.filter(regex='sign_number')

decision_tree = DecisionTreeRegressor(max_depth=100)
decision_tree.fit(X_train, y_train)

def citation_text(user_submit_features):
    del user_submit_features[0]
    user_submit_planets = pd.DataFrame(user_submit_features).transpose()
    user_submit_planets.columns = star_features.columns
    user_submit_planets = user_submit_planets.filter(regex='sign_number').astype(int)
    citations_day = decision_tree.predict(user_submit_planets).astype(float)
    citation_number = abs(citations_day * 365)
    decision_path = decision_tree.decision_path(user_submit_planets)
    i=1
    for node_id in decision_path.indices:
        if i == 1:
            col_name = user_submit_planets.columns[decision_tree.tree_.feature[node_id]]
            node1_str = 'Так как в день подачи публикации '+str(col_name)+' в '+str(int(user_submit_planets[col_name].values)).zfill(2)
        if i == 2:
            col_name = user_submit_planets.columns[decision_tree.tree_.feature[node_id]]
            node2_str = ' , '+str(col_name)+' в '+str(int(user_submit_planets[col_name].values)).zfill(2)
        if i == 3:
            if col_name == user_submit_planets.columns[decision_tree.tree_.feature[2]]:
                i == 4
                col_name = user_submit_planets.columns[decision_tree.tree_.feature[4]]
            else:
                col_name = user_submit_planets.columns[decision_tree.tree_.feature[3]]
            node3_str = ' , a '+str(col_name)+' в '+str(int(user_submit_planets[col_name].values)).zfill(2)
        i = i+1
    tree_prediction = node1_str + node2_str + node3_str
    tree_prediction = tree_prediction.replace('_sign_number', '')

    star_signs = {'01':'Овне','02':'Тельце','03':'Близнецах','04':'Раке','05':'Льве','06':'Деве','07':'Весах','08':'Скорпионе','09':'Стрельце','10':'Козероге', '11':'Водолее', '12':'Рыбах'}
    planets = {'Mercury':'Меркурий','Venus':'Венера','Mars':'Марс','Jupiter':'Юпитер','Saturn':'Сатурн','Uranus':'Уран','Neptune':'Нептун','Pluto':'Плутон','Moon':'Луна','Sun':'Солнце', 'North':'Голова', 'South': 'Хвост', 'Node':'Дракона', 'Syzygy':'Сизигия', 'Pars':'Часть', 'Fortuna': 'Фортуны'}

    tree_prediction = ' '.join([planets.get(i, i) for i in tree_prediction.split()])
    tree_prediction = ' '.join([star_signs.get(i, i) for i in tree_prediction.split()])
    tree_prediction = tree_prediction.replace(' ,', ',')

    tree_prediction = tree_prediction + ', через год после выпуска ваша статья будет цитирована ' + str(int(citation_number))
    if '12' in tree_prediction[-2:] or '13' in tree_prediction[-2:] or '14' in tree_prediction[-2:]:
        final_prediction = tree_prediction + ' раз'
    elif '2' in tree_prediction[-1:] or '3' in tree_prediction[-1:] or '4' in tree_prediction[-1:]:
        final_prediction = tree_prediction + ' раза'
    else:
        final_prediction = tree_prediction + ' раз'
    return final_prediction

