import requests
from bs4 import BeautifulSoup
import pandas as pd

def scr_nat_subj(pagenr, subject):
    URL = (
        f"https://www.nature.com/search?article_type=research%2Creviews&subject={subject}&page={pagenr}"
    )
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all('h2', class_='extra-tight-line-height')
    urls = [url.find('a')['href'] for url in results]
    return urls

subj = 'mathematics-and-computing'

urls = []
#urls.extend(scr_nat_subj(pagenr=1, subject='chemistry'))
for nr in range(1, 2):
    urls.extend(scr_nat_subj(pagenr=nr, subject=subj))
    print(nr)
    if not scr_nat_subj(pagenr=nr, subject=subj):
        break

info_list = []
info_list.append(['article_url','title','abstract','citations','submit_date','publish_date'])

def scr_nat_art(article):
    URL = (
        f"https://www.nature.com{article}"
    )
    print(article)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    title = soup.find('h1', class_='c-article-title').get_text()
    abstract = soup.find(id='Abs1-content').get_text()
    citations = soup.find_all('p', {'class':'c-article-metrics-bar__count'})
    if len(citations) > 1:
        citations = soup.find_all('p', {'class':'c-article-metrics-bar__count'})[1].get_text()
        if 'Altmetric' not in citations:
            citations = citations.replace('Citations ', '')
        else:
            citations = 0
    else:
        citations = 0
    biblio_info = soup.find_all('span', class_='c-bibliographic-information__value')
    submit_date = biblio_info[0].find('time')['datetime']
    if len(biblio_info) <= 3:
        publish_date = biblio_info[1].find('time')['datetime']
    else:
        publish_date = biblio_info[2].find('time')['datetime']
    list = [article, title, abstract, citations, submit_date, publish_date]
    print(citations)
    return list

for i in urls:
    try:
      info_list.append(scr_nat_art(article=i))
    except AttributeError:
        pass

info_df = pd.DataFrame.from_records(info_list)
info_df.columns = info_df.iloc[0]
info_df = info_df[1:].set_index('article_url')

info_df.to_csv('nature_computer_science.csv', encoding='utf-8')
