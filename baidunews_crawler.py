import requests
import csv
import re
from pyquery import PyQuery
from bs4 import BeautifulSoup
import chardet



def get_news_content(link):
    """
    根据新闻网址，获取新闻数据
    :return:  新闻内容
    """
    resp = requests.get(link)

    news_text = ''.join(re.findall('[\u4e00-\u9fa5]+', resp.text))
    if not news_text:
        chaset = chardet.detect(resp.content)['encoding']
        resp.encoding = chaset
        news_text = ''.join(re.findall('[\u4e00-\u9fa5]+', resp.text))
        return news_text
    return news_text


def get_keywords_news_links(keyword_link):
    """
    访问关键词百度网址，得到相关新闻的link
    :param keyword_link:
    :return:
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'}
    resp = requests.get(keyword_link, headers=headers)
    bsObj = BeautifulSoup(resp.text, 'html.parser')
    news_items = bsObj.find_all('div', {'class': 'result c-container '})
    news_links = []
    for item in news_items:
        links = re.findall('href="(.*?)"', str(item))
        news_links.extend(links)

    news_links = set([link for link in news_links if 'http://cache.baiducontent' not in link])
    return news_links


#主程序，访问并保存所有的新闻数据
def FetchAndSave():
    #百度风云榜页面网址(含有50个热门新闻的关键词)
    fengyunbang_url = 'http://top.baidu.com/buzz?b=1'
    resp = requests.get(fengyunbang_url)
    resp.encoding='gb2312'

    #新建excel文件保存数据。
    csvf = open('data.csv', 'a+', encoding='gbk', newline='')
    writer = csv.writer(csvf)
    writer.writerow(('news_content', 'keyword'))

    #从heml文件中解析出  事件字段和 网址字段
    doc = PyQuery(resp.text)
    for item in doc.items('.keyword'):
        keyword = item('a').text().split(' ')[0]
        keyword_link = item('a').attr.href
        news_links = get_keywords_news_links(keyword_link)
        for news_link in news_links:
            try:
                content = get_news_content(news_link)
                if content:
                    print(keyword, content[0:20])
                    writer.writerow((content, keyword))
            except:
                print(news_link)



FetchAndSave()

