# -* coding: utf-8 -*-

import urllib
import re

from BeautifulSoup import BeautifulSoup


#url = 'http://www.google.com/'
#url = 'http://www.saksfifthavenue.com/main/ProductDetail.jsp?FOLDER%3C%3Efolder_id=2534374306418051&PRODUCT%3C%3Eprd_id=845524443725813&R=3145891165708&P_name=CHANEL&N=4294907777+306418051+306610898&bmUID=kWWiyG3'
url = 'http://www.violetgrey.com/product/prep-plus-prime-skin-smoother/MAC-MHFK-01?icl=section_2_section_1_section_1_product_grid_1_product_grid_item_1&icn=image_MAC-MHFK-01'
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html,
                     convertEntities=BeautifulSoup.HTML_ENTITIES)
texts = soup.findAll(text=True)

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'noscript']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True

def normalize_text(s):
    return re.sub('\s+', ' ', s).strip()

visible_texts = filter(None, map(normalize_text, filter(visible, texts)))
print visible_texts
