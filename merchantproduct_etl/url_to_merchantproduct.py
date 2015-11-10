# -* coding: utf-8 -*-
# ex: set ts=4 et:

'''
investigate extracting product data from merchant sites via various means
this does NOT use merchant- or site-specific hacks. that path is seductive but futile
#
TODO: Next step: predict bamx_product_category_id
TODO: a real system would separate the doc fetching from extraction
TODO: access UPC database/api: interesting: upcindex.com, upcitemdb.com, google.com, etc.

The merchants we're most interested in are...
Sephora
Net-a-porter
Nordstrom
Ulta
Drugstore.com
Bloomingdales
'''

from collections import Counter
from decimal import Decimal
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from urlparse import urlparse, urljoin
import json
import numpy as np
import pdb
import pickle
import pprint
import random
import re
import unicodedata
import urllib2

from BeautifulSoup import BeautifulSoup


class SoupUtil(object):

    @staticmethod
    def url_abs(soup, page_url, url):
        base = soup.find('base')
        base_url = dict(base.attrs).get('href', page_url) if base else page_url
        return urljoin(base_url, url)

    @staticmethod
    def img_src_abs(soup, page_url, img):
        return SoupUtil.url_abs(soup, page_url, dict(img.attrs).get('src'))

    @staticmethod
    def strip_tags(txt):
        return u' '.join(BeautifulSoup(txt).findAll(text=True))


class MerchantProductPartial(object):

    domain_to_merch_name = {
        u'barneys.com': u'barneys',
        u'birchbox.com': u'birchbox',
        u'chanel.com': u'chanel',
        u'intothegloss.com': u'into the gloss',
        u'macys.com': u'macys',
        u'maybelline.com': u'maybelline',
        u'net-a-porter.com': u'net-a-porter',
        u'nordstrom.com': u'nordstrom',
        u'sokoglam.com': u'soko glam',
        u'shop.nordstrom.com': u'nordstrom',
        u'walgreens.com': u'walgreens',
        u'www.birchbox.com': u'birchbox',
        u'www.drugstore.com': u'drugstore.com',
        u'www.intothegloss.com': u'into the gloss',
        u'www.jcrew.com': u'jcrew',
        u'www.lordandtaylor.com': u'lord and taylor',
        u'www.macys.com': u'macys',
        u'www.net-a-porter.com': u'net-a-porter',
        u'www.saksfifthavenue.com': u'saks',
        u'www.sephora.com': u'sephora',
        u'www.violetgrey.com': u'violet grey',
        u'www.walmart.com': u'walmart',
        u'www1.macys.com': u'macys',
    }

    merch_name_canonical = {
        u'birchbox united states': u'birchbox',
    }

    # FIXME: ids not real
    merch_name_to_id = {
        u'barneys': 20,
        u'barneys new york': 20,
        u'birchbox': 15,
        u'birchbox united states': 15,
        u"bloomingdale's": 1,
        u'chanel': 18,
        u'drugstore.com': 2,
        u'into the gloss': 13,
        u'j.crew': 3,
        u'jcrew': 3,
        u'lord and taylor': 4,
        u'macys': 12,
        u'maybelline': 22,
        u'net-a-porter': 5,
        u'neiman marcus': 16,
        u'nordstrom': 6,
        u'peach & lily': 23,
        u'saks fifth avenue': 7,
        u'saks': 8,
        u'sephora': 9,
        u'soko glam': 17,
        u'tatcha': 21,
        u'target': 19,
        u'urban outfitters': 22,
        u'violet grey': 10,
        u'walgreens': 14,
        u'walmart': 11,
    }

    def __init__(self):
        self._bamx_product_category_id = None
        self._brand = None
        self._color = None
        self._currency = None
        self._description = None
        self._gtin_13 = None
        self._gtin_14 = None
        self._gtin_8 = None
        self._image_url = None
        self._is_in_stock = None
        self._merch_id = None
        self._merch_product_name = None
        self._merchant_name = None
        self._merchant_product_match_key = None
        self._model = None
        self._price = None
        self._store_page_url = None
        self._upc = None

    def __repr__(self):
        return (u'MerchantProductPartial (is_viable:%s):\n'
            '    bamx_prod_cat....%s\n'
            '    brand............%s\n'
            '    color............%s\n'
            '    currency.........%s\n'
            '    description......%s\n'
            '    gtin_8...........%s\n'
            '    gtin_13..........%s\n'
            '    gtin_14..........%s\n'
            '    image............%s\n'
            '    is_in_stock......%s\n'
            '    merch_id.........%s\n'
            '    merchant_name....%s\n'
            '    merch_prod_name..%s\n'
            '    merch_prod_mk....%s\n'
            '    model............%s\n'
            '    price............%s\n'
            '    store_page_url...%s\n'  % (
                self.is_viable(),
                self.bamx_product_category_id,
                self.brand,
                self.color,
                self.currency,
                repr(self.description),
                self._gtin_8,
                self._gtin_13,
                self._gtin_14,
                self.image_url,
                self.is_in_stock,
                self.merch_id,
                self.merchant_name,
                self.merch_product_name,
                self.merchant_product_match_key,
                self.model,
                self.price,
                self.store_page_url)).encode('utf8')

    def is_viable(self):
        return bool(
            self.merch_id
            and (self.is_in_stock is not False)
            and (self.bamx_product_category_id is not None)
        )

    @property
    def bamx_product_category_id(self):
        return self._bamx_product_category_id

    @bamx_product_category_id.setter
    def bamx_product_category_id(self, x):
        self._bamx_product_category_id = x

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, x):
        if x:
            if isinstance(x, basestring) and x.lower() != u'null':
                self._color = x

    @property
    def gtin_8(self):
        return self._gtin_8

    @gtin_8.setter
    def gtin_8(self, x):
        self._gtin_8 = x

    @property
    def gtin_13(self):
        return self._gtin_13

    @gtin_13.setter
    def gtin_13(self, x):
        self._gtin_13 = x

    @property
    def gtin_14(self):
        return self._gtin_14

    @gtin_14.setter
    def gtin_14(self, x):
        self._gtin_14 = x

    @property
    def brand(self):
        return self._brand

    @brand.setter
    def brand(self, x):
        self._brand = x

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, x):
        self._model = x

    @property
    def currency(self):
        return self._currency

    @currency.setter
    def currency(self, x):
        self._currency = x

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, x):
        if x:
            self._price = MerchantProductPartial.parse_price(x)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, x):
        if x:
            self._description = SoupUtil.strip_tags(x)

    @property
    def merchant_name(self):
        return self._merchant_name

    @merchant_name.setter
    def merchant_name(self, x):
        if x:
            self._merchant_name = x.lower()
            # canonicalize
            self._merchant_name = MerchantProductPartial.merch_name_canonical.get(self._merchant_name, self._merchant_name)
            self._merch_id = MerchantProductPartial.merch_name_to_id.get(self._merchant_name)

    @property
    def is_in_stock(self):
        return self._is_in_stock

    @is_in_stock.setter
    def is_in_stock(self, x):
        if x is True or x is False:
            self._is_in_stock = x
        elif x:
            self._is_in_stock = x.lower() in (u'instock', u'in stock')

    @property
    def image_url(self):
        return self._image_url

    @image_url.setter
    def image_url(self, x):
        self._image_url = x

    @property
    def merch_id(self):
        return self._merch_id

    @merch_id.setter
    def merch_id(self, x):
        self._merch_id = x

    @property
    def merch_product_name(self):
        return self._merch_product_name

    @merch_product_name.setter
    def merch_product_name(self, x):
        if x is not None:
            #if self.brand:
            #    x = self.strip_prefix_suffix(x, self.brand)
            #if self.merchant_name:
            #    x = self.strip_prefix_suffix(x, self.merchant_name)
            self._merch_product_name = SoupUtil.strip_tags(x)

    @classmethod
    def strip_prefix_suffix(cls, s, x):
        sl = s.lower()
        xl = x.lower()
        if sl.startswith(xl):
            s = s[len(xl):].lstrip()
        if sl.endswith(xl):
            s = s[:-len(xl)].rstrip()
        return s

    @property
    def merchant_product_match_key(self):
        return self._merchant_product_match_key or self.upc

    @merchant_product_match_key.setter
    def merchant_product_match_key(self, x):
        self._merchant_product_match_key = x

    @property
    def store_page_url(self):
        return self._store_page_url

    @store_page_url.setter
    def store_page_url(self, x):
        self._store_page_url = x

    @property
    def upc(self):
        return self._upc

    @upc.setter
    def upc(self, x):
        if x:
            if len(x) > 14:
                raise ValueError(x)
            elif len(x) == 14:
                self.gtin_14 = x
            elif len(x) == 13:
                self.gtin_13 = x
            elif len(x) == 12:
                self.gtin_13 = '0' + x
            else:
                raise ValueError(x)

    @staticmethod
    def parse_price(s):
        if s:
            try:
                return Decimal(s)
            except:
                m = re.search(r'[$](\d{1,2},\d{3}|\d{1,3}(?:\.\d{2})?)', s, re.UNICODE)
                if m:
                    return Decimal(m.groups()[0])
        return None

def user_agent():
    return random.choice(
        [
            'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2226.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0',
            'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10136',
        ])


def make_vectorizer():
    try:
        with open('/tmp/vecdump.pickle') as f:
            return pickle.load(f)
    except:
        pass


def make_classifier():
    try:
        with open('/tmp/clfdump.pickle') as f:
            return pickle.load(f)
    except:
        pass


def soupify_url(url):
    try:
        req = urllib2.Request(url,
                              headers={'User-Agent': user_agent()})
        html = urllib2.urlopen(req).read()
    except (urllib2.HTTPError, urllib2.URLError) as e:
        print e
        return None
    return BeautifulSoup(html, convertEntities=BeautifulSoup.HTML_ENTITIES)


def string_is_repetitive(s):
    c = Counter(zip(s, s[1:]))
    return len(c) <= len(s) / 3


def gtin_is_valid(maybe_gtin_str):
    '''
    determine whether a given possible GTIN/UPC is valid
    this specific method is fragile and stupid, but hey, it's something
    ideally, we should get API access and/or maintain our own db
    '''
    try:
        with open('upc-cache.json') as f:
            j = json.load(f)
            if maybe_gtin_str in j:
                return j[maybe_gtin_str]
    except IOError:
        j = dict()
    is_valid = ((not string_is_repetitive(maybe_gtin_str))
                 and (u'UPC number %s is associated with' % maybe_gtin_str in
                    unicode(soupify_url('http://www.upcindex.com/%s' % maybe_gtin_str))))
    j[maybe_gtin_str] = is_valid
    with open('upc-cache.json', 'w') as f:
        f.write(json.dumps(j))
    return is_valid


def to_string(s):
    if isinstance(s, list):
        x = u' '.join(s).strip()
        return x if x else None
    elif isinstance(s, basestring):
        return s
    elif s is None:
        return None
    else:
        return unicode(s)


def meta_attrs(tag):
    d = dict(tag.attrs)
    return d.get(u'property'), d.get(u'content')


def try_meta_og(soup, mp):
    '''
    ref: http://ogp.me/
    ref: https://developers.facebook.com/docs/payments/product
    '''

    og = dict(meta_attrs(o)
                for o in soup.findAll('meta',
                    attrs={'property': re.compile('^og:')}))

    if og:
        #if True or str(og.get(u'og:type')).lower() == 'product':
        # conceptually we should limit to
        if not mp.merchant_name:
            mp.merchant_name = og.get(u'og:provider_name') or og.get(u'og:site_name')
        if mp.is_in_stock is None:
            if og.get(u'og:availability'):
                mp.is_in_stock = og[u'og:availability'].lower() in (u'instock', u'in stock')
        if not mp.brand:
            mp.brand = og.get(u'og:brand')
        if not mp.currency:
            mp.currency = og.get(u'og:price:currency')
        if not mp.description:
            mp.description = og.get(u'og:description')
        if not mp.image_url:
            mp.image_url = og.get(u'og:image')
        if not mp.merch_product_name:
            mp.merch_product_name = og.get(u'og:title')
        if not mp.merchant_product_match_key:
            mp.merchant_product_match_key = og.get(u'og:product_id')
        if not mp.price:
            mp.price = og.get(u'og:price:amount') or og.get(u'og:price:standard_amount')
        if not mp.store_page_url:
            mp.store_page_url = og.get(u'og:url')
        if not mp.upc:
            mp.upc = og.get(u'og:upc')

    return mp, og


def try_meta_product(soup, mp):

    product = dict(meta_attrs(o)
                    for o in soup.findAll('meta',
                        attrs={'property': re.compile('^product:')}))

    if product:
        if mp.is_in_stock is None:
            mp.is_in_stock = product.get(u'product:availability')
        if not mp.brand:
            mp.brand = product.get(u'product:brand')
        if not mp.color:
            mp.color = product.get(u'product:color')
        if not mp.price:
            mp.price = product.get(u'product:price:amount')
        if not mp.currency:
            mp.currency = product.get(u'product:price:currency')
        if not mp.merchant_product_match_key:
            mp.merchant_product_match_key = product.get(u'product:retailer_item_id') or product.get(u'product:retailer_part_no')
        if not mp.merchant_name:
            mp.merchant_name = product.get(u'product:retailer')

    return mp, product


def try_schemaorg_product(soup, mp):
    '''
    ref: https://en.wikipedia.org/wiki/Microdata_(HTML)
    ref: http://www.schema.org/Product
    ref: https://developers.google.com/structured-data/rich-snippets/products
    ref: https://support.google.com/merchants/answer/6069143?hl=en
    '''

    schemaproduct = soup.findAll(True,
                        attrs={'itemtype': re.compile('^http://schema.org/Product')})

    def producttag(tag):
        attrs = dict(tag.attrs)
        return {
            attrs.get(u'itemprop'):
                (attrs.get(u'content') if tag.name == u'meta' else
                    [t.strip() for t in tag.findAll(text=True) if t.strip()])
        }

    schemaproductdict = {}
    for d in [producttag(tag)
        for tag in schemaproduct + [item for s in schemaproduct for item in s.findChildren()]
            if u'itemprop' in dict(tag.attrs)]:
        schemaproductdict.update(d)

    if schemaproductdict:
        if mp.brand is None:
            mp.brand = to_string(schemaproductdict.get(u'brand') or schemaproductdict.get(u'product:brand'))
        if not mp.is_in_stock:
            mp.is_in_stock = to_string(schemaproductdict.get(u'availability') or schemaproductdict.get(u'product:availability'))
        if not mp.model:
            mp.model = schemaproductdict.get(u'model')
        if not mp.merch_product_name:
            mp.merch_product_name = to_string(schemaproductdict.get(u'name'))
        if not mp.currency:
            mp.currency = to_string(schemaproductdict.get(u'priceCurrency'))
        if not mp.price:
            mp.price = to_string(schemaproductdict.get(u'price'))
        if not mp.upc:
            try:
                mp.upc = to_string(schemaproductdict.get(u'gtin14')
                                or schemaproductdict.get(u'gtin13')
                                or schemaproductdict.get(u'gtin12')
                                or schemaproductdict.get(u'gtin8')
                                or schemaproductdict.get(u'productID')
                                or schemaproductdict.get(u'mpn'))
            except ValueError:
                pass
        if not mp.merchant_product_match_key:
            mp.merchant_product_match_key = to_string(schemaproductdict.get(u'sku') or schemaproductdict.get(u'mpn'))

    return mp, schemaproductdict


def try_meta_twitter(soup, mp):
    '''
    twitter microdata
    looks similar to "og:" but is less common; still worth supporting
    '''

    twit = dict(meta_attrs(o)
                for o in soup.findAll('meta',
                    attrs={'property': re.compile('^twitter:')}))

    if twit:
        if not mp.image_url:
            mp.image_url = twit.get(u'twitter:image')
        if not mp.merch_product_name:
            mp.merch_product_name = twit.get(u'twitter:title')
        if not mp.description:
            mp.description = twit.get(u'twitter:description')

    return mp, twit


def extract_text_patterns(soup, mp):
    '''
    explore extracting patterns... bleh...
    TODO: this could be its own module...
    '''

    doctxt = unicode(soup)

    possible_gtins = Counter(re.findall(r'\b[0-9]{12,14}\b', doctxt))
    possible_identifiers = Counter([i for i in re.findall(r'\b([0-9A-Z]{6,15}|[a-fA-F0-9]{8}-[a-fA-F0-9]{3}-[a-fA-F0-9]{3}-[a-fA-F0-9]{3}-[a-fA-F0-9]{12})\b', doctxt)
                                        if re.search(r'[0-9]', i)]) - possible_gtins
    # ref: https://en.wikipedia.org/wiki/ISO_4217
    possible_prices = Counter(re.findall(r'[$£€¥](?:[0-9]{1,2},)?[0-9]{1,3}(?:\.[0-9]{2})?', doctxt))

    if possible_gtins:
        # if one of the most frequently-mentioned gtins looks legit, use it, why not...
        try_gtins = sorted(possible_gtins.keys(),
                           key=lambda k: possible_gtins[k],
                           reverse=True)[:2]
        for g in try_gtins:
            if gtin_is_valid(g):
                mp.upc = g
                break

    if not mp.merchant_product_match_key and possible_identifiers:
        path_tokens = re.split(r'\W+', url)
        try_idents = sorted(possible_identifiers.keys(),
                            key=lambda k: possible_identifiers[k],
                            reverse=True)[:5]
        for i in try_idents:
            if i in path_tokens:
                mp.merchant_product_match_key = i
                break

        if not mp.merchant_product_match_key:
            if mp.image_url:
                try:
                    parsed_image_url = urlparse(mp.image_url)
                    image_url_tokens = re.split(r'\W+', parsed_image_url.path)
                    for i in try_idents:
                        if i in image_url_tokens:
                            mp.merchant_product_match_key = i
                            break
                except:
                    pass

    is_in_stock = None
    if mp.is_in_stock is None:
        if re.search(r'in stock', doctxt, re.IGNORECASE):
            if not re.search(r'out of stock', doctxt, re.IGNORECASE):
                is_in_stock = True
        mp.is_in_stock = is_in_stock

    patterns = {
        'possible_gtins': possible_gtins,
        'possible_identifiers': possible_identifiers,
        'possible_prices': possible_prices,
        'is_in_stock': is_in_stock,
    }

    return mp, patterns


def page_level_data(soup, mp, up):

    # calculate metadata

    page = {
        'title': soup.title.text,
        'canonical_url': None,
        'meta_description': None,
        'domain_to_merch_name': MerchantProductPartial.domain_to_merch_name.get(up.netloc) if up.netloc else None,
    }

    # favor meta "canonical" url over anything else if it exists
    try:
        canonical = soup.find('link', rel='canonical')
        try:
            page['canonical_url'] = dict(canonical.attrs)[u'href']
        except Exception as e:
            print e
    except Exception as e:
        print e

    meta_content = None
    try:
        page['meta_description'] = dict(soup.find('meta', attrs={u'name': u'description'}).attrs)[u'content']
    except Exception as e:
        print e

    # apply

    if page['canonical_url']:
        mp.store_page_url = page['canonical_url']

    # fallback to page-level metadata if we're missing key fields
    if not mp.merch_product_name:
        mp.merch_product_name = soup.title.text
    if not mp.description and page['meta_description']:
        mp.description = page['meta_description']
    if not mp.merchant_name and page['domain_to_merch_name']:
        mp.merchant_name = page['domain_to_merch_name']

    return mp, page


def norm(s):
    # ref: http://stackoverflow.com/questions/2365411/python-convert-unicode-to-ascii-without-errors#7782177
    normal = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    return u' '.join(re.findall('\w+', re.sub(r'<\/?\w+>', u' ', normal.lower())))


category = {
    0: u'Unsure',
    1: u'Other',
    2: u'Beauty',
    3: u'Clothing',
    4: u'Handbags',
    5: u'Jewelry',
    6: u'Shoes',
}

def prepare_product_name(mp, prodname):
    prodname = norm(prodname).split(u' ')
    if not prodname:
        return None
    if prodname[-1] == u'com':
        prodname.pop()
    if mp.brand:
        brand = norm(mp.brand).split(u' ')
        if prodname[:len(brand)] == brand:
            prodname = prodname[len(brand):]
        if prodname[-len(brand):] == brand:
            prodname = prodname[0:-len(brand)]
    return u' '.join(prodname)

def guess_category(mp, clf, vec):
    if clf and vec and mp.merch_product_name:
        prodname = prepare_product_name(mp, mp.merch_product_name)
        prob = clf.predict_proba(vec.transform([prodname]))[0]
        p = max(prob)
        if p > 0.80:
            i = prob.argmax() + 2
            mp.bamx_product_category_id = category[i]
        print 'prob:', prob, 'p:', p
    return mp


def url_to_mp(url, clf, vec):
    '''
    given an arbitrary url
        fetch its contents
        convert contents into possible MerchantProductPartial object
    '''

    print url

    up = urlparse(url)
    print up

    if 'bam-x.com' in up.netloc:
        return None

    soup = soupify_url(url)
    if not soup:
        return None

    #pdb.set_trace()

    ### meta data

    mp = MerchantProductPartial()

    pprint.pprint({
        u'title': soup.title.text,
        u'description': dict(soup.find('meta', attrs={u'name':u'description'}).attrs)[u'content'],
    })

    ## canonical page-level info
    # ...we also fall back to other, weaker, page-level info at the end

    # scan page for various standard and semi-standard microdata formats
    mp, og = try_meta_og(soup, mp)
    mp, product = try_meta_product(soup, mp)
    mp, schemaorg_product = try_schemaorg_product(soup, mp)
    mp, twit = try_meta_twitter(soup, mp)
    mp, page = page_level_data(soup, mp, up)
    mp, patterns = extract_text_patterns(soup, mp)

    mp = guess_category(mp, clf, vec)

    pprint.pprint(og)
    pprint.pprint(product)
    pprint.pprint(schemaorg_product)
    pprint.pprint(twit)
    pprint.pprint(page)
    pprint.pprint(patterns)

    return mp


np.set_printoptions(suppress=True, precision=2)

clf = make_classifier()
vec = make_vectorizer()

urls = [
    'https://www.birchbox.com/shop/sedu-6000i-blow-dryer',
    #'http://www.neimanmarcus.com/Aquazzura-Christy-Lace-Up-Pointed-Toe-Flat-Aquazzura/prod182720184_cat47570743__/p.prod?icid=&searchType=EndecaDrivenCat&rte=%252Fcategory.jsp%253FitemId%253Dcat47570743%2526pageSize%253D29%2526No%253D0%2526refinements%253D&eItemId=prod182720184&cmCat=product',
    #'http://www.saksfifthavenue.com/main/ProductDetail.jsp?PRODUCT%3C%3Eprd_id=845524446864324&R=400876898709&P_name=Tome&Ntt=tome&N=0&bmUID=l3o8jS1',
    #'http://www.sephora.com/mia-skin-cleansing-system-P285163?skuId=1311976&om_mmc=ppc-GG&mkwid=sMa5HlUsG&pcrid=50233217079&pdv=c&site=_search&country_switch=&lang=en&gclid=CMnbqdjaj8cCFcwXHwodyXkNOA',
]
'''
    'http://www.sephora.com/tinted-brow-gel-P187202',
    'http://shop.nordstrom.com/s/lancer-skincare-sheer-fluid-sun-shield-spf-30/3565107',
    'http://www.drugstore.com/herbacin-kamille-paraben-free-hand-cream/qxp336897',
    'http://www.ulta.com/ulta/browse/productDetail.jsp?productId=xlsImpprod3670103',
    'http://www.amazon.com/Fingernail-Mask-Moisture-Nourish-masks/dp/B00QKSRL44',
    'http://www.net-a-porter.com/us/en/product/511820?cm_mmc=LinkshareUS-_-QFGLnEolOWg-_-ProductSearch-_-us-_-Body_Moisturizer-_-Cr%C3%83%C2%A8me&gclid=CLrOzNjZzsUCFVURHwodN58ATA&siteID=QFGLnEolOWg-Jmi2gtrcHC_Y4LYG0kav6w',
    "http://www.barneys.com/Kiehl's-Since-1851-Facial-Fuel-Eye-De-Puffer-500449556.html?gclid=CPqi0tGZ9MYCFYMWHwodm7AIFA",
    'http://www.target.com/p/nuxe-huile-prodigieuse-multi-purpose-dry-oil-50-ml/-/A-16625092',
    'http://www.violetgrey.com/product/la-laque-nail-lacquer/YSL-L22152?icl=section_2_section_1_section_1_product_grid_1_product_grid_item_1&icn=image_YSL-L22152',
    'http://www1.macys.com/shop/product/bobbi-brown-tinted-moisturizer-broad-spectrum-spf-15?ID=558139',
    'http://www.arcona.com/product/8900.html',
    'http://www1.bloomingdales.com/shop/product/armani-luminous-silk-foundation?ID=149955','http://www.chanel.com/en_US/fragrance-beauty/Fragrance-N%C3%82%C2%B05-N%C3%82%C2%B05-88181',
    'http://sokoglam.com/collections/missha/products/time-revolution-first-treatment-essence',
    'http://www.neimanmarcus.com/CHANEL-b-LE-BLUSH-CR-200-ME-DE-CHANEL-b-br-Cream-Blush/prod161240173/p.prod',
    'https://www.birchbox.com/shop/brands/juliette-has-a-gun?utm_source=linkshare&utm_medium=affiliate&utm_campaign=QFGLnEolOWg&siteID=QFGLnEolOWg-zBso0nAhdYKdEsqnmnlpew',
    'http://www.dermstore.com/product_Vitamin+C+Face+Wash_53084.htm',
    'http://www.walgreens.com/store/c/brylcreem-original-hair-groom/ID=prod8649-product',
    'http://intothegloss.com/2015/04/best-drugstore-mascara/',
    'https://www.tatcha.com/shop/moisture-rich-silk-cream',
    'http://www.saksfifthavenue.com/main/ProductDetail.jsp?PRODUCT%3C%3Eprd_id=845524446826872&R=3348901250696&P_name=Dior&sid=14EDB2159CC0&Ntt=dior+addict&N=0&bmUID=kXsCpxu',
    'http://www.urbanoutfitters.com/urban/catalog/productdetail.jsp?id=32566127&cm_mmc=CJ-_-Affiliates-_-rewardStyle-_-11292048',
    'http://www.maybelline.com/Products/Eye-Makeup/Mascara/great-lash-washable-mascara.aspx',
    'http://peachandlily.com/products/cremorlab-herb-tea-blemish-minus-calming-mask',
    'http://www.shuuemura-usa.com/product/4935421374675,default,pd.html',
    'http://us.caudalie.com/shop-products/collections/vinexpert/vinexpert-firming-serum.html',
    'https://www.jcrew.com/womens_category/outerwear/collection/PRD~C5451/C5451.jsp?srcCode=AFFI00001&siteId=QFGLnEolOWg-blRLLKnl3KH%2FYu20afZcIQ',
    'http://www.lordandtaylor.com/webapp/wcs/stores/servlet/en/lord-and-taylor/shoes/telluride-leather-and-halfcair-slide-sandals',
    'http://www.walmart.com/ip/44992748?findingMethod=wpa&cmp=-1&pt=hp&adgrp=-1&plmt=1145x345_B-C-OG_TI_8-20_HL_MID_HP&bkt=&pgid=0&adUid=16c32ae1-91ac-4944-9cd1-a66c46719a7d&adpgm=hl',
]
'''
for url in urls:
    print url_to_mp(url, clf, vec)
