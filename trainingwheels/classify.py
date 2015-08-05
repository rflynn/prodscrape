# -*- encoding: utf-8 -*-

#
# given a [(bamx_category_id, merch_product_name),...]
# use the product name to predict the category using statistical means via scikit
#
# sudo pip install pandas sklearn scipy

# TODO: MIT or CalTech videos online re: machine learning (MIT OpenCourseware?)

import pdb

from collections import defaultdict, Counter
import os
import pandas
import pickle
import pprint
import numpy as np
import re
from sklearn import cross_validation
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import unicodedata


# ref: http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html

def norm(s):
    # ref: http://stackoverflow.com/questions/2365411/python-convert-unicode-to-ascii-without-errors#7782177
    normal = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    return u' '.join(re.findall('\w+', re.sub(r'<\/?\w+>', u' ', normal.lower())))

assert norm(u'\N{LATIN SMALL LETTER E WITH GRAVE}') == 'e'

category = {
    0: u'Unsure',
    1: u'Other',
    2: u'Beauty',
    3: u'Clothing',
    4: u'Handbags',
    5: u'Jewelry',
    6: u'Shoes',
}


def trainfold(df, train_index):
    '''
    given a dataframe and a kfold train_index
    build a classifier and vectorizer using df[train_index]
    '''

    vectorizer = CountVectorizer(encoding='utf-8',
                                 stop_words='english',
                                 ngram_range=(1,1),
                                 lowercase=True)

    counts = vectorizer.fit_transform(df['merch_product_name'][train_index])

    targets = df['bamx_product_category_id'][train_index]

    #classifier = MultinomialNB()
    classifier = svm.SVC(kernel='linear', probability=True)
    #classifier = svm.LinearSVC()
    classifier.fit(counts, targets)

    return classifier, vectorizer


def testfold(df, classifier, vectorizer, test_index):
    '''
    given a classifier and vectorizer, test it against df[test_index]
    '''

    # test

    cats = df['bamx_product_category_id'][test_index]
    names = df['merch_product_name'][test_index]

    predictions = classifier.predict(vectorizer.transform(names))
    #predictions = classifier.predict_proba(vectorizer.transform(names))

    cm = confusion_matrix(cats, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    correct = sum(cat == p for cat, p in zip(cats, predictions))
    '''
    correct = sum(max(p) <= 0.98 or cat == p.argmax() + 2
                    for cat, p in zip(cats, predictions))
    '''

    '''
    for cat, name, p in zip(cats, names, predictions):
        if cat != p:
            print cat, p, name
            #pass
    '''
    print '%d of %d (%3.1f%%)' % (correct, len(test_index), float(correct) * 100 / len(test_index))

    return cm_normalized


if __name__ == '__main__':

    pprint.pprint(category)

    df = pandas.read_csv('merchant_product_cat_and_name_minus_1_fixed1.csv.gz',
                         encoding='utf8',
                         compression='gzip')
    df = df.reindex(np.random.permutation(df.index)).reset_index() # shuffle
    orig_df = df
    print 'len(df):', len(df)

    catcnt = Counter(df['bamx_product_category_id'].values)
    catfewest = min(catcnt.values())
    catmean = sum(catcnt.values()) / len(catcnt)
    cats = len(catcnt)
    catmost = min(catmean, 20000)

    print 'catcnt:', catcnt
    print 'catfewest:', catfewest

    # NOTE: trimming out excessive imbalance seems to help...
    # TODO: there has to be a built-in way to do this?
    print 'reducing imbalance in %d categories...' % cats
    cnts = Counter()
    dropus = []
    for i, catid in enumerate(df['bamx_product_category_id']):
        if cnts[catid] > catmost:
            dropus.append(i)
        else:
            cnts[catid] += 1

    df = df.drop(dropus).reset_index()

    print 'updated cnts:', cnts
    print 'len(df):', len(df)
    assert len(df) == sum(cnts.values())

    # FIXME: for some reason this results in cat 1 not getting used at all? wtf...
    #kf = cross_validation.KFold(len(df))
    kf = cross_validation.StratifiedKFold(df['bamx_product_category_id'].values, shuffle=True)

    np.set_printoptions(suppress=True, precision=2)

    '''
    for train_index, test_index in kf:
        #print 'TRAIN:', len(train_index), train_index, 'TEST:', len(test_index), test_index
        classifier, vectorizer = trainfold(df, train_index)
        cm = testfold(df, classifier, vectorizer, test_index)
        print cm
    '''

    vectorizer = None
    classifier = None

    if os.path.exists('/tmp/vecdump.pickle'):
        with open('/tmp/vecdump.pickle') as f:
            vectorizer = pickle.load(f)

    if os.path.exists('/tmp/clfdump.pickle'):
        with open('/tmp/clfdump.pickle') as f:
            classifier = pickle.load(f)

    if not (vectorizer and classifier):
        # train on whole dataset...
        classifier, vectorizer = trainfold(df, np.array(xrange(len(df))))

    vecdump = pickle.dumps(vectorizer)
    print 'vec pickled len:', len(vecdump)
    try:
        with open('/tmp/vecdump.pickle', 'w') as f:
            f.write(vecdump)
    except:
        pass

    clfdump = pickle.dumps(classifier)
    print 'clf pickled len:', len(clfdump)
    try:
        with open('/tmp/clfdump.pickle', 'w') as f:
            f.write(clfdump)
    except:
        pass

    examples = [
        norm(u'Alexander McQueen-Studded Leather Platform Sandals'),
        norm(u'Elements Punk Labradorite & Crystal Confetti Chandelier Earrings'),
        norm(u'crewcuts Boys Birkenstock Arizona Sandals'),
        norm(u'Sephora: SEPHORA COLLECTION : Rouge Shine Lipstick : lipstick'),
        norm(u'Multicolored cotton-poplin Concealed zip fastening along back 100% cotton Dry clean Designer color: Green'),
        norm(u'Dolce & Gabbana | Portofino printed cotton-poplin dress | NET-A-PORTER.COM'),
        norm(u"'Take Your Marc' Leather Crossbody Bag"),
        norm(u'Aussie 3 Minute Miracle Moist Deep Conditioning Treatment'),
        norm(u"Burberry Medium Banner House Check Tote | Bloomingdale's"),
        norm(u'CHANEL-<b>COCO MADEMOISELLE</b><br>Eau de Parfum Spray'),
        norm(u'M·A·C Prep + Prime Skin Smoother'),
        norm(u'Collection striped popover jacket'),
        norm(u'Telluride Calf Hair Slide Sandals | Lord and Taylor'),
        norm(u'Microsoft Surface 3 10.8" Tablet 64GB Windows 8.1 - Walmart.com'),
        norm(u'Microsoft Surface 3 10.8" Tablet 64GB Windows 8.1'),
    ]
    examples = orig_df['merch_product_name'][0:len(df)]
    predictions = classifier.predict_proba(vectorizer.transform(examples))

    d = defaultdict(list)
    for p, e in zip(predictions, examples):
        m = max(p)
        #print m, e, p
        cat = p.argmax() + 2 if m > 0.80 else 0 # if unsure, don't use
        d[category[cat]].append((round(m, 3), e))
    pprint.pprint(dict(d))

