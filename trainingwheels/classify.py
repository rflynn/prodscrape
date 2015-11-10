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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import unicodedata
from joblib import Parallel, delayed
import sys
from copy import deepcopy


from_field = 'merch_product_name'
from_field = 'description'
to_field = 'bamx_product_category_id'

# ref: http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html

def norm(s):
    # ref: http://stackoverflow.com/questions/2365411/python-convert-unicode-to-ascii-without-errors#7782177
    normal = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    return u' '.join(re.findall('\w+', re.sub(r'<\/?\w+>', u' ', normal.lower())))

assert norm(u'\N{LATIN SMALL LETTER E WITH GRAVE}') == u'e'

category = {
    0: u'Unsure',
    1: u'Other',
    2: u'Beauty',
    3: u'Clothing',
    4: u'Handbags',
    5: u'Jewelry',
    6: u'Shoes',
}


def train(df, from_field, train_index, test_index):
    clf, vec = trainfold(df, from_field, train_index)
    return clf, vec, testfold(df, clf, vec, test_index)

def trainfold(df, from_field, train_index):
    '''
    given a dataframe and a kfold train_index
    build a classifier and vectorizer using df[train_index]
    '''

    '''
    # NOTE: this actually works pretty well, but isn't parallelizable...
    vec = TfidfVectorizer(encoding='utf-8',
                            lowercase=True,
                            ngram_range=(1, 1),
                            stop_words='english')
    '''

    vec = HashingVectorizer(decode_error='ignore',
                            encoding='utf-8',
                            lowercase=True,
                            n_features=2 ** 16,
                            ngram_range=(1, 1),
                            non_negative=True,
                            stop_words='english')

    counts = vec.fit_transform(df[from_field][train_index])

    targets = df['bamx_product_category_id'][train_index]

    #classifier = MultinomialNB() # fast notparallel? 96% acc, 5 sec, unsure 10%
    #classifier = svm.SVC(kernel='linear', probability=True) # slow, 98% acc, 150 sec, unsure 5%
    #classifier = svm.SVR(kernel='linear') # can't do multiclass and probability
    classifier = SGDClassifier(loss='modified_huber', n_jobs=8) # fast 98% acc, 3 sec unsure 11%
    #classifier = SGDClassifier(loss='log', n_jobs=8) # fast 97% acc, 27% unsure
    #classifier = RandomForestClassifier(n_jobs=8) # slower, parallelizable, 97% acc, 100MB data, 25s, unsure 8%
    #classifier = svm.LinearSVC() # can't do probability
    classifier.fit(counts, targets)

    return classifier, vec


def testfold(df, classifier, vectorizer, test_index):
    '''
    given a classifier and vectorizer, test it against df[test_index]
    '''

    # test

    cats = df['bamx_product_category_id'][test_index]
    names = df[from_field][test_index]

    predictions = classifier.predict(vectorizer.transform(names))
    predictions_prob = classifier.predict_proba(vectorizer.transform(names))

    #correct = sum(cat == p for cat, p in zip(cats, predictions))
    correct = sum(max(p) <= 0.90 or cat == p.argmax() + 2
                    for cat, p in zip(cats, predictions_prob))

    '''
    for cat, name, p in zip(cats, names, predictions):
        if cat != p:
            print cat, p, name
            #pass
    '''
    print '%d of %d (%3.1f%%)' % (correct, len(test_index), float(correct) * 100 / len(test_index))

    cm = confusion_matrix(cats, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm_normalized


if __name__ == '__main__':

    pprint.pprint(category)

    # bamx_product_category_id

    df = pandas.read_csv('merchant_product_cat_and_description_minus_1.tsv.gz',
                        #'merchant_product_cat_and_description_minus_1.csv.gz',
                         encoding='utf8',
                         compression='gzip',
                         sep='\t')
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
    # train
    for train_index, test_index in kf:
        clf, vec, cm = train(df, from_field, train_index, test_index)
        print cm
    '''
    # train
    p = Parallel(n_jobs=len(kf))(delayed(train)(df, from_field, train_index, test_index)
                    for train_index, test_index in kf)
    print p

    #pdb.set_trace()

    # merge results
    clfs = [clf for clf, _vec, _cm in p]
    vecs = [vec for _clf, vec, _cm in p]

    # merge multiple partial linear classifiers into one
    clen = len(clfs)
    clf = clfs.pop(0)
    vec = vecs[0]

    '''
    # FIXME: this doesn't work...
    for c in clfs:
        #clf.coef_ += c.coef_
        clf.dual_coef_ += c.dual_coef_
        clf.support_vectors_ += c.support_vectors_
        clf.intercept_ += c.intercept_
    clf.coef_ /= clen
    clf.dual_coef_ /= clen
    clf.intercept_ /= clen
    '''

    vec = None
    clf = None

    if os.path.exists('/tmp/vecdump.pickle'):
        with open('/tmp/vecdump.pickle') as f:
            vec = pickle.load(f)

    if os.path.exists('/tmp/clfdump.pickle'):
        with open('/tmp/clfdump.pickle') as f:
            clf = pickle.load(f)

    if not (vec and clf):
        # train on whole dataset...
        clf, vec = trainfold(df, from_field, np.array(xrange(len(df))))

    vecdump = pickle.dumps(vec)
    print 'vec pickled len:', len(vecdump)
    try:
        with open('/tmp/vecdump.pickle', 'w') as f:
            f.write(vecdump)
    except:
        pass

    clfdump = pickle.dumps(clf)
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
    examples = orig_df['description'][:len(df)]
    predictions = clf.predict_proba(vec.transform(examples))

    d = defaultdict(list)
    for p, e in zip(predictions, examples):
        m = max(p)
        #print m, e, p
        cat = p.argmax() + 2 if m > 0.90 else 0 # if unsure, don't use
        d[category[cat]].append((round(m, 3), e))
    for k, v in d.iteritems():
        pprint.pprint((k, v[:5]))
    #pprint.pprint(dict(d))
    print 'len(df):', len(df)
    unsurelen = len(d.get(u'Unsure', []))
    print 'Unsure: %d (%.1f%%)' % (unsurelen, float(unsurelen) / len(df) * 100)

