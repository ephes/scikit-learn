#!/usr/bin/python

import os
import sys
import pickle
import urllib
import tarfile

import numpy as np

try:
    import cElementTree as _et
except ImportError, err:
    import _elementtree as _et
from cStringIO import StringIO

from time import time
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.datasets.base import get_data_home, Bunch
from sklearn.feature_extraction.text import Vectorizer

class ReutersParser(object):
    _broken = ("&#1;", "&#2;", "&#3;", "\xfc", u"\xfc".encode('utf-8'),
               "&#5;", "&#22;", "&#27;", "&#30;", "&#31;", "&#127;")

    def cleanup_sgml(self, chunk):
        for item in self._broken:
            chunk = chunk.replace(item, "")
        chunk = chunk.replace('<!DOCTYPE lewis SYSTEM "lewis.dtd">',
            "<document>")
        return "%s</document>" % chunk

    def get_topics(self, topics):
        return [topic.text for topic in topics]

    def get_text(self, text):
        tagmap = dict.fromkeys(("title", "dateline", "body"))
        for item in text:
            tag = item.tag.lower()
            if tag in tagmap:
                tagmap[tag] = item.text
        return tagmap

    def parse_doc(self, elem):
        doc = {}
        doc["attrs"] = dict(elem.items())
        for item in elem:
            if item.tag == "TOPICS":
                doc["topics"] = self.get_topics(item)
            elif item.tag == "DATE":
                doc["date"] = item.text
            elif item.tag == "TEXT":
                doc.update(self.get_text(item))
        return doc

    def parse_sgml(self, filename):
        stream = StringIO(self.cleanup_sgml(file(filename).read()))
        for _, elem in _et.iterparse(stream):
            if elem.tag == "REUTERS":
                yield self.parse_doc(elem)

class ReutersCorpus(object):
    def __init__(self, raw_docs, multiclass=False):
        self.topics = {}
        self.target_names = []

        self.docs = list(self.get_docs(raw_docs))
        if multiclass:
            self.docs = self.filter_multi_label(self.docs)
        self.docs = self.filter_empty_cats(self.docs)

        # labels have to be without gaps
        self._renumber_topics()

    def _renumber_topics(self):
        self.topics = {}
        self.target_names = []
        for doc in self.docs:
            self._add_topics(doc)

    def _add_text(self, doc):
        #doc["text"] = " ".join([doc.get(tag) or "" for tag in 
        #    ("title", "dateline", "body")])
        doc["text"] = " ".join([doc.get(tag) or "" for tag in
            ("dateline", "body")])
        title = " ".join([doc.get("title") or "" for i in range(1)])
        doc["text"] = "%s %s" % (title, doc["text"])

    def _add_modapte(self, doc):
        attrs = doc["attrs"]
        doc["modapte"] = "unused"
        if attrs["LEWISSPLIT"] == "TRAIN" and attrs["TOPICS"] == "YES":
            doc["modapte"] = "train"
        elif attrs["LEWISSPLIT"] == "TEST" and attrs["TOPICS"] == "YES":
            doc["modapte"] = "test"

    def _add_topics(self, doc):
        doc["cats"] = []
        for topic in doc["topics"]:
            if topic not in self.topics:
                self.target_names.append(topic)
                topic_id = len(self.target_names)
                self.topics[topic] = topic_id
            topic_id = self.topics[topic]
            doc["cats"].append(topic_id)

    def get_docs(self, documents):
        modifiers = [self._add_text, self._add_modapte, self._add_topics]
        for doc in documents:
            for modifier in modifiers:
                modifier(doc)
            if doc["modapte"] != "unused":
                yield doc

    def filter_empty_cats(self, docs):
        # modapte yields 90 categories with 1 train and test doc at least
        train, test = set(), set()
        for doc in docs:
            if doc["modapte"] == "train":
                for cat in doc["cats"]:
                    train.add(cat)
            elif doc["modapte"] == "test":
                for cat in doc["cats"]:
                    test.add(cat)
        valid_cats = train.intersection(test)
        print "valid", len(valid_cats), valid_cats
        new_docs = []
        for doc in docs:
            doc["cats"] = [c for c in doc["cats"] if c in valid_cats]
            if len(doc["cats"]) > 0:
                new_docs.append(doc)
        return new_docs

    def filter_multi_label(self, docs):
        filtered_docs = []
        for doc in docs:
            if len(doc["cats"]) == 1:
                filtered_docs.append(doc)
        return filtered_docs

def download_corpus(data_home, cache_path):
    reuters_path = os.path.join(data_home, "reuters_21758")
    url = "http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz"
    archive_name = os.path.split(url)[-1]
    archive_path = os.path.join(reuters_path, archive_name)

    if not os.path.exists(reuters_path):
        os.makedirs(reuters_path)

    opener = urllib.urlopen(url)
    open(archive_path, 'wb').write(opener.read())
    
    tarfile.open(archive_path, "r:gz").extractall(path=reuters_path)
    rp = ReutersParser()
    sgmlfiles = [os.path.join(reuters_path, i)
        for i in os.listdir(reuters_path) if i.endswith(".sgm")]

    documents = []
    for filename in sgmlfiles:
        for doc in rp.parse_sgml(filename):
            doc["filename"] = filename
            documents.append(doc)

    open(cache_path, 'wb').write(pickle.dumps(documents).encode('zip'))

def get_corpus():
    data_home = get_data_home()
    reuters_cache = os.path.join(data_home, "reuters_21758.pkz")

    if not os.path.exists(reuters_cache):
        download_corpus(data_home, reuters_cache)

    return ReutersCorpus(
        pickle.loads(open(reuters_cache, 'rb').read().decode('zip')),
        multiclass=True,
    )

def split_docs_modapte(reuters_corpus):
    train, test = [], []
    for doc in reuters_corpus.docs:
        if doc["modapte"] == "train":
            train.append(doc)
        elif doc["modapte"] == "test":
            test.append(doc)
    return train, test

def get_bunch(target_names, documents):
    target = []
    text_data = []
    filenames = []
    for doc in documents:
        text_data.append(doc["text"])
        target.append(doc["cats"][0])
        filenames.append(doc["filename"])

    return Bunch(data=text_data,
                 filenames=filenames,
                 target_names=target_names,
                 target=np.array(target),
                 DESCR="reuters 21758")

def benchmark(clf, X_train, y_train, X_test, y_test, categories):
    print 80 * '_'
    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metrics.f1_score(y_test, pred)
    print "f1-score:   %0.3f" % score

    if hasattr(clf, 'coef_'):
        print "dimensionality: %d" % clf.coef_.shape[1]
        print "density: %f" % density(clf.coef_)

        print

    #if opts.print_report:
    if True:
        print "classification report:"
        print metrics.classification_report(y_test, pred,
                                            target_names=categories)

    #if opts.print_cm:
    if False:
        print "confusion matrix:"
        print metrics.confusion_matrix(y_test, pred)

    print
    return score, train_time, test_time
    
def main(args):
    reuters_corpus = get_corpus()
    train, test = split_docs_modapte(reuters_corpus)
    data_train = get_bunch(reuters_corpus.target_names, train)
    print "trainlen", len(data_train.data)
    categories = reuters_corpus.target_names
    data_test = get_bunch(reuters_corpus.target_names, test)
    print len(categories), categories
    print "testlen", len(data_test.data)
    y_train, y_test = data_train.target, data_test.target
    
    vectorizer = Vectorizer()
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    for clf, name in ((RidgeClassifier(tol=1e-1), "Ridge Classifier"),
                  (KNeighborsClassifier(n_neighbors=10), "kNN")):
        print 80 * '='
        print name
        results = benchmark(clf, X_train, y_train, X_test, y_test, categories)

    for penalty in ["l2", "l1"]:
        print 80 * '='
        print "%s penalty" % penalty.upper()
        # Train Liblinear model
        liblinear_results = benchmark(LinearSVC(loss='l2', penalty=penalty, C=1000,
                                                dual=False, tol=1e-3),
                            X_train, y_train, X_test, y_test, categories)

        # Train SGD model
        sgd_results = benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                              penalty=penalty),
                        X_train, y_train, X_test, y_test, categories)

    # Train SGD with Elastic Net penalty
    print 80 * '='
    print "Elastic-Net penalty"
    sgd_results = benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                          penalty="elasticnet"),
                    X_train, y_train, X_test, y_test, categories)

    # Train sparse Naive Bayes classifiers
    print 80 * '='
    print "Naive Bayes"
    mnnb_results = benchmark(MultinomialNB(alpha=.01),
                    X_train, y_train, X_test, y_test, categories)
    bnb_result = benchmark(BernoulliNB(alpha=.01),
                    X_train, y_train, X_test, y_test, categories)


    class L1LinearSVC(LinearSVC):

        def fit(self, X, y):
            # The smaller C, the stronger the regularization.
            # The more regularization, the more sparsity.
            self.transformer_ = LinearSVC(C=1000, penalty="l1",
                                          dual=False, tol=1e-3)
            X = self.transformer_.fit_transform(X, y)
            return LinearSVC.fit(self, X, y)

        def predict(self, X):
            X = self.transformer_.transform(X)
            return LinearSVC.predict(self, X)

    print 80 * '='
    print "LinearSVC with L1-based feature selection"
    l1linearsvc_results = benchmark(L1LinearSVC(),
                            X_train, y_train, X_test, y_test, categories)
    
if __name__ == '__main__':
    main(sys.argv)
