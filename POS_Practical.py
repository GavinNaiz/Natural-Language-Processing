# -*- coding: utf-8 -*-

from __future__ import division
import nltk
from cPickle import load

"""Ex.2"""

# count tokens and types in untagged corpus
#full_untagged = (open('da-korpus-untagged.txt', 'rU').read()).split()
text = "John met Peter in the court one week ago.\nAnn met him in the parliament yesterday.\nShe waved at him repeatedly.\nThat morning, Peter was discussing with three English lobbyists.\nHe did not notice his friend."
full_untagged = text.split("\n")

print 'words in untagged corpus: ', len(full_untagged)
print " "
print 'word types in untagged corpus', len(set(full_untagged))
print " "
# count tokens and types in tagged corpus
full_tagged = (open('da-korpus-tagged.txt', 'rU').read()).split()

print 'words in tagged corpus: ', len(full_tagged)
print " "
print 'word types in tagged corpus', len(set(full_tagged))
print " "

# token / tag tuples
input = open('da_tagged_words.pkl', 'rb')
tagged_words = load(input)
input.close
# inspecting:
print tagged_words[:10]
print " "

# get word and tag frequency
tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_words)
print tag_fd
print " "
word_fd = nltk.FreqDist(word for (word, tag) in tagged_words)
print word_fd
print " " 

"""Ex.3 Split training and test sets"""
input = open('da_tagged_sents.pkl')
tagged_sents = load(input)
input.close
print " "

split = int(len(tagged_sents)*0.9)

train = tagged_sents[:split]

test = tagged_sents[split:]

"""Ex.4 Taggers"""

"""Default tagger"""
#find most common Danish tag
tags = [tag for (word, tag) in tagged_words]
print "Most likely tag: ", nltk.FreqDist(tags).max()
print " "
default_tagger = nltk.DefaultTagger('TEGN')
default_tagger.tag(train)
print "Default tagger: ", default_tagger.evaluate(test)
print " "


"""Regular expression tagger"""
"""patterns = [
    (r'*ede$', 'V_PAST')
    (r'.^[A-ZÅÆØ][a-zåæø]+', 'EGEN')
    (r'*er$', 'V_PRES')
    (r'*erne$', 'N_DEF_PLU')
    (r'.^-?[0-9]+(.[0-9]+)?$', 'NUM')
    (r'.*', 'TEGN')
]"""


"""Lookup tagger"""
# find 100 most frequent words and store their most likely tag
fd = nltk.FreqDist(full_untagged)
cfd = nltk.ConditionalFreqDist(tagged_words)
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)


baseline_tagger = nltk.UnigramTagger(model=likely_tags)
print "LU tagged: ", baseline_tagger.evaluate(tagged_sents)
print "LU test: ", baseline_tagger.evaluate(test)
print " "

baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('TEGN'))
print "LU tagged w/ backoff: ", baseline_tagger.evaluate(tagged_sents)
print " "


"""Unigram tagger"""
unigram_tagger = nltk.UnigramTagger(train)
print "Unigram: ", unigram_tagger.evaluate(test)
print " "


"""Bigram tagger"""
bigram_tagger = nltk.BigramTagger(train)
print "Bigram: ", bigram_tagger.evaluate(test)
print " "


"""Trigram tagger with backoffs"""
t0 = nltk.DefaultTagger('TEGN')
t1 = nltk.UnigramTagger(train, backoff=t0)
t2 = nltk.BigramTagger(train, backoff=t1)
t3 = nltk.TrigramTagger(train, backoff=t2)
print "Trigram with backoffs: ", t3.evaluate(test)
print " "


""" """


