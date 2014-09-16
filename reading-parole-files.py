# -*- coding: cp1252 -*-

import nltk
from cPickle import load

##################################

# untagged and tagged text corpus, raw version
text = (open('da-korpus-untagged.txt', 'rU').read()).split()
# inspecting:
print text[:10]
print ""

text2 = (open('da-korpus-tagged.txt', 'rU').read()).split()
# inspecting:
print text2[:10]
print ""

##################################

# load tagged corpus of word-tag tuples

input = open('da_tagged_words.pkl', 'rb')
tagged_words = load(input)
input.close
# inspecting:
print tagged_words[:10]
print ""

##################################

# load untagged corpus divided into sentences

input = open('da_untagged_sents.pkl', 'rb')
textsents = load(input)
input.close
# inspecting:
print textsents[:3]
print ""

###################################

# load tagged corpus divided into sentences

input = open('da_tagged_sents.pkl', 'rb')
tagged_sents = load(input)
input.close
# inspecting:
print tagged_sents[:3]
print ""




