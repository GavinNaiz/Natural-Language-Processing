"""
######### START OF 4 #########
"""
# Imports
import nltk
from nltk.corpus import brown

# Load in the texts and split into sentences stored in a list.
texts = "John met Peter in the court one week ago.\nAnn met him in the parliament yesterday.\nShe waved at him repeatedly.\nThat morning, Peter was discussing with three english lobbyists.\nHe did not notice his friend."
texts = texts.split("\n")

# Load training corpus from the Brown corpus. We are using all data available.
tagged = brown.tagged_sents()

"""
  Train the tagger, starting with Default tagger up to BigramTagger.
  This is the most time consuming part(!).
"""
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(tagged, backoff=t0)
t2 = nltk.BigramTagger(tagged, backoff=t1)

# Create a dictionary linking words to their agreement feature.
masc_sing = ['he', 'him', 'john', 'peter', 'zebediah']
fem_sing = ['she', 'ann', 'friend', 'margreth', 'zelda']
inam_sing = ['it', 'car', 'court', 'morning', 'english', 'notice', 'parliament', 'week', 'yesterday', 'xylophone']
plural = ['they', 'cars', 'lobbyists']

# Grammar to use for chunking.
grammar = r"""
  NP: {<DT>?<JJ>*<NP><NNS>?}
    {<DT>?<JJ>*<NP>|<NNS>}
    {<DT>?<JJ>*<NNS?><NNS?>?}
    {<NNP>+}
    {<NP>}
    {<NR>}
    {<PPS>+}
    {<PP$>}
    {<PPO>}
"""

chunk_parser = nltk.RegexpParser(grammar)

results = []

# Run the sentences.
# Aiming for the format of ([ 'john', 'NNP', 'masc_sing'],[ 'ann', 'NNP', 'fem_sing'],[ 'him', 'PRP', 'masc_sing'])
for text in texts:

  # Tokenize the sentence.
  tmp = nltk.word_tokenize(text)

  # Use the trained BigramTagger to assign Part-of-Speech tags to all tokens.
  tmp = t2.tag(tmp)

  # Chunking the sentences into trees so that we can extract NPs
  tmp = chunk_parser.parse(tmp)

  """
    Extract all elements from the chunked trees that are Noun Phrases.
    Sanitize the trees and only stores the head nouns. Removing delimiters, adjectives and so on.
  """
  np_list = []
  for elem in tmp:
    if isinstance(elem, nltk.tree.Tree) and elem.node == "NP":
      if len(elem) < 2:
        np_list.append(list(elem[0]))
      else:
        np_list.append(list(elem[1]))

  for i in np_list:
    i[0] = i[0].lower()

  # Add agreement features to the head nouns.
  for pair in np_list:
    if pair[0] in masc_sing:
      pair.append('masc_sing')
    elif pair[0] in fem_sing:
      pair.append('fem_sing')
    elif pair[0] in inam_sing:
      pair.append('inam_sing')
    elif pair[0] in plural:
      pair.append('plural')
    else:
      # If the word doesn't occur in the dictionary.
      pair.append('UNDEFINED')

  results.append(np_list)

for i in results:
  print tuple(i), '\n'


"""
######### START OF 5 #########
"""

# The dictionary of centers for each utterance.
rules = {
  'u1': {
    'cf': ['john', 'peter', 'court', 'week'],
    'cp': ['john'],
    'cb': ['undefined']
  },
  'u2': {
    'cf': ['ann', 'him', 'parliament'],
    'cp': ['ann'],
    'cb': ['ann'],
    'link': {'him': 'john'}
  },
  'u3': {
    'cf': ['she', 'him'],
    'cp': ['she'],
    'cb': ['she'],
    'link': {'she': 'ann', 'him': 'john'}
  },
  'u4': {
    'cf': ['peter', 'lobbyists', 'morning'],
    'cp': ['peter'],
    'cb': ['undefined']
  },
  'u5': {
    'cf': ['he', 'friend'],
    'cp': ['he'],
    'cb': ['he'],
    'link': {'he': 'peter', 'friend': 'ann'}
  }
}

# The parameter sets provided in the assignment.
ut = [
  (['john', 'NNP', 'masc_sing'],['ann', 'NNP', 'fem_sing'],['him', 'PRP', 'masc_sing']),
  (['him', 'PRP', 'masc_sing'],['she', 'PRP', 'fem_sing'],['she', 'PRP', 'fem_sing']),
  (['she', 'PRP', 'fem_sing'],['morning', 'NNP', 'anim_sign'],['peter', 'NNP', 'masc_sing']),
  (['peter', 'NNP', 'masc_sing'],['he', 'PRP', 'masc_sing'],['he', 'PRP', 'masc_sing'])
]

"""
  The function that is calculating transitions between the utterances.

  Taking the Cb(Un), Cp(Un+1), Cb(Un+1) and the counter as parameters.

  Printing the calculated transitions to the screen.
"""
def transition(cb1, cp2, cb2, count):

  # Check for PRPs in the parameters and link them together with their corresponding part of previous utterance.
  # This results in that 'him' in parameter set two is linked together with 'john' from the first utterance.
  if cb1[1] == 'PRP':
    cb1[0] = rules['u' + str(count)]['link'][cb1[0]]
    
  if cp2[1] == 'PRP':
    cp2[0] = rules['u' + str(count + 1)]['link'][cp2[0]]
    
  if cb2[1] == 'PRP':
    cb2[0] = rules['u' + str(count + 1)]['link'][cb2[0]]

  """
    Scoring the transitions based on the transition rule-set.
    Score or 1 in the first test means that it is in the first
    column of the transition matrix. Score of 2 is in the second column.

    We are following the equality rule provided in the assignment
    by checking if tokens and agreement features are the same. In case of true
    we check if one is a pronoun.
  """
  score = 0
  if cb2[0] == cp2[0] and cb2[2] == cp2[2]:
    if cb2[1] == 'PRP' or cp2[1] == 'PRP':
      score = 1
  else:
    score = 2

  # The second part of the scoring is giving us the actual transition.
  # Instead of columns we are now testing for the rows.
  if score == 1:
    if cb2[0] == cb1[0] and cb2[2] == cb1[2]:
      if cb2[1] == 'PRP' or cb1[1] == 'PRP':
        trans = 'Continue' # 11
    else:
      trans = 'Retain' # 12

  if score == 2:
    if cb2[0] == cb1[0] and cb2[2] == cb1[2]:
      if cb2[1] == 'PRP' or cb1[1] == 'PRP':
        trans = 'Smooth-shift' # 21
    else:
      trans = 'Rough-shift' # 22

  # Print out the results.
  print trans


# Use a counter to keep track of what rule to refer to.
count = 1

# Iterate the parameter sets and run the function.
for u in ut:
  transition(u[0], u[1], u[2], count)
  count += 1
