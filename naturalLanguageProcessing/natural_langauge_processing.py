import nltk
from nltk import word_tokenize

ulysses = "Mrkgnao! the cat said loudly. She blinked up out of her avid shameclosing eyes, mewing \
plaintively and long, showing him her milkwhite teeth. He watched the dark eyeslits narrowing \
with greed till her eyes were green stones. Then he went to the dresser, took the jug Hanlon's\
milkman had just filled for him, poured warmbubbled milk on a saucer and set it slowly on the floor.\
— Gurrhr! she cried, running a.k.a. to lap."

doc = nltk.sent_tokenize(ulysses)
for s in doc:
    print(">",s)


sentence = "Mary had a little lamb it's fleece was white as snow."
# Default Tokenisation
tree_tokens = word_tokenize(sentence)   # nltk.download('punkt') for this

# Other Tokenisers
punct_tokenizer = nltk.tokenize.WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(sentence)

space_tokenizer = nltk.tokenize.SpaceTokenizer()
space_tokens = space_tokenizer.tokenize(sentence)

print("DEFAULT: ", tree_tokens)
print("PUNCT  : ", punct_tokens)
print("SPACE  : ", space_tokens)


pos = nltk.pos_tag(tree_tokens)
print(pos)
pos_space = nltk.pos_tag(space_tokens)
print(pos_space)


import re
regex = re.compile("^N.*")
nouns = []
for l in pos:
    if regex.match(l[1]):
        nouns.append(l[0])
print("Nouns:", nouns)


porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snowball = nltk.stem.snowball.SnowballStemmer("english")

print([porter.stem(t) for t in tree_tokens])
print([lancaster.stem(t) for t in tree_tokens])
print([snowball.stem(t) for t in tree_tokens])

sentence2 = "When I was going into the woods I saw a bear lying asleep on the forest floor"
tokens2 = word_tokenize(sentence2)

print("\n",sentence2)
for stemmer in [porter, lancaster, snowball]:
    print([stemmer.stem(t) for t in tokens2])


wnl = nltk.WordNetLemmatizer()
tokens2_pos = nltk.pos_tag(tokens2)

print('lemma', [wnl.lemmatize(t) for t in tree_tokens])
print('lemma', [wnl.lemmatize(t) for t in tokens2])
