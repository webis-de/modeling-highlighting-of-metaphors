grammar = "NP: {?*}"
import pandas as pd 
import nltk
import spacy 

# data = pd.read_csv(csvfile)
# texts = data.Sentences
# for text in texts:
#     tokens = nltk.word_tokenize(text)
#     print(tokens)
#     tag = nltk.pos_tag(tokens)
#     print(tag)
#     grammar = "NP: {<DT>?<JJ>*<NN>}"
#     cp  =nltk.RegexpParser(grammar)
#     result = cp.parse(tag)
#     print(result)

'''
From Spacy 
'''

chunker = spacy.load("en_core_web_sm")
doc = chunker('remind me the source of those alleged statistics? did you know that for 21 years FFL dealers have had to do background checks of buyers and there was not a SINGLE study that found those backgroun \
checks decreased gun crime AT ALL. and those were comprehensive studies by groups trying to support the Brady Bill.')

for chunk in doc.noun_chunks:
    print(chunk.text)



