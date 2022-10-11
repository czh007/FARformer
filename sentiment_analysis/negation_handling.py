import spacy
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='german')
nlp = spacy.load('de')
neg_words = ['nicht', 'kein', 'nirgends', 'nirgendwo', 'niemand', 'niemals', 'nirgendwohin', 'nie']



def check_for_negated_sentence(sentence):
    negation = 1
    root = [token for token in sentence if token.dep_ == "ROOT"]
    
    if len(root) == 1:
    	# uses the syntactic dependency between the words to detect if root is negated (token.children are the dependend words of token)
    	children= [stemmer.stem(child.text) for child in root[0].children]
    	if any(neg_word in children for neg_word in neg_words):
    		negation = -1
    		
    return negation
    		

def check_for_negation(sentence, word):
    negation = check_for_negated_sentence(sentence)
    	
    for token in sentence:

        if word == token:

            # uses the syntactic dependency between the words to detect which word is negated (token.children are the dependend words of token)
            children = [stemmer.stem(child.text) for child in token.children]
            
            if any(neg_word in children for neg_word in neg_words):
                negation = negation* (-1)
            	   
    return negation


if __name__ == "__main__":
    doc1 = nlp('sind')
    doc2 = nlp('sind nicht')

    