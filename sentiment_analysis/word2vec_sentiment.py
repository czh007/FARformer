import pandas as pd
import json
import spacy
import regex as re
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec
from time import time

import logging  
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.WARN)

pd.io.json._json.loads = lambda s, *a, **kw: json.loads(s)

class SentiW2v:
    text = None
    def __init__(self):
        self.w2v_model = Word2Vec(min_count=20,
                            window=2,
                            size=300,
                            sample=6e-5,
                            alpha=0.03,
                            min_alpha=0.0007,
                            negative=20,
                            workers=multiprocessing.cpu_count()-1)
        self.nlp = spacy.load("de", disable=["tagger", "parser","ner"])

    def cleaning(self,doc):
        # lemma of all stop words 
        
        txt = [token.lemma_ for token in doc if not token.is_stop]
        if len(txt)>2:
            return ' '.join(txt)

    def set_text_from_file(self,path):
        df = pd.read_json(path, orient='index')        
        self.text=df["text"]

    def set_text_from_pandas(self, data):
        df = data
        self.text=df["text"]

    def clean_and_train(self):
        brief_cleaning = (re.sub("[^A-Za-züäöÜÄÖß']",' ', str(row)).lower() for row in self.text)
        print(f"Text length: {len(self.text)}")
        if len(self.text) == 0:
            return
        t = time()
        txt = [self.cleaning(doc) for doc in self.nlp.pipe(brief_cleaning, batch_size=50,n_threads=-1)]
        print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

        df_clean = pd.DataFrame({'clean': txt})
        df_clean = df_clean.dropna().drop_duplicates()
    
        sent = [row.split() for row in df_clean["clean"]]

        phrases = Phrases(sent, min_count=30, progress_per=10000)

        bigram = Phraser(phrases)
        sentences = bigram[sent]

        t = time()
        self.w2v_model.build_vocab(sentences, progress_per=10000)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, 
                                                            epochs=30, report_delay=1)
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        self.w2v_model.init_sims(replace=True) # called when finished training for memory efficiency

    def print_most_similar(self, word):
        try:
            print(self.w2v_model.wv.most_similar(word))
        except KeyError:
            print(f"The word {word} is not in the in vocabulary.")

    def get_most_similar(self,word, number):
        try:
            return self.w2v_model.wv.most_similar(word, topn=number)
        except KeyError:
            print(f"The word {word} is not in the in vocabulary.")


def similarity_by_year(input_path :  str,output_path : str, search_words : list,
                       start_year=2007, end_year=2015, number_most_sim=10):
    nlp = spacy.load("de")
    search_doc = nlp(" ".join(search_words))
    search_words = [token.lemma_ for token in search_doc]
    

    try:
        content = pd.read_json(input_path, orient="index", precise_float=True) # , numpy=True)
    except ValueError:
        return
        
    content = content[["date","text","url"]]
    most_sim = {}

    content["date"]=content['date'].astype('str')
    for year in range(start_year,end_year+1):
        print(f"\nStart analysis for year {year}")
        sw=SentiW2v()
        t=time()
        sub_cont= content[content["date"].str.contains(str(year))]
        print(f"Time to reduce content {round((time() - t) / 60, 2)} mins")
        sw.set_text_from_pandas(sub_cont)
        sw.clean_and_train()
        most_sim[str(year)] = { key : sw.get_most_similar(key ,number_most_sim) for key in search_words}
    with open(output_path+"_by_year.json","w") as f:
        json.dump(most_sim,f)

def similarity_by_publisher(input_path: str,output_path: str, search_words: list, 
                            start_year=2007, end_year=2015, number_most_sim=10):
   
    content = pd.read_json(input_path, orient="index")
    content = content[["date","text","url"]]
    most_sim = {}
    list_publishers = []
    for i , row in content.iterrows():
        try:
            publisher = row["url"].split("//")[1].split("/")[0].split(".")[1]
        except KeyError:
            print("KeyError no url")
            print(row)
            continue
        if publisher in list_publishers:
            continue
        else:
            list_publishers.append(publisher)
            sw=SentiW2v()
            sub_cont=content[content["url"].str.contains(publisher)]
            sw.set_text_from_pandas(sub_cont)
            try:
                sw.clean_and_train()
                most_sim[publisher]={ key : sw.get_most_similar(key ,number_most_sim) for key in search_words}     
            except RuntimeError:
                print(f"RuntimeError in training for publisher: {publisher}")
    with open(output_path+"_by_publisher.json","w") as f:
        json.dump(most_sim,f)

def similarity_by_year_and_publisher(input_path : str, output_path : str,search_words : list,
                                     start_year=2007, end_year=2015, number_most_sim=10):

    content = pd.read_json(input_path, orient="index")
    content=content[["date","text","url"]]
    most_sim={}
    list_publishers = []
    for i , row in content.iterrows():
        year = str(row["date"]) .split("-")[0]
        try:
            publisher = row["url"].split("//")[1].split("/")[0].split(".")[1]
        except KeyError:
            print("KeyError no url")
            print(row)
            continue
        if publisher not in list_publishers:
            list_publishers.append(publisher)
            most_sim[publisher]={}

        if publisher in list_publishers and year in most_sim[publisher]:
            continue
        sw=SentiW2v()
        sub_cont=content[content["url"].astype(str).str.contains(publisher)] #content[content["url"].str.contains(publisher)]
        sub_cont=sub_cont[sub_cont["date"].astype(str).str.contains(year)]
        sw.set_text_from_pandas(sub_cont)
        try:
            sw.clean_and_train()
            most_sim[publisher][year]={ key : sw.get_most_similar(key ,number_most_sim) for key in search_words}
        except RuntimeError:
            continue
    with open(output_path+"_by_publisher_and_year.json","w") as f:
        json.dump(most_sim,f)



if __name__=="__main__":
    input_file_path="../data/relevant_articles.json"
    search_word_list=["grenze", "abschiebung","migration", "einwanderung"]
    
    if False:
        similarity_by_year(input_file_path,"hadsj.json",search_word_list)
        
    if False:
        similarity_by_publisher(input_file_path,"kljsdf.json",search_word_list)
    
    if True:
        similarity_by_year_and_publisher(input_file_path,"hadsj.json",search_word_list)