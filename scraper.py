from bs4 import BeautifulSoup
import requests
from datetime import datetime
from collections import Counter
import pandas as pd
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
import spacy
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
nltk.download('wordnet')
nltk.download('stopwords')

### take user input ###
user_input = "prince"
# user_input = input ("Artist Name: ")
user_input = user_input.replace(' ', '-')
user_input = user_input.replace(' ', '')
### start timer ###
start = datetime.now()
### scraping song links ###
print('--- scraping song links ---')
source = requests.get(f'https://www.songlyrics.com/{user_input}-lyrics/').text
soup = BeautifulSoup(source, 'lxml')
songlist = soup.find('div', class_='listbox')
tracklist = songlist.find('table', class_='tracklist').tbody
song_links = []
artist_details = []
for song in tracklist.find_all('tr', itemprop="itemListElement"):
    if song.td.text in [str(x) for x in range(10 + 1)]:
        link = song.find('a')['href']
        if link not in song_links:
            song_links.append(link)
### collecting song details ###
print('--- scraping song details text ---')
for val in song_links:
    song_title = val[27:-1].split('/', 1)[1]
    song_title = song_title[:-6].replace('-', ' ').capitalize()
    artist_name = user_input.replace('-', ' ').title()
### scraping song text ###
    songsource = requests.get(val).text
    soup2 = BeautifulSoup(songsource, 'lxml')
    block = soup2.find('div', id='songLyricsContainer')
    if block.find('p').text != False:
        text = block.find('p').text
        if 'feat.' not in text:
            permitted = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
            songtext = text.lower()
            songtext = ' '.join(word for word in songtext.split() if word[0]!='[')
            songtext = songtext.replace("\n", " ").strip()
            songtext = "".join(c for c in songtext if c in permitted)
            songtext = songtext.replace("  ", " ").capitalize()
            artist_details.append([artist_name, song_title, songtext])
### create a data frame ###
print('--- Data Frame ---')
df = pd.DataFrame (artist_details,columns=['Artist Name', 'Song Title', 'Song Text'])                    
### analysing text ###
def sent_to_words(sentence):
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
top_words = []
for val in df['Song Text']:
    val = remove_stopwords(val.lower())
    split_val = val.split()
    count = Counter(split_val)
    comm = count.most_common(5)
    top_words.append(comm)
df['Top Words'] = top_words
### dataframe ###
print(df.shape)
print(df.head())
### cleaning text ### 
songs_1 = list(df['Song Text'])
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
clean_text = []
for val in songs_1: 
    val = val.lower()
    split_ = val.split()
    stop_f = " ".join([i for i in split_ if i not in stop])
    punc_f = ''.join(ch for ch in stop_f if ch not in exclude)
    norm = " ".join(lemma.lemmatize(i) for i in split_)
    clean_text.append(norm)
### encoding dictionary ###
doc_coded = []
for doc in clean_text:
    split_ = doc.split()
    doc_coded.append(split_)
dictionary = corpora.Dictionary(doc_coded)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_coded]
### topic modeling ###
Lda = gensim.models.ldamodel.LdaModel
final = []
for val in doc_term_matrix:
    ldamodel = Lda(doc_term_matrix, num_topics=8, id2word = dictionary, passes=50)
    topics = ldamodel.print_topics(num_topics=8, num_words=1)
    final.append(topics)
df['Topics'] = final
### data frame ###
print(df.shape)
print(df.head())

#### finish timer ###
print('--- runtime ---')
break1 = datetime.now()
print("Elapsed time: {0}".format(break1-start)) # show timer