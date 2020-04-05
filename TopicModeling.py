import pandas as pd
import numpy as np
import string
#Importing nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
#Importing sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#Importing matplotlib for wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud 


class CustomAnalyzer(object):
    def __init__(self,stopwords_,synonyms):
        self.syns = synonyms
        self.stopwords_ = stopwords_
    def customanalyzer(self,strDoc):        
        #Replacing punctuations    
        puncts = string.punctuation
        for i in range(len(puncts)):
            strDoc = strDoc.replace(puncts[i], ' ')
        
        # Tokenizing the input document string 
        word_tokens = word_tokenize(strDoc)        
        word_tokens = [word for word in word_tokens if (word!="th")\
                       and ("''" != word) and ("``" != word) \
                           and (word!="'s") and not word.isnumeric()]
            
        # Map synonyms
        for i in range(len(word_tokens)):
            if word_tokens[i] in self.syns:
                word_tokens[i] = self.syns[word_tokens[i]]
           
            # Remove stop words
        punctuation = list(string.punctuation)+['..', '...']
        pron = ['it', 'him', 'they', 'we', 'us', 'them','i', 'he', 'she','let',]
        generic = ["'d", "co", "ed", "put", "say", "get", "can", "become",\
                    "los", "sta", "la", "use", "else", "could", "would", "come", "take",'th','s']
        stop = stopwords.words('english') + punctuation + pron + generic + self.stopwords_
        filtered_terms = [word for word in word_tokens if (word not in stop) and \
                          (len(word)>1) and (not word.replace('.','',1).isnumeric()) \
                          and (not word.replace("'",'',2).isnumeric())]
            
        # Lemmatization & Stemming - Stemming with WordNet POS               
        tagged_tokens = pos_tag(filtered_terms, lang='eng')        
            
        # Stemming with for terms without WordNet POS
        stemmer = SnowballStemmer("english")
        wordnet_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
        wordnetlem = WordNetLemmatizer()
        stemmed_tokens = []
        for tagged_token in tagged_tokens:
            term = tagged_token[0]
            pos  = tagged_token[1]
            pos  = pos[0]
            try:
                pos   = wordnet_tags[pos]
                lemma_token = wordnetlem.lemmatize(term, pos=pos)
                if lemma_token not in stop:
                    stemmed_tokens.append(lemma_token)
            except:
                lemma_token = stemmer.stem(term)
                if lemma_token not in stop:
                    stemmed_tokens.append(lemma_token)
        return stemmed_tokens


#Reading data from CSV file
df = pd.read_csv("train.csv")
text_content = 'comment_text'
#let's take a peek at data
df.head()

#declaring number of topics to be created from text data
total_topics = 15 


# Declaring synonyms for commonly used words in English text content
synonyms_ = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying","thank":"thanks"}

# Declaring additional stopwords based on text content and context
stopwords_ = ['article','Articles','page','wikipedia','talk','edit', 'one','make','please','like','see','think',
      'know','source','go','also','time','add','people','user','well','need','block','may','want','link','image',
      'good','name','delete','find','look','thanks','work','remove','even','comment','help','write','information',
      'change','way','list','give','deletion','editor','question','section','point','thing','try','wiki',
      'first','wp','new','fact','seem','state','read','reference','discussion','right','thank','many','place','much',
      'ask','revert','really','mean','reason','call','include','sinc','edits','create','word','back','note','tag',
      'post','someone','policy','wiki','show','leave','issue','two','year','still','stop',
      'content',	'hi',	'case',	'consider',	'something',	'keep',	'claim',	'http',	'mention',	'without',
      'problem',	'let',	'day',	'person',	'utc',	'request',	'welcome',	'believe',	'anoth',	'might',
      'subject',	'feel',	'part',	'free',	'start',	'however',	'sure',	'never',	'tell',	'book',	'view',
      'copyright',	'actually',	'anything',	'follow',	'agree',	'regard',	'continue',	'best',	'hope',	'understand',
      'site',	'long',	'provide',	'already',	'term',	'great',	'move',	'com',	'nothing',	'review',	'though',
      'notice',	'little',	'explain',	'message',	'last',	'anyone',	'must',	'others',	'contribution',	'speedy',
      'example',	'number',	'account',	'style',	'text',	'bad',	'title',	'sorry',	'appear',	'rather',
      'fair',	'different',	'ip',	'matter',	'life',	'non',	'cite',	'suggest',	'report',	'template',	'guideline',
      'correct',	'statement',	'old',	'lot',	'address',	'original',	'probably',	'language',	'everi',	'material',
      'top',	'simply',	'consensus',	'hello',	'either',	'live',	'interest',	'far',	'least',	'notable',	'yes','date',
      'enough',	'etc',	'idea',	'base',	'around',	'admin',	'ban',	'real',	'version',	'www',	'website',	'yet',	'evidence',
      'clear',	'encyclopedia',	'quote',	'end',	'research',	'topic',	'picture',	'clearly',	'medium',	'ever',	'file',
      'maybe',	'exist',	'instead',	'country',	'org',	'pov',	'criterion',	'important',	'true',	'oh',	'always',
      'happen',	'perhaps',	'quite',	'whether',	'care',	'big',	'lead',	'bit',	'administrator',	'contribute',	'sign',
      'citation',	'answer',	'allow',	'second',	'sentence',	'three',	'line',	'several',	'hey',	'high',	'man',
      'argument',	'project',	'current',	'kind',	'redirect',	'action',	'general',	'refer',	'common',	'mind',
      'summary',	'concern',	'course',	'discuss',	'present',	'result',	'possible',	'main',	'accept',	'test','learn',
      'order',	'play',	'type',	'less',	'dont',	'jpg',	'en',	'member',	'attempt',	'ok',	'sense',	'party',	'week',
      'form',	'info',	'notability',	'position',	'act',	'side',	'contribs',	'company',	'city',	'entry',	'warn',	'four',
      'specific',	'news',	'publish',	'appropriate',	'standard',	'single',	'detail',	'anyway',	'open',	'reply','cause',
      'fix',	'meet',	'next',	'describe',	'system',	'film',	'copy',	'full',	'although',	'per',	'upload',	'relevant',
      'away',	'lol',	'stay',	'record',	'large',	'speak',	'recent',	'band',	'search',	'run',	'official',	'process',	'public',
      'month',	'area',	'response',	'currently',	'everyone',	'especially',	'later','release',	'able',	'propose',
      'check','paragraph','web','otherwise','generally']
#instantiating custom analyzer class object
ca = CustomAnalyzer(stopwords_,synonyms_)
max_df = 1.9 
min_df = 10
cv = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None,lowercase=True,\
                     analyzer = ca.customanalyzer)
#Fitting count vectorizer 
td = cv.fit_transform(np.array(df[text_content].values.astype('U')))

#getting the terms(features) generated by count vectorizer
terms = cv.get_feature_names()

# next step is to implement Latent Drichilet Allocation 
lda = LatentDirichletAllocation(n_components=total_topics, 
            max_iter=15,learning_method='online', 
            learning_offset=15,random_state=1234)
ldaTransform = lda.fit_transform(td)
#declaring number of terms we need per topic
terms_count = 25
#Looping over lda components to get topics and their related terms with high probabilities
for idx,topic in enumerate(lda.components_):    
    print('Topic# ',idx+1)
    abs_topic = abs(topic)
    topic_terms = [[terms[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
    topic_terms_sorted = [[terms[i], topic[i]] for i in abs_topic.argsort()[:-terms_count - 1:-1]]
    topic_words = []
    for i in range(terms_count):
        topic_words.append(topic_terms_sorted[i][0])
    print(','.join( word for word in topic_words))
    print("")
    dict_word_frequency = {}
    
    for i in range(terms_count):
        dict_word_frequency[topic_terms_sorted[i][0]] = topic_terms_sorted[i][1]    
    wcloud = WordCloud(background_color="white",mask=None, max_words=100,\
                        max_font_size=60,min_font_size=10,prefer_horizontal=0.9,
                        contour_width=3,contour_color='black')
    wcloud.generate_from_frequencies(dict_word_frequency)       
    plt.imshow(wcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("Topic#"+str(idx+1), format="png")
  
   
