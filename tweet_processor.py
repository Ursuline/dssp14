#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:42:54 2020

tweet_processor.py

@author: charles meegnin
"""
import os
import requests
import string
import re
import emoji

from nltk.stem.snowball import FrenchStemmer
from langdetect import detect
from langdetect import DetectorFactory

from tweet_io import load_data_by_name

def load_stopwords():
    # First upload stopword file:
    filename = 'stopwords-fr.txt' # French stopwords
    path = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/'
    url = path + filename
    if not os.path.exists(filename): #check it's not already here
        r = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(r.content)

    #Read stopwords from file into list
    stopwords = list()
    with open(filename, 'r') as filehandle:
        filecontents = filehandle.readlines()

        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_place = line[:-1]

            # add item to the list
            stopwords.append(current_place)
    return(stopwords)


def extract_url(str):
    ur = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str)
    return ur


def remove_urls(tweet):
    urls = extract_url(tweet)
    if urls: # if list of urls isn't empty
        for url in urls:
            tweet = tweet.replace(url, "")
    return tweet


def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def remove_non_ascii(text):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


def process_surnames(tweet):
    """
    replaces possible spellings of politicians last names to last name
    """
    tweet = tweet.replace('jlmelenchon', 'melenchon')
    tweet = tweet.replace('jl melenchon', 'melenchon')
    tweet = tweet.replace('jean-luc melenchon', 'melenchon')
    tweet = tweet.replace('melanchon', 'melenchon')
    tweet = tweet.replace('france insoumise', 'lfi')
    tweet = tweet.replace(' insoumis ', ' lfi ')
    tweet = tweet.replace('franceinsoumise' , 'lfi')

    tweet = tweet.replace('villanicedric', 'villani')
    tweet = tweet.replace('cedricvillani', 'villani')
    tweet = tweet.replace('cedric villani', 'villani')

    tweet = tweet.replace('emmanuelmacron', 'macron')
    tweet = tweet.replace('emmanuel macron', 'macron')
    tweet = tweet.replace('enmarche', 'lrem')

    tweet = tweet.replace('angela merkel', 'merkel')
    tweet = tweet.replace('donald trump', 'trump')
    tweet = tweet.replace('ccastaner', 'castaner')
    tweet = tweet.replace(' pen ', ' lepen ')


    tweet = tweet.replace('rassemblement national', 'rn')
    tweet = tweet.replace('rnationaloff', 'rn')
    tweet = tweet.replace('rassemblementnational', 'rn')

    tweet = tweet.replace('francois ruffin', 'francoisruffin')
    tweet = tweet.replace('francois_ruffin', 'francoisruffin')
    tweet = tweet.replace('manonaubryfr', 'manonaubry')

    return tweet


def string_to_list(string):
    li = list(string.split(" "))

    return li


def remove_stopwords(tweet):
    """
    removes the stopwords from db
    allows custom words as stopwords2
    """
    # List of stopwords for the project
    stopwords2 = ['mme', 'madame', "monsieur", "m.", "m",
                  "s'est", "s’est", 'lequel', "laquelle", "lesquels", "lesquelles",
                  "ete", "etes", "etait", "etais", "etions", "or",
                  "ca", "ça", "est-ce", "voila",
                  '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
                  ]

    stopwords = load_stopwords() # from external file
    stopwords = stopwords + stopwords2 # my stopwords

    #convert tweet to list of tokens
    tweet = remove_apostrophes(tweet)

    tweet = string_to_list(tweet)
    #tweet_b4 = ' '.join(tweet)
    #b4 = len(tweet)

    tweet = [word for word in tweet if word not in stopwords]

    tweet = ' '.join(tweet) # convert to string

    return tweet


def remove_apostrophes(tweet):
    """Apostrophes are part of a word and not removed in remove_stopwords()"""
    # replace french apostrophe with apostrophe
    tweet = tweet.replace("’", "'")

    tweet = tweet.replace("l'", "")
    tweet = tweet.replace("c'", "")
    tweet = tweet.replace("d'", "")
    tweet = tweet.replace("j'", "")
    tweet = tweet.replace("s'", "")
    tweet = tweet.replace("t'", "")
    tweet = tweet.replace("m'", "")
    tweet = tweet.replace("n'", "")
    tweet = tweet.replace("jusqu'", "")
    tweet = tweet.replace("qu'", "")

    return tweet


def remove_punctuation(tweet):
    """
    1. Remove punctuation and add typical french punctuation
    2. set to lower case
    """
    punctuation = set(string.punctuation)
    punctuation.add('»') # French quotes
    punctuation.add('«')
    punctuation.add("’") # French apostrophe
    punctuation.add(u"'️")

    tweet = ''.join(ch for ch in tweet if ch not in punctuation)

    return tweet


def process_idioms(tweet):
    """ fine-tuning :
        tweet is a string
        - remove slogans
        - remove signatures
        - various other tweaks
        - must be run after process_accents()
    """

    # slogans
    tweet = tweet.replace('onarrive', '')
    tweet = tweet.replace('alertedemocratie', '')
    tweet = tweet.replace('jevoteinsoumis', '')
    tweet = tweet.replace('entoutefranchise', '')
    tweet = tweet.replace('maintenantlepeuple', '')
    tweet = tweet.replace('lacriseetapres', '')

    # signature
    tweet = tweet.replace('mlp', '')
    tweet = tweet.replace('amiensfi', '')
    tweet = tweet.replace('rouenfi', '')
    tweet = tweet.replace('toulousefi', '')
    tweet = tweet.replace('limogesfi', '')
    tweet = tweet.replace('caenfi', '')
    tweet = tweet.replace('lyonfi', '')
    tweet = tweet.replace('saintbrieucfi', '')
    tweet = tweet.replace('bordeauxfi', '')
    tweet = tweet.replace('besanconfi', '')

    tweet = tweet.replace('francaises', 'francais')
    tweet = tweet.replace('francaise', 'francais')

    tweet = tweet.replace('cooperations', 'cooperation')
    tweet = tweet.replace('protections', 'protection')

    tweet = tweet.replace(' ue', ' europe')
    tweet = tweet.replace('Union européenne', 'europe')
    tweet = tweet.replace('union européenne', 'europe')
    tweet = tweet.replace('union europeenne', 'europe')
    tweet = tweet.replace('europeens', 'europe')
    tweet = tweet.replace('europeenes', 'europe')
    tweet = tweet.replace('europeennes', 'europe')
    tweet = tweet.replace('europenes', 'europe')
    tweet = tweet.replace('europeen', 'europe')
    tweet = tweet.replace('europeene', 'europe')
    tweet = tweet.replace('europeenne', 'europe')
    tweet = tweet.replace('europene', 'europe')
    tweet = tweet.replace('europes', 'europe')
    tweet = tweet.replace('europe2019', 'europe')
    tweet = tweet.replace('quelleestvotreeurope', 'europe')

    tweet = tweet.replace(' retraitees', ' retraite')
    tweet = tweet.replace(' retraites', ' retraite')
    tweet = tweet.replace('reformedesretraites', 'retraite')
    tweet = tweet.replace('reformeretraites', 'retraite')
    tweet = tweet.replace('meetingretraites', 'meeting retraite')

    tweet = tweet.replace('grevegenerale', 'greve')
    tweet = tweet.replace('grevedu5decembre', 'greve')
    tweet = tweet.replace('greve5decembre', 'greve')
    tweet = tweet.replace('greve10decembre', 'greve')
    tweet = tweet.replace('greve12decembre', 'greve')
    tweet = tweet.replace('grevedu17decembre', 'greve')
    tweet = tweet.replace('greve17decembre', 'greve')
    tweet = tweet.replace('greve22decembre', 'greve')
    tweet = tweet.replace('greve24decembre', 'greve')
    tweet = tweet.replace('greve27decembre', 'greve')
    tweet = tweet.replace('greve28decembre', 'greve')
    tweet = tweet.replace('greve9janvier', 'greve')
    tweet = tweet.replace('greve11janvier', 'greve')
    tweet = tweet.replace('greve23janvier', 'greve')
    tweet = tweet.replace('greve24janvier', 'greve')
    tweet = tweet.replace('GreveMondialePourLeClimat', 'climat greve')
    tweet = tweet.replace('grevegenerale5fevrier', 'greve')
    tweet = tweet.replace('greve greve greve', 'greve')
    tweet = tweet.replace('greve greve', 'greve')

    tweet = tweet.replace('retraites', 'retraite')

    # 20 April
    # tweet = tweet.replace('religions', 'religion')
    # tweet = tweet.replace('libres', 'liberte')
    # tweet = tweet.replace('libre', 'liberte')
    # tweet = tweet.replace('sociales', 'social')
    # tweet = tweet.replace('sociale', 'social')
    # tweet = tweet.replace('travailler', 'travail')
    # tweet = tweet.replace('jeunes ', 'jeunesse ')
    # tweet = tweet.replace('jeune ', 'jeunesse ')
    # end 20 April

    tweet = tweet.replace('islams', 'islam')
    tweet = tweet.replace('islamistes', 'islam')
    tweet = tweet.replace('islamiste', 'islam')
    tweet = tweet.replace('islamismes', 'islam')
    tweet = tweet.replace('islamisme', 'islam')
    tweet = tweet.replace('musulmanes', 'islam')
    tweet = tweet.replace('musulmane', 'islam')
    tweet = tweet.replace('musulmans', 'islam')
    tweet = tweet.replace('musulman', 'islam')
    tweet = tweet.replace('arabes', 'islam')
    tweet = tweet.replace('arabe', 'islam')

    tweet = tweet.replace('juifs', 'juif')
    tweet = tweet.replace('antisemites', 'antisemite')

    tweet = tweet.replace("climats", 'climat')
    tweet = tweet.replace("climatiques", 'climat')
    tweet = tweet.replace("climatique", 'climat')

    tweet = tweet.replace("ecologiques", 'ecologique')
    tweet = tweet.replace("ecologie", 'ecologique')

    tweet = tweet.replace('intelligence artificielle', 'ai')
    tweet = tweet.replace('etatsunis', 'usa')

    tweet = tweet.replace('migratoires', 'immigration')
    tweet = tweet.replace('migratoire', 'immigration')
    tweet = tweet.replace('immigrants', 'immigration')
    tweet = tweet.replace('immigrant', 'immigration')
    tweet = tweet.replace('migrants', 'immigration')
    tweet = tweet.replace('migrant', 'immigration')
    tweet = tweet.replace('immigrations', 'immigration')

    tweet = tweet.replace('gilets jaunes', 'giletsjaunes')
    tweet = tweet.replace('gilet jaune', 'giletsjaunes')
    tweet = tweet.replace('giletjaune', 'giletsjaunes')
    tweet = tweet.replace(' gjs', ' giletsjaunes')
    tweet = tweet.replace(' gj', ' giletsjaunes')

    tweet = tweet.replace('votes', 'vote')
    tweet = tweet.replace('voter', 'vote')
    tweet = tweet.replace('votee', 'vote')
    tweet = tweet.replace('votez', 'vote')

    tweet = tweet.replace('luttent', 'lutte')
    tweet = tweet.replace('luttes', 'lutte')
    tweet = tweet.replace('lutter', 'lutte')

    tweet = tweet.replace('terroristes', 'terrorisme')
    tweet = tweet.replace('terroriste', 'terrorisme')

    tweet = tweet.replace('fondamentalistes', 'fondamentalisme')
    tweet = tweet.replace('fondamentaliste', 'fondamentalisme')

    tweet = tweet.replace('elections', 'election')
    tweet = tweet.replace('libertes', 'liberte')

    tweet = tweet.replace('peuples', 'peuple')

    tweet = tweet.replace('violents', 'violence')
    tweet = tweet.replace('violentes', 'violence')
    tweet = tweet.replace('violente', 'violence')
    tweet = tweet.replace('violent', 'violence')
    tweet = tweet.replace('violences', 'violence')

    tweet = tweet.replace('policieres', 'police')
    tweet = tweet.replace('policiere', 'police')
    tweet = tweet.replace('policiers', 'police')
    tweet = tweet.replace('policee', 'police')
    tweet = tweet.replace('policier', 'police')

    tweet = tweet.replace('dangereux', 'danger')
    tweet = tweet.replace('dangereuses', 'danger')
    tweet = tweet.replace('dangereuse', 'danger')

    tweet = tweet.replace('menacees', 'menace')
    tweet = tweet.replace('menacee', 'menace')
    tweet = tweet.replace('menacants', 'menace')
    tweet = tweet.replace('menacante', 'menace')
    tweet = tweet.replace('menacant', 'menace')
    tweet = tweet.replace('menaces', 'menace')
    tweet = tweet.replace('menacer', 'menace')

    tweet = tweet.replace('libres', 'libre')
    tweet = tweet.replace('victimes', 'victime')
    tweet = tweet.replace('emplois', 'emploi')
    tweet = tweet.replace(' lois ', ' loi ')
    tweet = tweet.replace(' forts', ' fort')
    tweet = tweet.replace('defis', 'defi')
    tweet = tweet.replace('travaux', 'travail')
    tweet = tweet.replace('attaques', 'attaque')
    tweet = tweet.replace('eborgneurs', 'eborgneur')

    tweet = tweet.replace('veux', 'vouloir')
    tweet = tweet.replace('veut', 'vouloir')
    tweet = tweet.replace('voulu', 'vouloir')
    tweet = tweet.replace('voulons', 'vouloir')
    tweet = tweet.replace('voulez', 'vouloir')
    tweet = tweet.replace('veulent', 'vouloir')

    tweet = tweet.replace('il faut', 'devoir')
    tweet = tweet.replace('devons', 'devoir')

    tweet = tweet.replace('annees', 'ans')
    tweet = tweet.replace('annee', 'ans')

    tweet = tweet.replace('construction', 'construire')

    tweet = tweet.replace('arretera', 'arreter')
    tweet = tweet.replace('regardent', 'regarder')
    tweet = tweet.replace('faisons', 'faire')

    tweet = tweet.replace('greve22decembre', 'greve')
    tweet = tweet.replace('greve9janvier', 'greve')
    tweet = tweet.replace('grevistes', 'greve')
    tweet = tweet.replace('greviste', 'greve')

    tweet = tweet.replace(' 000', '000')
    tweet = tweet.replace(' publics', ' public')
    tweet = tweet.replace(' publiques', ' public')
    tweet = tweet.replace(' publique', ' public')

    return tweet


def map_to_tv(tweet):
    tv = 'TV_RADIO'
    tweet = tweet.replace('cnews', tv)
    tweet = tweet.replace('emacrontf1', tv)
    tweet = tweet.replace('macrontf1', tv)
    tweet = tweet.replace('e1matin', tv)
    tweet = tweet.replace('bfmpolitique', tv)
    tweet = tweet.replace('le79inter', tv)
    tweet = tweet.replace('lemissionpolitique', tv)
    tweet = tweet.replace('sudradiomatin', tv)
    tweet = tweet.replace('bfmtv', tv)
    tweet = tweet.replace('rmcinfo', tv)
    tweet = tweet.replace('rtlfrance', tv)
    tweet = tweet.replace('bourdindirect', tv)
    tweet = tweet.replace('jjbourdinrmc', tv)
    tweet = tweet.replace('europe1', tv)
    tweet = tweet.replace('legrandrdv', tv)
    tweet = tweet.replace('legrandjury', tv)
    tweet = tweet.replace('france2tv', tv)
    tweet = tweet.replace('rmc', tv)
    tweet = tweet.replace('jpelkabbach', tv)
    tweet = tweet.replace('19hruthelkrief', tv)
    tweet = tweet.replace('lci', tv)
    tweet = tweet.replace('entoutefranchise', tv)
    tweet = tweet.replace('brunetneumann', tv)
    tweet = tweet.replace('punchline', tv)
    tweet = tweet.replace('classiquematin', tv)
    tweet = tweet.replace('rtlmatin', tv)
    tweet = tweet.replace('les4v', tv)
    tweet = tweet.replace('jlm'+tv, tv)

    return tweet


def perform_stemming(tweet):
    stemmer = FrenchStemmer()
    #print(stemmer.stem('continuation')) # test word
    tweet = [stemmer.stem(w) for w in tweet]

    return tweet


def process_accents(tweet):
    """Remove accents and ç"""
    tweet = tweet.replace('é', 'e')
    tweet = tweet.replace('è', 'e')
    tweet = tweet.replace('ê', 'e')
    tweet = tweet.replace('ë', 'e')
    tweet = tweet.replace('ç', 'c')
    tweet = tweet.replace('â', 'a')
    tweet = tweet.replace('à', 'a')
    tweet = tweet.replace('ô', 'o')
    tweet = tweet.replace('ü', 'u')
    tweet = tweet.replace('ù', 'u')
    tweet = tweet.replace('î', 'i')
    tweet = tweet.replace('œ', 'oe')

    return tweet


def last_tweak(tweet):
    """Filter whatever went through the cracks"""
    tweet = tweet.replace('aujourhui', "aujourdhui")
    tweet = tweet.replace("aujourd'hui", "aujourdhui")
    tweet = tweet.replace(u'&#65039;','')
    tweet = tweet.replace('franceinsoumise000','lfi')

    return tweet


def detect_language(text):
    try :
        language = detect(text)
    except:
        #print(f"translator skipped '{text}'")
        return('')

    return language


def iterate_preprocess(raw_tweets, raw_labels, screen_names, stemmer_flag):
    tweets = list() # processed tweets
    labels = list() # processed labels
    n = len(screen_names)
    for handle in range(n):
        #print(f'[tweet_processor::iterate_preprocess]: Preprocessing {screen_names[handle]}...')
        tweet, label = preprocess(raw_tweets[handle], raw_labels[handle], stemmer_flag)
        tweets.append(tweet)
        labels.append(label)
        #print(f'[tweet_processor::iterate_preprocess]: Completed preprocessing {screen_names[handle]}\n')

    return tweets, labels


def preprocess(raw_tweets, labels, stemming_flag = True):
    """
    raw_tweets, labels: list
    Performs various tweet pre-processing steps:
        1. remove non-French tweets
        2. replace accented characters
        3. processes surnames
        4. remove slogans & signatures
        5. removes URLs
        6. removes emojis
        7. removes punctuation
        8. set to lower case
        9. filters stopwords (2 steps)
        10. (optionally) stemming

        returns list of tweets as strings and labels
    """
    tv_flag = False

    if stemming_flag == True : print("*** Stemming performed ***")
    #if tv_flag == False: print('*** No TV/Radio mapping ***')

    preprocessed_tweets = list()
    del raw_tweets[0] # remove header row
    del labels[0]

    tweet_index = 0
    for tweet in raw_tweets: #tweet is string raw_tweets is list()
        # Process only French tweets
        DetectorFactory.seed = 0
        if detect_language(tweet) == 'fr':
            #print(tweet, '\n')
            # Set tweets to lower case
            tweet = tweet.lower()
            tweet = tweet.replace(u'\xa0', u' ')
            tweet = tweet.replace(u'\n', u' ')
            tweet = tweet.replace(u',', u' ')
            tweet = tweet.replace(u'.', u' ')
            tweet = tweet.replace('il faut', 'devoir')

            # Replace accented characters
            tweet = process_accents(tweet)

            # Replace stopwords
            tweet = remove_stopwords(tweet)

            # Take URLs out of tweets (before punctuation)
            tweet = remove_urls(tweet)

            # Remove punctuation
            tweet = remove_punctuation(tweet)

            # Take emojis out of tweets (may be redundant with broader remove_non_ascii)
            tweet = remove_emoji(tweet)

            # Remove non-ascii characters
            tweet = remove_non_ascii(tweet)

            # unify different spellings of politicians last names
            tweet = process_surnames(tweet)

            # perform project-specific tweet-tweaking
            tweet = process_idioms(tweet)

            # Map all tv & radio shows to TV_RADIO
            if tv_flag: tweet = map_to_tv(tweet)

            # Remove non-ascii characters

            # Remove stopwords a second time
            tweet = remove_stopwords(tweet)

            # Last ditch filter
            tweet = last_tweak(tweet)

            # Optionally perform stemming
            if stemming_flag: tweet = perform_stemming(tweet)

            preprocessed_tweets.append(tweet)

            tweet_index += 1
        else :
            del labels[tweet_index]

    print(f'Retaining {len(preprocessed_tweets)} tweets out of {len(raw_tweets)} raw tweets')
    return preprocessed_tweets, labels


if __name__ == '__main__':
    # pass the handle of the target user
    handle = 'JLMelenchon'

    # Load the data from the csv file
    raw_tweets, class_labels = load_data_by_name(handle)
    print(class_labels[0], raw_tweets[0])
    print(class_labels[1], raw_tweets[1])

    # Send to pre-processing routines
    stemmer_flag = False # set to True to perform stemming
    processed_tweets = preprocess(raw_tweets, stemmer_flag)