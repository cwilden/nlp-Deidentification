import sklearn
import xmltodict
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
import pycountry
import us
import pprint
import xlrd
import string
import re
import pandas as pd
import sklearn_crfsuite
import os
import sys
import pickle

#global word2id dictionary and lists of X_train, Y_train for model later
word_ids = dict()
X_train = []
Y_train = []

#Function to retrieve all features for document
#Input: all_tokens: list of all words in the document
#sentences: list of all sentences in the document
#Output: list of all features for document
def get_features(all_tokens, sentences):
    #--------------------------------------------------------------------------
    #First feature: Bag of words (unigram, bigram and trigram [-2, 2])
    #--------------------------------------------------------------------------
    bow = dict()

    #UNIGRAM
    for token in all_tokens:
        if(token not in bow.keys()):
            bow[token] = 1
        else:
            bow[token] += 1

    for sentence in sentences:
        for index, word in enumerate(sentence.split()):
            #BIGRAM
            if(index < len(sentence.split()) - 1):
                nextword = sentence.split()[index + 1]
                bigram = word + ' ' + nextword
                if(bigram not in bow.keys()):
                    bow[bigram] = 1
                else:
                    bow[bigram] += 1

            #TRIGRAM
            if(index < len(sentence.split()) - 2):
                secword = sentence.split()[index + 1]
                thirdword = sentence.split()[index + 2]
                trigram = word + ' ' + secword + ' ' + thirdword
                if(trigram not in bow.keys()):
                    bow[trigram] = 1
                else:
                    bow[trigram] += 1


    #REDUCED BAG OF WORDS: keep only words that appear more than once
    reducedbow = dict()
    for key, count in bow.items():
        if(count > 1):
            reducedbow[key] = count


    # --------------------------------------------------------------------------
    # Second feature: Tuples of (token, Part of speech POS tag) Ex. ('fly', 'NN')
    #--------------------------------------------------------------------------
    pos_tags = nltk.pos_tag(all_tokens)

    # --------------------------------------------------------------------------
    # Third feature: List of combinations of tokens and POS tags
    # w0 p-1, w0 p0, w0 p1, w0 p-1 p0, w0 p0 p1, w0 p-1 p1, w0 p-1 p0 p1
    # w0: current word
    # p-1 p0 p1: last current and next pos tags
    #--------------------------------------------------------------------------
    all_combinations = []
    for index, token in enumerate(pos_tags):
        combination = []
        word = token[0]
        tag = token[1]
        #w0 and p-1
        if(index > 1):
            prevTup = pos_tags[index - 1]
            first = word + ' ' + prevTup[1]
            combination.append(first)

        #w0 and p0
        current = word + ' ' + tag
        combination.append(current)

        #w0 and p1
        if(index < len(pos_tags) - 1):
            nextTup = pos_tags[index + 1]
            next = word + ' ' + nextTup[1]
            combination.append(next)

        #w0 p-1 p0
        if(index > 1):
            prevTup = pos_tags[index - 1]
            dprevcur = word + ' ' + prevTup[1] + ' ' + tag
            combination.append(dprevcur)

        #w0 p0 p1
        if(index < len(pos_tags) - 1):
            nextTup = pos_tags[index + 1]
            dcurnext = word + ' ' + tag + ' ' + nextTup[1]
            combination.append(dcurnext)

        #w0 p-1 p1
        if(index > 1) and (index < len(pos_tags) - 1):
            prevTup = pos_tags[index - 1]
            nextTup = pos_tags[index + 1]
            dprevnext = word + ' ' + prevTup[1] + ' ' + nextTup[1]
            combination.append(dprevnext)

        #w0 p-1 p0 p1
        if(index > 1) and (index < len(pos_tags) - 1):
            prevTup = pos_tags[index - 1]
            nextTup = pos_tags[index + 1]
            dprevcurnext = word + ' ' + prevTup[1] + ' ' + tag + ' ' + nextTup[1]
            combination.append(dprevcurnext)

        all_combinations.append(combination)


    # --------------------------------------------------------------------------
    # Fourth feature: Sentence information
    # list of length of sentence and punctuation if present
    # --------------------------------------------------------------------------
    all_token_sent_info = []
    for sentence in sentences:
        sent_info = []
        splitted_sent = sentence.split()
        #LENGTH OF SENTENCE
        length = len(splitted_sent)
        #must be converted to bytes for model
        sent_info.append(bytes(length))

        #CHECK FOR PUNCTUATION
        last_word = splitted_sent[-1]
        last_char = last_word[-1]
        puncts = ['.', '?', '!']
        if(last_char in puncts):
            sent_info.append(last_char)

        #APPEND INFORMATION FOR EACH WORD IN SENTENCE
        for token in range(length):
            all_token_sent_info.append(sent_info)


    # --------------------------------------------------------------------------
    # Fifth feature: Affixes
    # lists of prefixes and suffixes from length 1 - 5
    # Constraint: do this for words with length > 4 only and non-digits
    # --------------------------------------------------------------------------
    all_prefixes = []
    all_suffixes = []
    for token in all_tokens:
        prefixes = []
        suffixes = []
        tokenLength = len(token)
        #prefixes
        if(tokenLength > 4) and (not token[0].isdigit()):
            for index in range(0,6):
                if(index > 0):
                    pre = token[0:index]
                    prefixes.append(pre)
        all_prefixes.append(prefixes)

        #suffixes
        if(tokenLength > 4) and (not token[0].isdigit()):
            for index in range(0,6):
                if(index > 1):
                    suff = token[len(token) - index:len(token)]
                    suffixes.append(suff)
        all_suffixes.append(suffixes)


    # --------------------------------------------------------------------------
    # Sixth feature: Wordshapes
    # lists of wordshapes: a, A for lower and uppercase; # for digits; - for punctuation
    # Ex. Document.123 -> Aaaaaaaa-###
    # --------------------------------------------------------------------------
    all_wordshapes = []
    for sentence in sentences:
        tokens = sentence.split()
        for rawtoken in tokens:
            wordshape = ''
            for char in rawtoken:
                if (char.islower()):
                    wordshape += 'a'
                if (char.isupper()):
                    wordshape += 'A'
                if (char.isdigit()):
                    wordshape += '#'
                if (char in string.punctuation):
                    wordshape += '-'
            all_wordshapes.append(wordshape)


    # --------------------------------------------------------------------------
    # Seventh feature: Section information
    # lists of section headers for each token Ex. Medications
    # --------------------------------------------------------------------------
    #Initial sections extracted from document structure
    sections = ['medications', 'allergies', 'history', 'physical', 'problems',
                'assessment', 'discharge', 'vital', 'maintenance']

    #Go through document to find section headers
    current_sections = ['']
    for tok in all_tokens:
        if(tok[-1] == ':' and len(tok) > 1) or (tok in sections):
            current_sections.append(tok.lower())
    section_tokens = []
    tokeni = 0
    sectioni = 0
    #Append to section tokens: each section per current token
    while(sectioni < len(current_sections) - 1 and tokeni < len(all_tokens) - 1):
        nextsection = current_sections[sectioni + 1]
        currenttoken = all_tokens[tokeni]
        if (currenttoken == nextsection):
            sectioni += 1
        section_tokens.append([currenttoken, nextsection])
        tokeni += 1

    #Append last section to tokens
    remaining = len(all_tokens) - len(section_tokens)
    startindex = len(all_tokens) - remaining
    for index in range(remaining):
        token = all_tokens[startindex + index]
        section_tokens.append([token, current_sections[-1]])


    # --------------------------------------------------------------------------
    # Eighth feature: Stanford Named Entity Recognition NER tags
    # PERSON, ORGANIZATION, etc.
    # --------------------------------------------------------------------------
    from nltk.tag import StanfordNERTagger
    ner_tags = []
    st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
                           'stanford-ner.jar',
                           encoding='utf-8')
    for sent in sentences:
        tokenizedtext = sent.split()
        classifiedtext = st.tag(tokenizedtext)
        for cf in classifiedtext:
            ner_tags.append(cf)


    # --------------------------------------------------------------------------
    # Ninth feature: Word2Vec with lemmatized words and unique ids
    # --------------------------------------------------------------------------
    #Lemmatize words in document
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lem_words = set()
    for token in all_tokens:
        lem_words.add(lemmatizer.lemmatize(token))

    #Assign unique ids
    for index, word in enumerate(lem_words):
        if(word not in word_ids.keys()):
            word_ids[word] = len(word_ids) + index

    #Out of vocabulary word id
    word_ids['outofvocab'] = len(word_ids) + 1

    #Convert sentences to word ids
    sentences_to_int = []
    for sent in sentences:
        lem_sent = []
        for w in sent.split():
            lem_word = lemmatizer.lemmatize(w)
            id = word_ids.get(lem_word)
            #If word is out of vocabulary
            if(id == None):
                id = word_ids.get('outofvocab')
            lem_sent.append(id)
        sentences_to_int.append(lem_sent)


    # --------------------------------------------------------------------------
    # Tenth feature: Dictionary features such as Country, State, City, Zip code
    # --------------------------------------------------------------------------
    #Lowercase countries
    countries = []
    for country in pycountry.countries:
        countries.append(country.name.lower())

    #Lowercase states
    states = set()
    for s in us.states.STATES:
        split = str(s).split(':')
        name = str(split[0]).lower()
        states.add(name)

    #Lowercase cities and zipcodes from downloaded document
    cities = set()
    zipcodes = set()
    ziploc = 'USA-Zip.xls'
    wb = xlrd.open_workbook(ziploc)
    sheet = wb.sheet_by_index(0)
    for index in range(sheet.nrows):
        cities.add(str(sheet.cell_value(index, 1)).lower())
        cities.add(str(sheet.cell_value(index, 6)).lower())
        zipcodes.add(sheet.cell_value(index, 0))
        zipcodes.add(sheet.cell_value(index, 5))
    zipcodes.remove('ZIP code')

    #Find any matches
    token_set = set(all_tokens)
    country_matches = token_set.intersection(countries)
    state_matches = token_set.intersection(states)
    city_matches = token_set.intersection(cities)
    #Cities have names that can have semantic meaning, so remove these:
    removeMatches = ['days', 'call', 'lead', 'home', '1', 'post', 'small', 'rule', 'given', 'felt', 'only', 'quality', 'may', 'page', 'light', 'hospital', 'lack', 'normal', 'axis', 'and', 'contact', 'energy', 'clear', 'arm', 'apex', 'progress']
    city_matches_list = []
    for cit in city_matches:
        if(cit not in removeMatches):
            city_matches_list.append(cit)
    zip_matches = token_set.intersection(zipcodes)


    # --------------------------------------------------------------------------
    # Return ALL Features for each token
    # --------------------------------------------------------------------------
    print('Tokens')
    print(all_tokens)
    print('POS tags')
    print(pos_tags)
    print('Combinations')
    print(all_combinations)
    print('Sentence Info')
    print(all_token_sent_info)
    print('Prefixes')
    print(all_prefixes)
    print('Suffixes')
    print(all_suffixes)
    print('Wordshapes')
    print(all_wordshapes)
    print('Section tokens')
    print(section_tokens)
    print('Named Entity Recognition tags')
    print(ner_tags)

    #List that will store dictionaries of each feature
    all_features = []
    for index, token in enumerate(all_tokens):
        #Storing each key: feature in dictionary
        feature_dict = dict()

        # Ninth feature as first : Unique ID for word
        if(token in word_ids.keys()):
            id = word_ids[token]
        else:
            id = word_ids['outofvocab']
        # must be converted to bytes for model
        feature_dict['wordid'] = bytes(id)

        #Bag of words
        if (token in reducedbow.keys()):
            feature_dict['bow'] = bytes(reducedbow[token])
        else:
            feature_dict['bow'] = bytes(len(word_ids) + 1)

        #POS tags
        feature_dict['postag'] = list(pos_tags[index])

        #Combinations of word and pos tags
        feature_dict['wordcombination'] = all_combinations[index]

        #Sentence information
        feature_dict['sentenceinfo'] = all_token_sent_info[index]

        #Affixes
        feature_dict['prefixes'] = all_prefixes[index]
        feature_dict['suffixes'] = all_suffixes[index]

        #Wordshapes
        feature_dict['wordshape'] = all_wordshapes[index]

        #Section headers
        feature_dict['section'] = section_tokens[index]

        #Stanford NER tag
        feature_dict['nertag'] = list(ner_tags[index])

        #location dictionary matches
        #Country
        if country_matches:
            if(token in country_matches):
                feature_dict['country'] = True
        else:
                feature_dict['country'] = False

        #State
        if (state_matches):
            if(token in state_matches):
                feature_dict['state'] = True
        else:
            feature_dict['state'] = False

        #Cities
        if (city_matches_list):
            if(token in city_matches_list):
                feature_dict['city'] = True
        else:
            feature_dict['city'] = False

        #Zip Codes
        if (zip_matches):
            if(token in zip_matches):
                feature_dict['zipcode'] = True
        else:
            feature_dict['zipcode'] = False

        ##Rule based features using Regex
        #Phone number variations
        phonematch1 = re.compile(r"\(\d{3}\)[- \t]?\d{3}[- \t]\d{4}")
        phonematch2 = re.compile(r"\d{3}[- \t]\d{3}[- \t]\d{4}")
        phonematch3 = re.compile(r"\d{3}\\d{1}-\d{4}")

        #Fax
        faxmatch = re.compile(r"[Ff]ax.*\d{3}[- \t]\d{3}[- \t]\d{4}")

        #Medical Record Number Format
        medrecordmatch1 = re.compile(r"\d{3}[- ]\d{2}[- ]\d{2}[- ]\d{1}")
        medrecordmatch2 = re.compile(r"\d{3}[- ]\d{2}[- ]\d{2}")

        #Email
        emailmatch = re.compile(r"\S+@\S+")

        #Ipaddr
        ipaddrmatch = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

        #Initialize features to be false
        feature_dict['phone'] = False
        feature_dict['fax'] = False
        feature_dict['medicalrecord'] = False
        feature_dict['emails'] = False
        feature_dict['ipaddr'] = False

        ###Find matches
        #Phone
        phones1 = re.findall(phonematch1, token)
        phones2 = re.findall(phonematch2, token)
        phones3 = re.findall(phonematch3, token)
        if (phones1 != None and phones1 != []) or (phones2 != None and phones2 != []) or (phones3 != None and phones3 != []):
            feature_dict['phone'] = True
        #Fax
        faxes = re.findall(faxmatch, token)
        if (faxes != None and faxes != []):
            feature_dict['fax'] = True

        #Medical Records
        medrecords1 = re.findall(medrecordmatch1, token)
        if(medrecords1 != None and medrecords1 != []):
            feature_dict['medicalrecord'] = True
        medrecords2 = re.findall(medrecordmatch2, token)
        if (medrecords2 != None and medrecords2 != []):
                feature_dict['medicalrecord'] = True

        #Email
        emails = re.findall(emailmatch, token)
        if (emails != None and emails != []):
            feature_dict['email'] = True

        #Ipaddr
        ipaddr = re.findall(ipaddrmatch, token)
        if (ipaddr != None and ipaddr != []):
            feature_dict['ipaddr'] = True

        all_features.append(feature_dict)

    pprint.pprint(all_features)
    return all_features


#Function to handle training data
def process_training_file(inputfile):
    #read in XML file
    with open(inputfile) as fd:
        doc = xmltodict.parse(fd.read())

    text = doc["deIdi2b2"]["TEXT"]
    trainingTags = doc["deIdi2b2"]["TAGS"]

    # --------------------------------------------------------------------------
    #            PRE-PROCESSING
    # --------------------------------------------------------------------------
    # Convert to list of sentences
    splitted = text.splitlines()
    stripsplit = []
    for s in splitted:
        if(s != ''):
            stripsplit.append(s.strip())

    training_sentences = []
    for sentence in stripsplit:
        if(len(sentence) > 0):
            training_sentences.append(sentence)

    # --------------------------------------------------------------------------
    #                TOKENIZE
    # --------------------------------------------------------------------------
    training_tokens = []
    def train_tokenize(sentence):
        for word in sentence.split():
            training_tokens.append(word.lower())

    for sentence in training_sentences:
        train_tokenize(sentence)

    # --------------------------------------------------------------------------
    #                BUILD X AND Y TRAIN
    # --------------------------------------------------------------------------

    #Call get features function ()
    xtrain = get_features(training_tokens, training_sentences)
    #Append to global X_train
    X_train.append(xtrain)

    #Get training labels
    #Ex. <DATE id="P0" start="16" end="26" text="2074-12-05" TYPE="DATE" comment="" />
    training_dict = dict()
    for key, value in trainingTags.items():
        if isinstance(value, list):
            dictionary = value[0]
        else:
            dictionary = value

        token = dictionary.get('@text')
        splittoken = token.split()
        tag = dictionary.get('@TYPE')
        if(len(splittoken)) > 1:
            for tok in splittoken:
                training_dict[tok.lower()] = tag
        else:
            training_dict[token.lower()] = tag

    #Y_Train
    ytrain = []
    for token in training_tokens:
        if(token in training_dict.keys()):
            ytrain.append(training_dict.get(token))
        else:
            #NONE
            ytrain.append('NONE')
    Y_train.append(ytrain)


#Function to handle each file in folder
def select_files_in_folder(dir, ext):
    for file in os.listdir(dir):
        if file.endswith('.%s' % ext):
            yield os.path.join(dir, file)


# --------------------------------------------------------------------------
#                          MODEL
# --------------------------------------------------------------------------
for file in select_files_in_folder('training-PHI-Gold-Set1', 'xml'):
    process_training_file(file)

crf = sklearn_crfsuite.CRF(algorithm='l2sgd', c2=0.1, max_iterations=100)
crf.fit(X_train, Y_train)
filename = "baseline.model.sav"
loaded_model = pickle.dump(crf, open(filename, 'wb'))
with open("wordDict.txt", 'wb') as handle:
    pickle.dump(word_ids, handle)
