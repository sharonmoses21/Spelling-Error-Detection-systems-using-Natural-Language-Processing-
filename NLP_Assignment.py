import re  # tokenize and convert to lower case
from collections import Counter  # counts words
import nltk  # for POS tagging
import customtkinter as ctk

# nltk.download('averaged_perceptron_tagger')    # only if needed

threshold = 0  # probability value below real-words will be checked
proposals = 10  # how many proposals should be made for wrong words
distBonus = 2  # bonus for distance 1 words to allow them higher priority before distance 2 and 3

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("Spelling & Grammar Checker")
root.geometry("720x550")


# correction of wrong punctuation --> likely not needed (word by word check). And higher identification of questions marks difficult.
# edit distance can be improved. For example, watr --> has distance 1 to water and war. Options are:
#       with similarity between proposed word and original (and not only frequency). OR
#       weights for certain letters which are more likely to be misspelled (see slide deck 5, slide 29). OR
#       prioritize addition higher than other operations (likely not true for all words/cases) OR
#       with Part of speech --> if "watr" appears after an article or determiner like "the," it is more likely that
#           "water" is correct. Requires use of libraries (then they could use for other task as well and easier). OR
#       semantic analysis of text. If text is about certain topics (ocean, sailing, military) certain words are more likely OR
#       focus on logic words instead all possible combinations (e.g., common mix-up table, no three vocals following each other, ...)
# Backtrace can be used to find shorter distances between words
# Stochastic POS Taggers

# System inherent: bigrams show high p (1) if one exotic bigram gets used
#

# Test cases **************************
# Turnbin goes sout to the river bead
# in thiese case
# a bee is an insect
# the likelihood too calculate
# Tee watr are dirty
# Hi. How are you, doing? Doesn't is It: is.
# Did you have tee later?
# I have to go to teh tolet and then go shoping at the grocery stor
# watr (position 7) or south (position 8 or 9)
# stor or tolet (only one addition but distance 2 overrules - for, to, top, ...)

# **********************************************************************
# ***** Function definition ********************************************
# **********************************************************************

def tokenize(text):
    """
    converts text into lower case and tokenize it, It also removes all punctuations (except ')
    :param text: string to be converted
    :return: a list of single words in lower case
    """
    return re.findall(r"[\w']+", text.lower())  # preserves '


def tokenizeSent(text):
    """
    converts text into lower case and tokenize sentences, It also removes all punctuations (except ') and saves them
    :param text: string to be converted
    :return: a list of single words in lower case and a dictionary of found puncation
   """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''  # all possible punctations for split
    split_words = []  # resulting words
    current_word = ''
    punctos = {}
    pos = 0

    text = text.lower()

    for char in text:
        if char.isalpha() or char == "'":  # is alpha checks that all characters are letters (and only them). Plus '
            current_word += char  # build word from scratch
        elif char in punctuations:  # if it is an punctuation
            if current_word:  # if previous build word is not empty
                split_words.append(current_word)  # full word found and added to split list
                current_word = ''  # reset collecting variable
            pos += 1  # stores the amount of sentences or sub-strings
            punctos[pos] = char  # saving name and position of punctuation
        elif char.isspace():  # required for empty spaces
            if current_word:  # if previous word is not empty (not beginning)
                current_word += char

    if current_word:  # last word doesn't has space or punctuation at and. Need to be added separate
        split_words.append(current_word)

    return split_words, punctos


def loadCorpus(path):
    """

    :param path:
    """
    file = open(path, "r", encoding="utf8")  # open file in read mode
    # corpus = file.read()               # read whole file
    corpus = tokenize(file.read())  # read whole file
    file.close()  # close file
    # corpus = pd.read_csv(path)
    return corpus


def probability(word):
    """
    calculates probability of 'word' in corpus
    :param word: word which probability will be calculated
    :return: probability value (float)
    """
    N = sum(wCount.values())  # total sum of all words
    return wCount[word] / N  # probability of one specific word in corpus


def ngrams(text, n):
    """
    create n-grams out of text
    :return: list of n-grams
    """
    result = [tuple(text[i:i + n]) for i in range(len(text) - n + 1)]
    return result


def findBigrams(text):
    """
    Creates a dictionary with bigrams and their number of occurrences
    :param text: corpus which should bigramed
    :return: dictionary of bigrams and their count
    """
    ngram = ngrams(text, n=2)

    bigram_counts = {}
    for bigram in ngram:
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    return bigram_counts


def checkBigramAlternatives(gram):
    output = {}
    if len(findUnknown([gram[0]])) == 0 and len(
            findUnknown([gram[1]])) == 0:  # if words exist (no unknown word returned)
        if gram not in bigrams:  # if this bigram doesn't exist
            output[gram] = 0  # zero occurrence probability
        else:
            v1 = bigrams[gram]  # probability of a bigram is as nominator the count of the bigram
            v2 = wCount[gram[0]]  # divided by the count of the previous word
            output[gram] = (v1 / v2)  # probability of each bigram is stored
    else:  # unknown word will be returned with 0 occurrence probability
        output[gram] = 0  # zero occurrence probability

    return output


def calculateProb(gram):
    if gram in bigrams:
        v1 = bigrams[gram]  # probability of a bigram is as nominator the count of the bigram
        v2 = wCount[gram[0]]  # divided by the count of the previous word
        return (v1 / v2)  # probability of each bigram is stored
    else:
        return 0


def checkBigram(gram):
    """
    Checks if bigram exists
    :param gram: the bigram in question
    :return: True if the bigram exist, false if not
    """
    if len(findUnknown([gram[0]])) == 0 and len(
            findUnknown([gram[1]])) == 0:  # if both words exist (no unknown word returned)
        return True
    else:
        return False  # if one word don't exist return false


def executeReplacement(pgram, gram, text):
    """
    find and replace real word issues with new word
    :param pgram: previous bigram to make sure that change doesn't worsen the probability there
    :param gram: the gram which has an issue
    :param text: Sentence in which bigram should be replaced
    :return:
    """
    global threshold

    # if this bigram doesn't exist or if probability of this bigram is not higher than threshold
    if gram not in bigrams or calculateProb(gram) <= threshold:
        realProposals = realWordCheck(gram)  # find all possible alternative bigrams
        if realProposals:  # returns true if not empty (meaning alternatives found)
            userInputTextbox.tag_remove("gram_highlight", 1.0, ctk.END)
            # Highlight all the real-words
            startIndex = 1.0
            while True:
                startIndex = userInputTextbox.search(gram[0], startIndex, stopindex=ctk.END)
                if not startIndex:
                    break
                endIndex = f"{startIndex}+{len(gram[0])}c"
                userInputTextbox.tag_add("gram_highlight", startIndex, endIndex)
                startIndex = endIndex

            # Call the Grammar Suggestions Display function when a highlighted word is clicked
            userInputTextbox.bind("<Button-1>", lambda event: gramSuggestionsDisplay(realProposals, event))

        else:  # no alternative found (with first part of bigram)
            # if no proposals are made because issue might have been second word
            realProposals = realWordCheckInv(gram)  # find all possible alternative bigrams based on second word
            if realProposals:  # returns true if not empty (meaning alternatives found)
                userInputTextbox.tag_remove("gram_highlight", 1.0, ctk.END)
                # Highlight all the real-words
                startIndex = 1.0
                while True:
                    startIndex = userInputTextbox.search(gram[1], startIndex, stopindex=ctk.END)
                    if not startIndex:
                        break
                    endIndex = f"{startIndex}+{len(gram[1])}c"
                    userInputTextbox.tag_add("gram_highlight", startIndex, endIndex)
                    startIndex = endIndex

                # Call the Grammar Suggestions Display function when a highlighted word is clicked
                userInputTextbox.bind("<Button-1>", lambda event: gramSuggestionsDisplay(realProposals, event))


def lemCheck(gram):
    """
    create lemmatized word of bigram and check if this creates better results then current bigram
    :param gram: the gram which has an issue
    :return: a bigram with or without lemmatized word
    """
    global threshold

    lemWord = lemLookUp.get(gram[0], gram[0])  # find lemmatization of first part of bigram
    curProb = calculateProb(gram)  # get likelyloohd of current bigram
    newgram = (lemWord, gram[1])  # create new bigram with lemmatized word
    newProb = calculateProb(newgram)  # what is probability of lemmatized bigram

    if newProb > curProb:  # if the new bigram is better than change word permantly to lemmatized version
        return newgram
    else:  # if not better or no change keep it as is
        return gram


def realWordCheck(gram):
    """
    finds bigrams which could be used instead of 'gram'. Selection is based on highest probability
    :param gram: gram which need to be replaced
    :return: a list of possible replacement bigram sorted according to probability
    """
    global proposals
    global threshold
    results = {}  # store all found possibilities

    # find alternative words for first word of bigram with distance 1 and 2 and 3
    alternatives = knownWords(editDDistance1(gram[0])) + knownWords(editDistance2(gram[0])) + knownWords(
        editDistance3(gram[0]))
    for word in alternatives:  # go through each proposal
        key = (word, gram[1])  # create new bigram with these words
        results[key] = calculateProb(key)  # and calculate the chance of this new bigram to occur

    # add orginal bigram to list in case all alternatives are worse
    results[gram] = calculateProb(gram)

    # to find most likely replacements dictionary need to be sorted
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    top5 = []
    i = 0
    for win, value in results:  # find the highest probabilities
        if i < proposals:  # only store top 5 replacement words
            i += 1  # only allow certain amount of proposed spelling corrections
            if value > threshold:  # if word exists (and is better then threshold / previous bigram)
                top5.append(win[0])
        else:
            break

    return top5


def realWordCheckInv(gram):
    """
    finds bigrams which could be used instead of 'gram'. Selection is based on highest probability. Check for second word
    :param gram: gram which need to be replaced
    :return: a list of possible replacement bigram sorted according to probability
    """
    # inverted bigram ****************************
    global proposals
    global threshold
    results = {}  # store all found possibilities

    # find alternative words for first word of bigram with distance 1 and 2 and 3
    # alternatives = knownWords(editDDistance1(gram[1])) + knownWords(editDistance2(gram[1])) + knownWords(editDistance3(gram[1]))
    alternatives = knownWords(editDDistance1(gram[1])) + knownWords(editDistance2(gram[1]))
    for word in alternatives:  # go through each proposal
        key = (gram[0], word)  # create new bigram with these words
        results[key] = calculateProb(key)  # and calculate the chance of this new bigram to occure

    # add orginal bigram to list in case all alternatives are worse
    results[gram] = calculateProb(gram)

    # to find most likely replacements dictionary need to be sorted
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    top5 = []
    i = 0
    for win, value in results:  # find the highest probabilities
        if i < proposals:  # only store top 5 replacement words
            i += 1  # only allow certain amount of proposed spelling corrections
            if value > threshold:  # if word exists (and is better then threshold / previous bigram)
                top5.append(win[1])
        else:
            break

    return top5


def removePunctuation(text):
    """
    remove all special characters
    :param text: string or word which need to be removed of special characters
    :return: a cleared list
    """
    punctuations = '.,;:"&%$!@-_'  # elements which need to be removed
    result = []
    for word in text:  # go through element in text
        if word not in punctuations:  # ignore punctuations
            result.append(word)  # and form punctation free text

    return result


def spellingCorrection(w, prevW, nextW):
    """
    run spelling correction functions
    :param w: word which should be corrected
    :param nextW: word after w
    :return:
    """
    global proposals
    global distBonus
    altDict = {}  # store all found possibilities
    results = {}  # store final proposals

    # find all words with distance 1 or 2 and only existing words
    alternatives = knownWords(editDDistance1(w))
    for element in alternatives:  # go through all potential new words
        altDict[element] = probability(
            element) * distBonus  # add probability of this word as multiple (because it is distance 1)
    alternatives = knownWords(editDistance2(w))
    for element in alternatives:  # go through all potential new words
        altDict[element] = probability(element)  # add probability of this word

    # POS tag list:
    """
            CC coordinating conjunction
            CD cardinal digit
            DT determiner
            EX existential there (like: "there is" ... think of it like "there exists")
            FW foreign word
            IN preposition/subordinating conjunction
            JJ adjective 'big'
            JJR adjective, comparative 'bigger'
            JJS adjective, superlative 'biggest'
            LS list marker 1)
            MD modal could, will
            NN noun, singular 'desk'
            NNS noun plural 'desks'
            NNP proper noun, singular 'Harrison'
            NNPS proper noun, plural 'Americans'
            PDT predeterminer 'all the kids'
            POS possessive ending parent's
            PRP personal pronoun I, he, she
            PRP$ possessive pronoun my, his, hers
            RB adverb very, silently,
            RBR adverb, comparative better
            RBS adverb, superlative best
            RP particle give up
            TO to go 'to' the store.
            UH interjection errrrrrrrm
            VB verb, base form take
            VBD verb, past tense took
            VBG verb, gerund/present participle taking
            VBN verb, past participle taken
            VBP verb, sing. present, non-3d take
            VBZ verb, 3rd person sing. present takes
            WDT wh-determiner which
            WP wh-pronoun who, what
            WP$ possessive wh-pronoun whose
            WRB wh-abverb where, when
            """

    if prevW:  # if a previous word exist (not empty)
        if nextW:  # if a next word exist (not empty)
            prevPOS = nltk.pos_tag([prevW])[0][1]  # find POS tags of one word ahead and one back
            nextPOS = nltk.pos_tag([nextW])[0][1]

            # to find most likely replacements dictionary need to be sorted (and only return key of dictionary)
            sortAlt = sorted(altDict.items(), key=lambda x: x[1], reverse=True)
            #            alternatives = sorted(altDict, key=lambda x: altDict[x], reverse=True)
            alternatives = [i[0] for i in sortAlt[:proposals * 2]]  # only twice the proposals will be used

            for biG in alternatives:  # go through each proposal
                valid = True  # flag to see if current proposal makes sense
                # POS tagged section
                wPOS = nltk.pos_tag([biG])[0][1]
                # define for POS rules (and their violation)
                if prevPOS == wPOS == nextPOS:  # very unlikely that same POS tag is three times in a row
                    valid = False
                if prevPOS == "DT" and not (wPOS != "JJ" or wPOS != "JJR" or wPOS != "JJS" or
                                            wPOS != "NN" or wPOS != "NNS" or wPOS != "NNP" or wPOS != "NNPS"):  # after a DT either a JJ or NN should come
                    valid = False
                if prevPOS == "IN" and not (
                        wPOS != "NN" or wPOS != "NNS" or wPOS != "NNP" or wPOS != "NNPS"):  # after a preposition a noun should come
                    valid = False
                if wPOS == "IN" and not (nextPOS != "NN" or nextPOS != "NNS" or nextPOS != "NNP" or nextPOS != "NNPS"):
                    valid = False
                if wPOS == "DT" and not (nextPOS != "JJ" or nextPOS != "JJR" or nextPOS != "JJS" or
                                         nextPOS != "NN" or nextPOS != "NNS" or nextPOS != "NNP" or nextPOS != "NNPS"):
                    valid = False

                if valid:  # only check for bigrams if POS made sense
                    # Bigram section
                    # build bigrams with new words and see if they exist and which bigram has highest probability instead single word
                    key = (biG, nextW)  # create new bigram with alternatives and the next word
                    # some bigrams don't exist (corpus) so combine probability of single word and bigram
                    results[key[0]] = probability(biG)  # probability for single word
                    results[key[0]] += calculateProb(key)  # and calculate the chance of this new bigram to occure
            results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        else:  # if only a previous word exist (not empty)
            prevPOS = nltk.pos_tag([prevW])[0][1]  # find POS tags of one word ahead and one back

            # to find most likely replacements dictionary need to be sorted (and only return key of dictionary)
            sortAlt = sorted(altDict.items(), key=lambda x: x[1], reverse=True)
            alternatives = [i[0] for i in sortAlt[:proposals * 2]]  # only twice the proposals will be used

            for biG in alternatives:  # go through each proposal
                valid = True  # flag to see if current proposal makes sense
                # POS tagged section
                wPOS = nltk.pos_tag([biG])[0][1]
                # define for POS rules (and their violation)
                if prevPOS == "DT" and (wPOS != "JJ" or wPOS != "JJR" or wPOS != "JJS" or
                                        wPOS != "NN" or wPOS != "NNS" or wPOS != "NNP" or wPOS != "NNPS"):  # after a DT either a JJ or NN should come
                    valid = False
                if prevPOS == "IN" and (
                        wPOS != "NN" or wPOS != "NNS" or wPOS != "NNP" or wPOS != "NNPS"):  # after a preposition a noun should come
                    valid = False

                if valid:  # only check for bigrams if POS made sense
                    # Bigram section
                    # build bigrams with new words and see if they exist and which bigram has highest probability instead single word
                    key = (biG, nextW)  # create new bigram with alternatives and the next word
                    # some bigrams don't exist (corpus) so combine probability of single word and bigram
                    results[key[0]] = probability(biG)  # probability for single word
                    results[key[0]] += calculateProb(key)  # and calculate the chance of this new bigram to occure
            results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    elif nextW:  # if only a next word exist (not empty)
        nextPOS = nltk.pos_tag([nextW])[0][1]

        # to find most likely replacements dictionary need to be sorted (and only return key of dictionary)
        sortAlt = sorted(altDict.items(), key=lambda x: x[1], reverse=True)
        alternatives = [i[0] for i in sortAlt[:proposals * 2]]  # only twice the proposals will be used

        for biG in alternatives:  # go through each proposal
            valid = True  # flag to see if current proposal makes sense
            # POS tagged section
            wPOS = nltk.pos_tag([biG])[0][1]
            # define for POS rules (and their violation)
            if wPOS == "IN" and (nextPOS != "NN" or nextPOS != "NNS" or nextPOS != "NNP" or nextPOS != "NNPS"):
                valid = False
            if wPOS == "DT" and (nextPOS != "JJ" or nextPOS != "JJR" or nextPOS != "JJS" or
                                 nextPOS != "NN" or nextPOS != "NNS" or nextPOS != "NNP" or nextPOS != "NNPS"):
                valid = False

            if valid:  # only check for bigrams if POS made sense
                # Bigram section
                # build bigrams with new words and see if they exist and which bigram has highest probability instead single word
                key = (biG, nextW)  # create new bigram with alternatives and the next word
                # some bigrams don't exist (corpus) so combine probability of single word and bigram
                results[key[0]] = probability(biG)  # probability for single word
                results[key[0]] += calculateProb(key)  # and calculate the chance of this new bigram to occure
        results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # no preceding or follow-up word exist
    else:
        results = sorted(altDict.items(), key=lambda x: x[1], reverse=True)

    top5 = []
    i = 0
    for win, value in results:  # find the highest probabilities
        if i < proposals:  # only store top 5 replacement words
            i += 1  # only allow certain amount of proposed spelling corrections
            if value > 0:  # if word exists (and don't have probability 0)
                top5.append(win)
        else:
            break

    return top5


def replaceWord(word, text, substitute):
    """
    replace one word (word) within a text (text) with a new word (subsitute)
    :return: new string with corrected word
    """
    result = text
    wordIndex = text.lower().find(word)  # if the word is part of input text (should be)
    if wordIndex != -1:  # if word was found
        if text[wordIndex].isupper():  # if word starts with capital letter
            substitute = substitute.capitalize()
            word = word.capitalize()
        result = text.replace(word, substitute)

    return result


def findUnknown(words):
    """
    checks what words in 'words' are not part of the corpus --> non-words
    :param words: words to be checked
    :return: if not existing returns the word if all can be found return an empty list
    """
    result = []
    for w in words:  # for all words to be checked (e.g. input sentence of user)
        if w not in wCount:  # if this is an unknown word --> non-word
            result.append(w)  # add
    return result


def knownWords(words):
    """
    checks what words in 'words' are part of the corpus --> real-words
    :param words: words to be checked
    :return: if existing returns the word if all can be found return an empty list
    """
    # existing = set(w for w in words if w in wCount)
    result = []
    for w in words:  # for all words to be checked (e.g. input sentence of user)
        if len(w) > 1 or w == 'a' or w == 'i':  # not for b, c, d, ...
            if w in wCount:  # if this is a unknown word --> non-word
                result.append(w)  # add
    return result


def editDDistance1(word):
    """
    find all combination of words which are one distance away from 'word'. Damerau-Levenshtein Distance
    insertion and deletion are distance 1. Swapping and replacement as well distance 1 (as they represent most common
    typing/spelling errors.
    :param word: input word
    :return: list of all possible words which are distance 1 away
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'  # possible input letters (only lower case)
    splits = [(word[:i], word[i:]) for i in range(
        len(word) + 1)]  # separate word into chunks (first letter and rest, first two letters and rest, and so on)
    # distance operations
    deletes = [L + R[1:] for L, R in splits if R]  # delete one letter
    inserts = [L + c + R for L, R in splits for c in letters]  # add each possible letter in between current position
    swapping = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]  # swap this and next letter
    # not for a or I or any other one-letter word. Otherwise, replace generates a list of all letters (because most are included in corpus)
    if len(word) > 1:
        replaces = [L + c + R[1:] for L, R in splits if R for c in
                    letters]  # change current letter with each possible letter
    else:
        replaces = list(word)  # for one letter words keep them
    result = (deletes + inserts + swapping + replaces)  # combine all operations in one list
    return list(set(result))  # set creates a unique list (removes double entries)


def editDistance1(word):
    """
    find all combination of words which are one distance away from 'word'. Levenshtein Distance
    Only insertion and deletion are distance 1
    typing/spelling errors.
    :param word: input word
    :return: list of all possible words which are distance 1 away
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'  # possible input letters (only lower case)
    splits = [(word[:i], word[i:]) for i in range(
        len(word) + 1)]  # separate word into chunks (first letter and rest, first two letters and rest, and so on)
    # distance operations
    deletes = [L + R[1:] for L, R in splits if R]  # delete one letter
    inserts = [L + c + R for L, R in splits for c in letters]  # add each possible letter in between current position
    result = (deletes + inserts)  # combine all operations in one list
    return list(set(result))  # set creates a unique list (removes double entries)


def editDistance2(word):
    """
    find all words with distance 2
    Two insertion, two deletion, one insertion and one deletion,
    one swap, or one replacement (same as distance 1 - otherwise words would deviate to much)
    :param word: original word
    :return: all words which have a distance of 2 to 'word'
    """
    result = []
    for e1 in editDDistance1(word):  # find all words which are one distance away
        for e2 in editDistance1(
                e1):  # out of all these '1-distance-words' find all words which are one distance away --> making in total 2 distance away
            result.append(e2)
    return list(set(result))  # set creates a unique list (removes double entries)


def editDistance3(word):
    """
    find all words with distance 3 (for real-words needed)
    :param word: original word
    :return: all words which have a distance of 3 to 'word'
    """
    result = []
    for e1 in editDDistance1(word):  # find all words which are one distance away
        for e2 in editDistance1(
                e1):  # out of all these '1-distance-words' find all words which are one distance away --> making in total 2 distance away
            for e3 in editDistance1(
                    e2):  # out of all these '2-distance-words' find all words which are one distance away --> making in total 3 distance away
                result.append(e3)

    return list(set(result))  # set creates a unique list (removes double entries)


def findNext(word, sentence):
    """
    finds word after "word" in sentence
    """
    # search string (required for match command). \b for whole word. \s checks for whitespace. \w finds characters after the whitespace
    pattern = r"\b" + re.escape(word) + r"\s+(\w+)"
    match = re.search(pattern, sentence)

    if match:
        return match.group(1)
    else:
        return ""


def findPrev(word, sentence):
    """
    finds word before "word" in sentence
    """
    # search string (required for match command). \b for whole word. \s checks for whitespace. \w finds characters after the whitespace
    pattern = r"\b(\w+)\s+" + re.escape(word)
    match = re.search(pattern, sentence)

    if match:
        return match.group(1)
    else:
        return ""


# **********************************************************************
# ***** Main program ***************************************************
# **********************************************************************

# corpus = tokenize("Hence, this isn't an easy test! Let's still try it?!")
filePath = "C:/Users/Tinashe Frekainos II/Downloads/eng-simple_wikipedia_2021_100K-sentences.txt"
corpus = loadCorpus(filePath)
wCount = Counter(corpus)  # counts all words and stores result in dictionary
# wPOS = nltk.pos_tag(list(wCount.keys()))    # creates a tuple for each word with it part-of-speech
bigrams = findBigrams(corpus)  # creates bigrams
lemLookUp = {
    "are": "is",
    "was": "be",
    "were": "be",
    "running": "run",
    "did": "do",
    "has": "have",
    "dogs": "dog",  # needed?? Very tedious
    "cats": "cat",
}  # was intended for lemmatization (to check for wrong grammar use in was, has, ...)


def spellingSuggestionsDisplay(wordsList, sentence, event=None):
    suggestionTextbox.delete(1.0, ctk.END)
    index = userInputTextbox.index(ctk.CURRENT)
    word = userInputTextbox.get(index + "wordstart", index + " wordend")   # Get the clicked highlighted word

    if word in wordsList:
        newWords = spellingCorrection(word, findPrev(word, sentence), findNext(word, sentence))
        if newWords:  # returns true if it is not empty
            # Display all the available suggestions for the clicked highlighted word
            for item in newWords:
                suggestionTextbox.insert(ctk.END, item + "\n")

        # Call the Change Spelling function when a suggestion is chosen
        suggestionTextbox.bind("<Button-1>", lambda e: spellingChange(word, event))


def spellingChange(word, event=None):
    phrase = ""
    if event:
        newIndex = suggestionTextbox.index(ctk.CURRENT)
        newWord = suggestionTextbox.get(newIndex + "wordstart", newIndex + "wordend")       # Get the chosen suggestion
        phrase = userInputTextbox.get(1.0, ctk.END).replace(word, newWord)                  # Get the current phrase and replace the chosen mispelled word with the chosen replacement
        userInputTextbox.delete(1.0, ctk.END)           # Clear the textbox
        userInputTextbox.insert(ctk.END, phrase)        # Insert the revised phrase

    nonWordProcess()


def gramSuggestionsDisplay(gramProposals, event=None):
    suggestionTextbox.delete(1.0, ctk.END)
    index = userInputTextbox.index(ctk.CURRENT)
    word = userInputTextbox.get(index + "wordstart", index + " wordend")

    # Display all the available suggestions for the clicked highlighted word
    for item in gramProposals:
        suggestionTextbox.insert(ctk.END, item + "\n")

    # Call the Change Grammar function when a suggestion is chosen
    suggestionTextbox.bind("<Button-1>", lambda e: gramChange(word, event))


def gramChange(word, event=None):
    phrase = ""
    if event:
        newIndex = suggestionTextbox.index(ctk.CURRENT)
        newWord = suggestionTextbox.get(newIndex + "wordstart", newIndex + "wordend")       # Get the chosen suggestion
        phrase = userInputTextbox.get(1.0, ctk.END).replace(word, newWord)                  # Get the current phrase and replace the chosen mispelled word with the chosen replacement
        userInputTextbox.delete(1.0, ctk.END)           # Clear the textbox
        userInputTextbox.insert(ctk.END, phrase)        # Insert the revised phrase
        suggestionTextbox.delete(1.0, ctk.END)          # Clear suggestion box


def dictionaryDisplay(*args):
    document = open(filePath, 'r', encoding="utf8").read().split()

    wordSearch = searchTextEntry.get().strip()
    dictionaryTextbox.delete(1.0, ctk.END)

    wordSearch = re.sub(r'[^\w\s]', '', wordSearch)
    matchingWords = set()
    for wordCheck in document:
        if wordCheck.startswith(wordSearch):
            wordCheck = re.sub(r'[^\w\s]+$', '', wordCheck)
            matchingWords.add(wordCheck)

    for wordCheck in matchingWords:
        dictionaryTextbox.insert(ctk.END, wordCheck + "\n")


def nonWordProcess(event=None):
    # Clear suggestions textbox and get userinput from main textbox
    userinput = userInputTextbox.get(1.0, ctk.END)
    userInputTextbox.tag_remove("word_highlight", 1.0, ctk.END)
    # handle sentence by sentence ****************
    userSentences, punctuations = tokenizeSent(
        userinput)  # splitting the text into sentences and store punctuations for later use
    if len(punctuations) == 0:  # if no punctation was entered and variable is empty
        punctuations[1] = "."

    s = 0
    fullText = ""
    for sentence in userSentences:
        s += 1

        # non-word section ***************************
        uWords = findUnknown(tokenize(sentence))  # find all unknown words in user input.
        newText = sentence
        if len(uWords) >= 1:  # if unknown words exist
            for word in uWords:
                newWords = spellingCorrection(word, findPrev(word, sentence), findNext(word, sentence))
                if newWords:  # returns true if it is not empty
                    startIndex = 1.0
                    # Highlight all the non-words in the sentence
                    while True:
                        startIndex = userInputTextbox.search(word, startIndex, stopindex=ctk.END)
                        if not startIndex:
                            break
                        endIndex = f"{startIndex}+{len(word)}c"
                        userInputTextbox.tag_add("word_highlight", startIndex, endIndex)
                        startIndex = endIndex
                if event:
                    # Call the Spelling Suggestions Display function when a highlighted word is clicked
                    spellingSuggestionsDisplay(uWords, sentence, event)


def realWordProcess(event=None):
    # Clear suggestions textbox and get userinput from main textbox
    suggestionTextbox.delete(1.0, ctk.END)
    userinput = userInputTextbox.get(1.0, ctk.END)
    # handle sentence by sentence ****************
    userSentences, punctuations = tokenizeSent(
        userinput)  # splitting the text into sentences and store punctuations for later use
    if len(punctuations) == 0:  # if no punctation was entered and variable is empty
        punctuations[1] = "."

    s = 0
    for sentence in userSentences:
        s += 1

        newText = sentence
        # real word section ***************************
        text = tokenize(newText)  # tokenize input text
        realCorrection = newText
        if len(text) > 1:  # if at least two words were entered a real-word check is possible
            inputgram = ngrams(text, n=2)  # creates bigrams of the input string
            prevgram = 0  # check any change also on previous bigram
            for gram in inputgram:  # go through each bigram
                if checkBigram(gram):  # check if bigrams only contains real words
                    lgram = lemCheck(gram)  # check if lemmatized word is better than current
                    realCorrection = replaceWord(gram[0], realCorrection,
                                                 lgram[0])  # also change the running sentence
                    gram = (lgram[0], gram[1])  # create new/old bigram
                executeReplacement(prevgram, gram, realCorrection)


userInputTextbox = ctk.CTkTextbox(root, height=450, width=450, font=('Arial', 15))
userInputTextbox.place(x=20, y=40)

suggestionsLabel = ctk.CTkLabel(root, text="Suggestions")
suggestionsLabel.place(x=570, y=10)

suggestionTextbox = ctk.CTkTextbox(root, height=150, width=200, font=('Arial', 15))
suggestionTextbox.place(x=500, y=40)

dictionaryLabel = ctk.CTkLabel(root, text="Dictionary")
dictionaryLabel.place(x=570, y=210)

searchTextEntry = ctk.CTkEntry(root, height=30, width=200, font=('Arial', 15))
searchTextEntry.place(x=500, y=240)

dictionaryTextbox = ctk.CTkTextbox(root, height=210, width=200, font=('Arial', 15))
dictionaryTextbox.place(x=500, y=280)

checkButton = ctk.CTkButton(root, height=30, width=120, text="Check Spelling", command=nonWordProcess)
checkButton.place(x=350, y=500)

checkGramButton = ctk.CTkButton(root, height=30, width=120, text="Check Grammar", command=realWordProcess)
checkGramButton.place(x=220, y=500)

userInputTextbox.insert(1.0, "Type here...")
userInputTextbox.tag_config("word_highlight", background="red")
userInputTextbox.tag_bind("word_highlight", "<Button-1>", nonWordProcess)
userInputTextbox.tag_config("gram_highlight", background="blue")
userInputTextbox.tag_bind("gram_highlight", "<Button-1>", realWordProcess)
suggestionTextbox.configure(cursor='arrow')
searchTextEntry.bind("<KeyRelease>", dictionaryDisplay)

root.mainloop()