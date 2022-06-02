import nltk

nltk.download("stopwords")
nltk.download("popular")


# %% [markdown]
#
#
#
#
# ## BERT Extractive Summarizer
# Summarize the text using BERT extractive summarizer. This is used to find important sentences and useful sentences from the complete text.

# %%
from summarizer import Summarizer

with open("egypt.txt", "r") as f:
    full_text = f.read()

model = Summarizer()
result = model(full_text, min_length=60, max_length=500, ratio=0.4)

summarized_text = "".join(result)


# %% [markdown]
# ## Keyword Extraction
# Get important keywords from the text and filter those keywords that are present in the summarized text.

# %%
import itertools
import re
import string

import pke
from nltk.corpus import stopwords


def get_nouns_multipartite(text):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text)
    #    not contain punctuation marks or stopwords as candidates.
    pos = {"PROPN"}
    # pos = {'VERB', 'ADJ', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
    stoplist += stopwords.words("english")
    # extractor.candidate_selection(pos=pos, stoplist=stoplist)
    extractor.candidate_selection(pos=pos)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1, threshold=0.75, method="average")
    keyphrases = extractor.get_n_best(n=20)

    return [key[0] for key in keyphrases]


keywords = get_nouns_multipartite(full_text)
filtered_keys = [
    keyword for keyword in keywords if keyword.lower() in summarized_text.lower()
]


# %% [markdown]
# ## Sentence Mapping
# For each keyword get the sentences from the summarized text containing that keyword.

from flashtext import KeywordProcessor

# %%
from nltk.tokenize import sent_tokenize


def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences


def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key, values in keyword_sentences.items():
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences


sentences = tokenize_sentences(summarized_text)
keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)


# %% [markdown]
# ## Generate MCQ
# Get distractors (wrong answer choices) from Wordnet/Conceptnet and generate MCQ Questions.

import json
import random
import re

# %%
import requests
from conceptnet_lite import Label, edges_between, edges_for
from nltk.corpus import wordnet as wn
from pywsd.lesk import adapted_lesk, cosine_lesk, simple_lesk
from pywsd.similarity import max_similarity

# %%
conceptnet_lite.connect()


# %%
# Distractors from Wordnet
def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


def get_wordsense(sent, word):
    word = word.lower()

    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    if synsets := wn.synsets(word, "n"):
        wup = max_similarity(sent, word, "wup", pos="n")
        adapted_lesk_output = adapted_lesk(sent, word, pos="n")
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None


# Distractors from http://conceptnet.io/
def get_distractors_conceptnet(word):
    word = word.lower()
    original_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    distractor_list = []

    try:
        obj = Label.get(text=word, language="en").concepts
        edges = edges_for(obj, same_language=True)
    except Exception:
        return distractor_list
    for edge in edges:
        try:
            link = edge.end.text
            obj2 = Label.get(text=link, language="en").concepts
            edges2 = edges_for(obj2, same_language=True)
            for edge in edges2:
                if edge.relation.name in [
                    "PartOf",
                    "SimilarTo",
                    "DistinctFrom",
                    "MannerOf",
                ]:
                    word2 = edge.start.text
                    if (
                        word2 not in distractor_list
                        and original_word.lower() != word2.lower()
                    ):
                        distractor_list.append(word2.replace("_", " "))
        except Exception:
            continue

    return distractor_list


key_distractor_list = {}

for keyword in keyword_sentence_mapping:
    if wordsense := get_wordsense(keyword_sentence_mapping[keyword][0], keyword):
        distractors = get_distractors_wordnet(wordsense, keyword)
        if len(distractors) == 0:
            distractors = get_distractors_conceptnet(keyword)
    else:
        distractors = get_distractors_conceptnet(keyword)
    if len(distractors) != 0:
        key_distractor_list[keyword] = distractors


# %%
final_output = {"question_set": []}
for each in key_distractor_list:
    sentence = keyword_sentence_mapping[each][0]
    pattern = re.compile(each, re.IGNORECASE)
    output = pattern.sub(" _______ ", sentence)
    choices = [each.capitalize()] + key_distractor_list[each]
    top4choices = choices[:4]
    random.shuffle(top4choices)
    final_output["question_set"].append(
        {"question": output, "choices": top4choices, "answer": each}
    )
