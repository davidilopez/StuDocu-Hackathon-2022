import itertools
import json
import random
import re
import string
import conceptnet_lite

import nltk
import pke
import requests
from conceptnet_lite import Label, edges_for
from flashtext import KeywordProcessor
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from pywsd.lesk import adapted_lesk, cosine_lesk, simple_lesk
from pywsd.similarity import max_similarity

nltk.download("stopwords")
nltk.download("popular")

from summarizer import Summarizer


class MCQ:
    # Class constructor. Initialize the dependencies.
    def __init__(self):
        nltk.download("stopwords")
        nltk.download("popular")
        conceptnet_lite.connect()

    def bert_summarizer(self, full_text):
        model = Summarizer()
        result = model(full_text, min_length=60, max_length=500, ratio=0.4)

        return "".join(result)

    def get_nouns_multipartite(self, text):
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {"PROPN", "NOUN"}
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method="average")
        keyphrases = extractor.get_n_best(n=20)

        return [key[0] for key in keyphrases]

    def tokenize_sentences(self, text):
        sentences = [sent_tokenize(text)]
        sentences = [y for x in sentences for y in x]
        # Remove any short sentences less than 20 letters.
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_sentences_for_keyword(self, keywords, sentences):
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

    # Distractors from Wordnet
    def get_distractors_wordnet(self, syn, word):
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
            if name not in distractors:
                distractors.append(name)
        return distractors

    def get_wordsense(self, sent, word):
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
    def get_distractors_conceptnet(self, word):
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

    def run_pipeline(self, full_text):
        summarized_text = self.bert_summarizer(full_text)
        keywords = self.get_nouns_multipartite(full_text)
        filtered_keys = [
            keyword
            for keyword in keywords
            if keyword.lower() in summarized_text.lower()
        ]

        sentences = self.tokenize_sentences(summarized_text)
        keyword_sentence_mapping = self.get_sentences_for_keyword(
            filtered_keys, sentences
        )

        key_distractor_list = {}

        for keyword in keyword_sentence_mapping:
            if wordsense := self.get_wordsense(
                keyword_sentence_mapping[keyword][0], keyword
            ):
                distractors = self.get_distractors_wordnet(wordsense, keyword)
                if len(distractors) == 0:
                    distractors = self.get_distractors_conceptnet(keyword)
            else:
                distractors = self.get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors

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

        return final_output
