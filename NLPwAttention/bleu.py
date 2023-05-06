import nltk
import numpy


reference = "The NASA Opportunity rover is battling a massive dust storm on planet Mars."
candidate_1 = "The Opportunity rover is combating a big sandstorm on planet Mars."
candidate_2 = "A NASA rover is fighting a massive storm on planet Mars."

nltk.word_tokenize(reference.lower())
tokenizerd_ref = ()
tokenizerd_cand1 = candidate_1.lower()
tokenizerd_cand2 = candidate_2.lower()

tokenized_ref = nltk.word_tokenize(reference.lower())
""