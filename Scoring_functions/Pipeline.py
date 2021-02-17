from Helper.Helper_functions import mean_of_list, split_at_charset, split_into_sentences, POS_tagger, sort_by_POS_tags, word_frequencies
from Scoring_functions.Lexical_word_level import affinity_conc_score, mean_concreteness
from Scoring_functions.Statistics_word_level import word_length, syllable_count, word_frequency
from Scoring_functions.Statistics_sentence_level import mean_tags_by_sentence, stat_sentence_length
from Scoring_functions.Lexical_sentence_level import type_token_ratio, lexical_diversity, ratio_tags_a_to_tags_b
from Scoring_functions.Statistics_document_level import logical_incidence, connective_incidence, unique_lemma, Flescher_Kincaid_Grade_Level, Flescher_Reading_Ease
from Scoring_functions.Cohesion_Sentence_Sentence import tag_overlap, sentiment_shift, affinity_shift, tense_change

import datetime
import numpy as np

def pipeline(text, language, w2v_model, tagger, df_affinity, affinity_score_label, concreteness_label,
             df_background_corpus_frequency, background_corpus_size, df_connective, connective_type_label):
    result_dict = {}
    if language == "de":
        from Tagsets.Tagset_de import nouns_accept_tags, nouns_accept_tags_start_with, nouns_exclude_tags, nouns_exclude_tags_start_with
        from Tagsets.Tagset_de import pronouns_accept_tags, pronouns_accept_tags_start_with, pronouns_exclude_tags, pronouns_exclude_tags_start_with
        from Tagsets.Tagset_de import noun_pronouns_accept_tags, noun_pronouns_accept_tags_start_with, noun_pronouns_exclude_tags, noun_pronouns_exclude_tags_start_with
        from Tagsets.Tagset_de import verbs_accept_tags, verbs_accept_tags_start_with, verbs_exclude_tags, verbs_exclude_tags_start_with
        from Tagsets.Tagset_de import adverbs_accept_tags, adverbs_accept_tags_start_with, adverbs_exclude_tags, adverbs_exclude_tags_start_with
        from Tagsets.Tagset_de import adjectives_accept_tags, adjectives_accept_tags_start_with, adjectives_exclude_tags, adjectives_exclude_tags_start_with
        from Tagsets.Tagset_de import punctuation_accept_tags, punctuation_accept_tags_start_with, punctuation_exclude_tags, punctuation_exclude_tags_start_with
        from Tagsets.Tagset_de import punctuation_fin_accept_tags, punctuation_fin_accept_tags_start_with, punctuation_fin_exclude_tags, punctuation_fin_exclude_tags_start_with
        from Tagsets.Tagset_de import conjunctions_accept_tags, conjunctions_accept_tags_start_with, conjunctions_exclude_tags, conjunctions_exclude_tags_start_with
        from Tagsets.Tagset_de import logical_accept_tags, logical_accept_tags_start_with, logical_exclude_tags, logical_exclude_tags_start_with
        from Tagsets.Tagset_de import count_accept_tags, count_accept_tags_start_with, count_exclude_tags, count_exclude_tags_start_with
        from Tagsets.Tagset_de import content_accept_tags, content_accept_tags_start_with, content_exclude_tags, content_exclude_tags_start_with
        from Tagsets.Tagset_de import functional_accept_tags, functional_accept_tags_start_with, functional_exclude_tags, functional_exclude_tags_start_with
        from Tagsets.Tagset_de import article_accept_tags, article_accept_tags_start_with, article_exclude_tags, article_exclude_tags_start_with
        from Tagsets.Tagset_de import past_accept_tags, past_accept_tags_start_with, past_exclude_tags, past_exclude_tags_start_with
        from Tagsets.Tagset_de import present_accept_tags, present_accept_tags_start_with, present_exclude_tags, present_exclude_tags_start_with
    
    elif language == "en":
        from Tagsets.Tagset_en import nouns_accept_tags, nouns_accept_tags_start_with, nouns_exclude_tags, nouns_exclude_tags_start_with
        from Tagsets.Tagset_en import pronouns_accept_tags, pronouns_accept_tags_start_with, pronouns_exclude_tags, pronouns_exclude_tags_start_with
        from Tagsets.Tagset_en import noun_pronouns_accept_tags, noun_pronouns_accept_tags_start_with, noun_pronouns_exclude_tags, noun_pronouns_exclude_tags_start_with
        from Tagsets.Tagset_en import verbs_accept_tags, verbs_accept_tags_start_with, verbs_exclude_tags, verbs_exclude_tags_start_with
        from Tagsets.Tagset_en import adverbs_accept_tags, adverbs_accept_tags_start_with, adverbs_exclude_tags, adverbs_exclude_tags_start_with
        from Tagsets.Tagset_en import adjectives_accept_tags, adjectives_accept_tags_start_with, adjectives_exclude_tags, adjectives_exclude_tags_start_with
        from Tagsets.Tagset_en import punctuation_accept_tags, punctuation_accept_tags_start_with, punctuation_exclude_tags, punctuation_exclude_tags_start_with
        from Tagsets.Tagset_en import punctuation_fin_accept_tags, punctuation_fin_accept_tags_start_with, punctuation_fin_exclude_tags, punctuation_fin_exclude_tags_start_with
        from Tagsets.Tagset_en import conjunctions_accept_tags, conjunctions_accept_tags_start_with, conjunctions_exclude_tags, conjunctions_exclude_tags_start_with
        from Tagsets.Tagset_en import logical_accept_tags, logical_accept_tags_start_with, logical_exclude_tags, logical_exclude_tags_start_with
        from Tagsets.Tagset_en import count_accept_tags, count_accept_tags_start_with, count_exclude_tags, count_exclude_tags_start_with
        from Tagsets.Tagset_en import content_accept_tags, content_accept_tags_start_with, content_exclude_tags, content_exclude_tags_start_with
        from Tagsets.Tagset_en import functional_accept_tags, functional_accept_tags_start_with, functional_exclude_tags, functional_exclude_tags_start_with
        from Tagsets.Tagset_en import article_accept_tags, article_accept_tags_start_with, article_exclude_tags, article_exclude_tags_start_with
        from Tagsets.Tagset_en import past_accept_tags, past_accept_tags_start_with, past_exclude_tags, past_exclude_tags_start_with
        from Tagsets.Tagset_en import present_accept_tags, present_accept_tags_start_with, present_exclude_tags, present_exclude_tags_start_with

    else:
        print("Language not yet implemented - add Tagset_LANG.py file and import it in the pipline file.")

    print("#", end="")
    # <editor-fold desc="Preprocessing">
    segmented = split_at_charset(text=text, sep=["\n\n"])
    wtl = [POS_tagger(tagger=tagger, document=i) for i in segmented]
    
    # # <editor-fold desc="ParagraphsAsBoW">
    # words_by_seg = [i[0] for i in wtl]
    # tags_by_seg = [i[1] for i in wtl]
    # lemma_by_seg = [i[2] for i in wtl]
    # # </editor-fold>
    
    # <editor-fold desc="DocAsBoW">
    words = [j for i in wtl for j in i[0]]
    tags = [j for i in wtl for j in i[1]]
    lemma = [j for i in wtl for j in i[2]]
    # </editor-fold>
    
    # <editor-fold desc="SentencesAsBoW">
    lemma_by_sentence = split_into_sentences(aggregator_list=lemma, document_tags=tags,
                                             accept_tags=punctuation_fin_accept_tags,
                                             accept_tags_start_with=punctuation_fin_accept_tags_start_with,
                                             exclude_tags=punctuation_fin_exclude_tags,
                                             exclude_tags_start_with=punctuation_fin_exclude_tags_start_with)
    tags_by_sentence = split_into_sentences(aggregator_list=tags, document_tags=tags,
                                            accept_tags=punctuation_fin_accept_tags,
                                            accept_tags_start_with=punctuation_fin_accept_tags_start_with,
                                            exclude_tags=punctuation_fin_exclude_tags,
                                            exclude_tags_start_with=punctuation_fin_exclude_tags_start_with)
    # </editor-fold>
    
    document_sentences = len(lemma_by_sentence)
    document_words = len(lemma)
    
    if document_sentences > 0 and document_words > 0:
        pass
    else:
        return {}

    print("#", end="")
    # <editor-fold desc="Sort_by_tagsets">
    accept_tags = [count_accept_tags, content_accept_tags, functional_accept_tags, noun_pronouns_accept_tags,
                   punctuation_accept_tags, punctuation_fin_accept_tags, conjunctions_accept_tags, logical_accept_tags,
                   past_accept_tags, present_accept_tags]
    accept_tags_start_with = [count_accept_tags_start_with, content_accept_tags_start_with,
                              functional_accept_tags_start_with, noun_pronouns_accept_tags_start_with,
                              punctuation_accept_tags_start_with, punctuation_fin_accept_tags_start_with,
                              conjunctions_accept_tags_start_with,
                              logical_accept_tags_start_with, past_accept_tags_start_with, present_accept_tags_start_with]
    exclude_tags = [count_exclude_tags, content_exclude_tags, functional_exclude_tags, noun_pronouns_exclude_tags,
                    punctuation_exclude_tags, punctuation_fin_exclude_tags, conjunctions_exclude_tags, logical_exclude_tags,
                    past_exclude_tags, present_exclude_tags]
    exclude_tags_start_with = [count_exclude_tags_start_with, content_exclude_tags_start_with,
                               functional_exclude_tags_start_with, noun_pronouns_exclude_tags_start_with,
                               punctuation_exclude_tags_start_with, punctuation_fin_exclude_tags_start_with,
                               conjunctions_exclude_tags_start_with,
                               logical_exclude_tags_start_with, past_exclude_tags_start_with, present_exclude_tags_start_with]
    tagset_names = ["Count", "Content", "Functional", "Noun and Pronoun", "Punctuations",
                     "Punctuation Sentence Finishing", "Conjunctions", "Logical", "Past", "Present"]
    
    exclusive_accept_tags = [nouns_accept_tags, pronouns_accept_tags, verbs_accept_tags, adverbs_accept_tags,
                            adjectives_accept_tags, article_accept_tags]
    exclusive_accept_tags_start_with = [nouns_accept_tags_start_with, pronouns_accept_tags_start_with,
                                       verbs_accept_tags_start_with, adverbs_accept_tags_start_with,
                            adjectives_accept_tags_start_with, article_accept_tags_start_with]
    exclusive_exclude_tags = [nouns_exclude_tags, pronouns_exclude_tags, verbs_exclude_tags, adverbs_exclude_tags,
                            adjectives_exclude_tags, article_exclude_tags]
    exclusive_exclude_tags_start_with = [nouns_exclude_tags_start_with, pronouns_exclude_tags_start_with,
                                        verbs_exclude_tags_start_with, adverbs_exclude_tags_start_with,
                            adjectives_exclude_tags_start_with, article_exclude_tags_start_with]
    exclusive_tagset_names = ["Noun", "Pronoun", "Verb", "Adverb", "Adjective", "Article"]

    tagsets_by_sent_dict, tagsets_by_doc = sort_by_POS_tags(aggregator_by_sent=lemma_by_sentence, tags_by_sent=tags_by_sentence,
                                   accept=accept_tags, accept_star_with=accept_tags_start_with,
                                   exclude=exclude_tags, exclude_start_with=exclude_tags_start_with,
                                   order_tagsets=tagset_names,
                                   exclusive_accept=exclusive_accept_tags, exclusive_accept_star_with=exclusive_accept_tags_start_with,
                                   exclusive_exclude=exclusive_exclude_tags, exclusive_exclude_start_with=exclusive_exclude_tags_start_with,
                                   exclusive_order_tagsets=exclusive_tagset_names
                                   )

    # </editor-fold>

    print("#", end="")
    # <editor-fold desc="Word Frequencies">
    word_frequency_by_sentence_dict, word_frequency_by_document_dict = word_frequencies(lemma_by_sent=lemma_by_sentence)
    # </editor-fold>


    print("#", end="")
    # <editor-fold desc="Affinity Scores">
    affinity_concretness_label = affinity_score_label
    affinity_concretness_label.append(concreteness_label)
    (dict_affinities_by_sent, hitrate_affinities) = affinity_conc_score(lemma_dict_by_sent=word_frequency_by_sentence_dict, df_affinity=df_affinity,
                                                              affinity_conc_label=affinity_concretness_label, size_of_document=document_words)
    # </editor-fold>


    
    # </editor-fold>
    
    print("#", end="")
    # <editor-fold desc="Statistics Word Level">
    mean_word_length = word_length(document_word=words)
    syllables_list = syllable_count(document_words=words)
    mean_syllable_count = mean_of_list(syllables_list)
    log_word_freq, text_corpus_corr, unique_word_incidence = word_frequency(document_word_freq_dict=word_frequency_by_document_dict,
                                                                            document_size=document_sentences,
                                                                            df_background_corpus_frequency=df_background_corpus_frequency,
                                                                            background_corpus_size=background_corpus_size)
    
    result_dict = {**result_dict, **{"Mean word length": mean_word_length, "Mean syllable count": mean_syllable_count,
                                     "log word frequency": log_word_freq, "Vocabulary correlation": text_corpus_corr,
                                     "Unique word incidence": unique_word_incidence, "Document length (in words)": document_words,
                                     "Document length (in sentences)": document_sentences}}
    # </editor-fold>

    print("#", end="")
    # <editor-fold desc="Lexical Word Level">

    mean_concreteness_score = mean_concreteness(concreteness_label=concreteness_label, affinity_conc_dict=dict_affinities_by_sent)

    result_dict = {**result_dict, **{"Mean Concretness Score": mean_concreteness_score, "Hitrate Affinity Scores": hitrate_affinities}}
    
    # </editor-fold>

    print("#", end="")
    # <editor-fold desc="Statistical Sentence Level">
    
    (mean_sent_length, max_sentence_length) = stat_sentence_length(lemma_by_sent=lemma_by_sentence)
    
    mean_punctuations = mean_tags_by_sentence(tagsets_by_doc=tagsets_by_doc, tagset_name="Punctuations", document_sentence=document_sentences)
    
    mean_conjunctions = mean_tags_by_sentence(tagsets_by_doc=tagsets_by_doc, tagset_name="Conjunctions", document_sentence=document_sentences)
    
    mean_pronouns = mean_tags_by_sentence(tagsets_by_doc=tagsets_by_doc, tagset_name="Pronoun", document_sentence=document_sentences)
    
    mean_articles = mean_tags_by_sentence(tagsets_by_doc=tagsets_by_doc, tagset_name="Article", document_sentence=document_sentences)
    
    unique_content_incidence = unique_lemma(tagsets_by_doc=tagsets_by_doc, tagset_name="Content", document_sentences=document_sentences)

    result_dict = {**result_dict, **{"Mean sentence length": mean_sent_length, "Mean punctuation per sentence": mean_punctuations,
                                     "Mean conjunctions per sentence": mean_conjunctions, "Maximal sentence length": max_sentence_length,
                                     "Mean pronouns per sentence": mean_pronouns,
                                     "Mean articles per sentence": mean_articles, "Unique Content Incidence": unique_content_incidence}}
    
    # </editor-fold>

    print("#", end="")
    # <editor-fold desc="Lexical Sentence Level">
    cont_func_ratio = ratio_tags_a_to_tags_b(tagsets_by_doc=tagsets_by_doc, tagset_a="Content", tagset_b="Functional")
    pronoun_noun_ratio = ratio_tags_a_to_tags_b(tagsets_by_doc=tagsets_by_doc, tagset_a="Noun", tagset_b="Pronoun")
    adjective_verb_quotien = ratio_tags_a_to_tags_b(tagsets_by_doc=tagsets_by_doc, tagset_a="Adjective", tagset_b="Verb")

    
    # <editor-fold desc="Ratio Scores">
    type_token_ratio_nouns = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="Noun")
    type_token_ratio_noun_pronoun = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="Pronoun")
    type_token_ratio_pronoun = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="Noun and Pronoun")
    type_token_ratio_adverbs = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="Adverb")
    type_token_ratio_adjectives = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="Adjective")
    type_token_ratio_verbs = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="Verb")
    type_token_ratio_all_tags = type_token_ratio(tagsets_by_doc=tagsets_by_doc, tagset_name="all")
    
    

    # </editor-fold>
    mean_lexical_diversity = lexical_diversity(word_frequency_dict=word_frequency_by_document_dict, document_sentences=document_sentences)
    result_dict = {**result_dict, **{"Pronoun-noun ratio": pronoun_noun_ratio, "Content word-functional word ratio": cont_func_ratio,
                                     "Type-token ratio nouns": type_token_ratio_nouns, "Type-token ratio verbs": type_token_ratio_verbs,
                                     "Type-token ratio adverbs": type_token_ratio_adverbs, "Type-token ratio adjectives": type_token_ratio_adjectives,
                                     "Type-token ratio all words": type_token_ratio_all_tags, "Type-token ratio pronouns": type_token_ratio_pronoun},
                                    "Type-token ratio noun and pronoun": type_token_ratio_noun_pronoun, "Adjective Verb Quotient": adjective_verb_quotien,
                   "Mean lexical diversity per sentence": mean_lexical_diversity
                   }
    # </editor-fold>

    print("#", end="")
    # <editor-fold desc="Statistics Document Level">

    logical_incidence_score = logical_incidence(tagsets_by_doc=tagsets_by_doc, tagset_name="Logical", doc_words=document_words)
    
    connective_incidence_scores = connective_incidence(lemma=lemma, df_connective=df_connective, connective_type_label=connective_type_label)

    
    FRE = Flescher_Reading_Ease(document_words=words, document_syllables=syllables_list,
                                num_sentences=document_sentences)
    FKGL = Flescher_Kincaid_Grade_Level(document_words=words, document_syllables=syllables_list,
                                        num_sentences=document_sentences)
    
    result_dict = {**result_dict, **logical_incidence_score, **connective_incidence_scores, **{"Flescher Reading Ease": FRE,
                                                                                               "Flescher Kincaid Grade Level": FKGL}}
    
    # </editor-fold>

    print("#", end="")
    # <editor-fold desc="Cohesion_Sentence_Sentence">
    # <editor-fold desc="Overlaps">
    nouns_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="Noun")
    
    pronouns_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="Pronoun")
    
    noun_pronouns_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="Noun and Pronoun")
    
    adverbs_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="Adverb")
    
    adjectives_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="Adjective")
    
    verbs_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="Verb")
        
    all_words_overlap = tag_overlap(tagset_by_sent=tagsets_by_sent_dict, tagset_name="all")
    # </editor-fold>
    
    mean_tense_changes = tense_change(tagset_by_sent=tagsets_by_sent_dict, tagset_name_past="Past",
                                      tagset_name_present="Present")
    
    
    mean_sentiment_shift, mean_sentiment_hitrate = sentiment_shift(w2v_model=w2v_model, lemma_by_segment=lemma_by_sentence,
                                                                   tags_by_segment=tags_by_sentence,
                                                                   accept_tags=nouns_accept_tags,
                                                                   accept_tags_start_with=nouns_accept_tags_start_with,
                                                                   exclude_tags=nouns_exclude_tags,
                                                                   exclude_tags_start_with=nouns_exclude_tags_start_with)
    
    affinity_shift_scores = affinity_shift(affinity_score_dict=dict_affinities_by_sent, affinity_label=affinity_score_label)
    
    result_dict = {**result_dict,  **{"Noun overlap": nouns_overlap, "Pronoun overlap": pronouns_overlap,
                                                              "Noun Pronoun Overlap": noun_pronouns_overlap,
                                     "Verb Overlap": verbs_overlap, "Adverb Overlap": adverbs_overlap, "Adjective Overlap": adjectives_overlap,
                                                              "All Word Overlap": all_words_overlap,
                                     "Mean sentiment shift": mean_sentiment_shift, "Hitrate sentiment shift": mean_sentiment_hitrate,
                                      "Mean tense changes": mean_tense_changes},
                   **affinity_shift_scores,
                   }
    print(result_dict)
    # </editor-fold>
    
    
    return result_dict