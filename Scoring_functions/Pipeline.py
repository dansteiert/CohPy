from Helper.Helper_functions import mean_of_list, split_at_charset, split_into_sentences, POS_tagger
from Scoring_functions.Lexical_word_level import concreteness_score
from Scoring_functions.Statistics_word_level import word_length, syllable_count, word_frequency
from Scoring_functions.Statistics_sentence_level import length_aggregator_list, stat_sentence_length
from Scoring_functions.Lexical_sentence_level import type_token_ratio, lexical_diversity, noun_pronoun_proportion, content_functional_ratio
from Scoring_functions.Statistics_document_level import logical_incidence, connective_incidence, Flescher_Kincaid_Grade_Level, Flescher_Reading_Ease
from Scoring_functions.Cohesion_Sentence_Sentence import tag_overlap, semantic_shift, affinity_shift


import numpy as np

def pipeline(text, language, w2v_model, tagger, affinity_dict, affinity_score_label, concreteness_dict,
             word_freq_dict, word_freq_corpus_size, connective_dict):
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

    else:
        print("Language not yet implemented - add Tagset_LANG.py file and import it in the pipline file.")
    
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
    
    # </editor-fold>
    
    # <editor-fold desc="Statistics Word Level">
    mean_word_length = word_length(document_word=words)
    syllables_list = syllable_count(document_words=words)
    mean_syllable_count = mean_of_list(syllables_list)
    log_word_freq, text_corpus_corr, unique_word_incidence = word_frequency(lemma=lemma, word_freq_dict=word_freq_dict, word_dict_corpus_size=word_freq_corpus_size)
    
    result_dict = {**result_dict, **{"Mean word length": mean_word_length, "Mean syllable count": mean_syllable_count,
                                     "log word frequency": log_word_freq, "Vocabulary correlation": text_corpus_corr,
                                     "Unique word incidence": unique_word_incidence}}
    # </editor-fold>
    
    # <editor-fold desc="Lexical Word Level">
    conc_score_hitrate= [concreteness_score(lemma=l, conc_dict=concreteness_dict) for l in lemma_by_sentence]
    mean_concreteness = mean_of_list([i[0] for i in conc_score_hitrate])
    hitrate_concreteness = mean_of_list([i[1] for i in conc_score_hitrate])
    result_dict = {**result_dict, **{"Mean Concretness Score": mean_concreteness, "Hitrate Affinity Scores": hitrate_concreteness}}
    
    # </editor-fold>
    
    # <editor-fold desc="Statistical Sentence Level">
    
    (mean_sent_length, max_sentence_length, text_length) = stat_sentence_length(lemma_by_sent=lemma_by_sentence)
    
    mean_punctuations = mean_of_list([length_aggregator_list(aggregate=i, document_tags=i, accept_tags=punctuation_accept_tags,
                                                             accept_tags_start_with=punctuation_accept_tags_start_with,
                                                             exclude_tags=punctuation_exclude_tags,
                                                             exclude_tags_start_with=punctuation_exclude_tags_start_with) for i in
                                      tags_by_sentence])
    
    mean_conjunctions = mean_of_list([length_aggregator_list(aggregate=t, document_tags=t, accept_tags=conjunctions_accept_tags,
                                                              accept_tags_start_with=conjunctions_accept_tags_start_with,
                                                              exclude_tags=conjunctions_exclude_tags,
                                                              exclude_tags_start_with=conjunctions_exclude_tags_start_with)for t in tags_by_sentence])
    

    mean_pronouns = mean_of_list([length_aggregator_list(aggregate=t, document_tags=t, accept_tags=pronouns_accept_tags,
                                                              accept_tags_start_with=pronouns_accept_tags_start_with,
                                                              exclude_tags=pronouns_exclude_tags,
                                                              exclude_tags_start_with=pronouns_exclude_tags_start_with)for t in tags_by_sentence])
    

    mean_articles = mean_of_list([length_aggregator_list(aggregate=t, document_tags=t, accept_tags=article_accept_tags,
                                                              accept_tags_start_with=article_accept_tags_start_with,
                                                              exclude_tags=article_exclude_tags,
                                                              exclude_tags_start_with=article_exclude_tags_start_with)for t in tags_by_sentence])

    result_dict = {**result_dict, **{"Mean sentence length": mean_sent_length, "Mean punctuation per sentence": mean_punctuations,
                                     "Mean conjunctions per sentence": mean_conjunctions, "Maximal sentence length": max_sentence_length,
                                     "Words in Text": text_length, "Mean pronouns per sentence": mean_pronouns,
                                     "Mean articles per sentence": mean_articles}}
    
    # </editor-fold>
    
    # <editor-fold desc="Lexical Sentence Level">
    cont_func_ratio = mean_of_list([content_functional_ratio(document_tags=i, content_tags=content_accept_tags,
                                content_tags_start_with=content_accept_tags_start_with, exclude_content_tags=content_exclude_tags,
                             exclude_content_tags_start_with=content_exclude_tags_start_with, functional_tags=functional_accept_tags,
                             functional_tags_start_with=functional_accept_tags_start_with, exclude_functional_tags=functional_exclude_tags,
                             exclude_functional_tags_start_with=functional_exclude_tags_start_with) for i in tags_by_sentence])
    
    mean_lexical_diversity = mean_of_list([lexical_diversity(document_tags=i, accept_tags=[], accept_tags_start_with=[],
                                                             exclude_tags=punctuation_accept_tags,
                                                             exclude_tags_start_with=punctuation_accept_tags_start_with)
                                           for i in tags_by_sentence])
    
    # <editor-fold desc="Ratio Scores">
    type_token_ratio_nouns = type_token_ratio(document_tags=tags, accept_tags=nouns_accept_tags,
                                              accept_tags_start_with=nouns_accept_tags_start_with,
                                              exclude_tags=nouns_exclude_tags,
                                              exclude_tags_start_with=nouns_exclude_tags_start_with)
    type_token_ratio_adverbs = type_token_ratio(document_tags=tags, accept_tags=adverbs_accept_tags,
                                                accept_tags_start_with=adverbs_accept_tags_start_with,
                                                exclude_tags=adverbs_exclude_tags,
                                                exclude_tags_start_with=adverbs_exclude_tags_start_with)
    type_token_ratio_adjectives = type_token_ratio(document_tags=tags, accept_tags=adjectives_accept_tags,
                                                   accept_tags_start_with=adjectives_accept_tags_start_with,
                                                   exclude_tags=adjectives_exclude_tags,
                                                   exclude_tags_start_with=adjectives_exclude_tags_start_with)
    type_token_ratio_verbs = type_token_ratio(document_tags=tags, accept_tags=verbs_accept_tags,
                                              accept_tags_start_with=verbs_accept_tags_start_with,
                                              exclude_tags=verbs_exclude_tags,
                                              exclude_tags_start_with=verbs_exclude_tags_start_with)
    type_token_ratio_all_tags = type_token_ratio(document_tags=tags, accept_tags=[],
                                                 accept_tags_start_with=[],
                                                 exclude_tags=[],
                                                 exclude_tags_start_with=[])
    
    pronoun_noun_ratio = noun_pronoun_proportion(document_tags=tags, nouns_accept_tags=nouns_accept_tags,
                                            nouns_accept_tags_start_with=nouns_accept_tags_start_with,
                                            nouns_exclude_tags=nouns_exclude_tags,
                                            nouns_exclude_tags_start_with=nouns_exclude_tags_start_with,
                                            pronouns_accept_tags=pronouns_accept_tags,
                                            pronouns_accept_tags_start_with=pronouns_accept_tags_start_with,
                                            pronouns_exclude_tags=pronouns_exclude_tags,
                                            pronouns_exclude_tags_start_with=pronouns_exclude_tags_start_with)
    

    # </editor-fold>
    result_dict = {**result_dict, **{"Mean lexical diversity per sentence": mean_lexical_diversity, "Pronoun-noun ratio": pronoun_noun_ratio,
                                     "Type-token ratio nouns": type_token_ratio_nouns, "Type-token ratio verbs": type_token_ratio_verbs,
                                     "Type-token ratio adverbs": type_token_ratio_adverbs, "Type-token ratio adjectives": type_token_ratio_adjectives,
                                     "Type-token ratio all words": type_token_ratio_all_tags, "Content word-functional word ratio": cont_func_ratio}}
    # </editor-fold>
    
    # <editor-fold desc="Statistics Document Level">

    logical_incidence_score = logical_incidence(aggregate=lemma, document_tags=tags, accept_tags=logical_accept_tags,
                                                          accept_tags_start_with=logical_accept_tags_start_with,
                                                          exclude_tags=logical_exclude_tags,
                                                          exclude_tags_start_with=logical_exclude_tags_start_with)
    
    connective_incidence_scores = connective_incidence(lemma=lemma, connective_dict=connective_dict)

    
    FRE = Flescher_Reading_Ease(document_words=words, document_syllables=syllables_list,
                                num_sentences=len(lemma_by_sentence))
    FKGL = Flescher_Kincaid_Grade_Level(document_words=words, document_syllables=syllables_list,
                                        num_sentences=len(lemma_by_sentence))
    
    result_dict = {**result_dict, **logical_incidence_score, **connective_incidence_scores, **{"Flescher Reading Ease": FRE,
                                                                                               "Flescher Kincaid Grade Level": FKGL}}
    
    # </editor-fold>
    
    # <editor-fold desc="Cohesion_Sentence_Sentence">
    # <editor-fold desc="Overlaps">
    nouns_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence,
                                accept_tags=nouns_accept_tags,
                                accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
                                exclude_tags_start_with=nouns_exclude_tags_start_with)
    
    pronouns_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence, accept_tags=pronouns_accept_tags,
                                   accept_tags_start_with=pronouns_accept_tags_start_with, exclude_tags=pronouns_exclude_tags,
                                   exclude_tags_start_with =pronouns_exclude_tags_start_with)
    
    noun_pronouns_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence, accept_tags=noun_pronouns_accept_tags,
                                   accept_tags_start_with=noun_pronouns_accept_tags_start_with, exclude_tags=noun_pronouns_exclude_tags,
                                   exclude_tags_start_with =noun_pronouns_exclude_tags_start_with)
    
    adverbs_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence,
                                  accept_tags=adverbs_accept_tags,
                                  accept_tags_start_with=adverbs_accept_tags_start_with,
                                  exclude_tags=adverbs_exclude_tags,
                                  exclude_tags_start_with=adverbs_exclude_tags_start_with)
    
    adjectives_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence,
                                     accept_tags=adjectives_accept_tags,
                                     accept_tags_start_with=adjectives_accept_tags_start_with,
                                     exclude_tags=adjectives_exclude_tags,
                                     exclude_tags_start_with=adjectives_exclude_tags_start_with)
    
    verbs_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence,
                                accept_tags=verbs_accept_tags,
                                accept_tags_start_with=verbs_accept_tags_start_with, exclude_tags=verbs_exclude_tags,
                                exclude_tags_start_with=verbs_exclude_tags_start_with)
        
    all_words_overlap = tag_overlap(lemma_by_segment=lemma_by_sentence, tags_by_segment=tags_by_sentence,
                                accept_tags=[],
                                accept_tags_start_with=[], exclude_tags=[],
                                exclude_tags_start_with=[])
    

    
    # </editor-fold>
    
    mean_semantic_shift, mean_semantic_hitrate = semantic_shift(w2v_model=w2v_model, lemma_by_segment=lemma_by_sentence,
                                                                   tags_by_segment=tags_by_sentence,
                                                                   accept_tags=nouns_accept_tags,
                                                                   accept_tags_start_with=nouns_accept_tags_start_with,
                                                                   exclude_tags=nouns_exclude_tags,
                                                                   exclude_tags_start_with=nouns_exclude_tags_start_with)
    
    affinity_shift_scores = affinity_shift(lemma_by_sent=lemma_by_sentence, affinity_dict=affinity_dict, affinity_label=affinity_score_label)
    
    result_dict = {**result_dict, **affinity_shift_scores, **{"Noun overlap": nouns_overlap, "Pronoun overlap": pronouns_overlap,
                                                              "Noun Pronoun Overlap": noun_pronouns_overlap,
                                     "Verb Overlap": verbs_overlap, "Adverb Overlap": adverbs_overlap, "Adjective Overlap": adjectives_overlap,
                                                              "All Word Overlap": all_words_overlap,
                                     "Mean semantic shift": mean_semantic_shift, "Hitrate semantic shift": mean_semantic_hitrate}}
    # </editor-fold>
    
    
    return result_dict