from Helper.Helper_functions import mean_of_list, split_at_charset, split_into_sentences, POS_tagger
from Scoring_functions.Lexical_word_level import concreteness_score
from Scoring_functions.Statistics_word_level import word_length, syllable_count, word_frequency
from Scoring_functions.Statistics_sentence_level import length_aggregator_list
from Scoring_functions.Lexical_sentence_level import type_token_ratio, lexical_diversity, noun_pronoun_proportion, content_functional_ratio
from Scoring_functions.Statistics_document_level import logical_incidence, connective_incidence, Flescher_Kincaid_Grade_Level, Flescher_Reading_Ease
from Scoring_functions.Cohesion_Sentence_Sentence import tag_overlap, semantic_shift, affinity_shift


import numpy as np

def pipeline(text, language, w2v_model, tagger, affinity_dict, affinity_score_label, concreteness_dict,
             word_freq_dict, word_freq_corpus_size, connective_dict, pos_connective_name, neg_connective_name):
    
    if language == "de":
        from Tagsets.Tagset_de import nouns_accept_tags, nouns_accept_tags_start_with, nouns_exclude_tags, nouns_exclude_tags_start_with
        from Tagsets.Tagset_de import pronouns_accept_tags, pronouns_accept_tags_start_with, pronouns_exclude_tags, pronouns_exclude_tags_start_with
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
    
    elif language == "en":
        from Tagsets.Tagset_en import nouns_accept_tags, nouns_accept_tags_start_with, nouns_exclude_tags, nouns_exclude_tags_start_with
        from Tagsets.Tagset_en import pronouns_accept_tags, pronouns_accept_tags_start_with, pronouns_exclude_tags, pronouns_exclude_tags_start_with
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
    log_word_freq = word_frequency(lemma=lemma, word_freq_dict=word_freq_dict, word_freq_corpus_size=word_freq_corpus_size)
    
    results_stat_word_level = [mean_word_length, mean_syllable_count, log_word_freq]
    # </editor-fold>
    
    # <editor-fold desc="Lexical Word Level">
    conc_score_hitrate= [concreteness_score(lemma=l, conc_dict=concreteness_dict) for l in lemma_by_sentence]
    mean_concreteness = mean_of_list([i[0] for i in conc_score_hitrate])
    hitrate_concreteness = mean_of_list([i[1] for i in conc_score_hitrate])
    results_lex_word_level = [mean_concreteness, hitrate_concreteness]
    
    # </editor-fold>
    
    # <editor-fold desc="Statistical Sentence Level">
    mean_sentence_length = mean_of_list([len(i) for i in lemma_by_sentence])
    
    mean_punctuations = mean_of_list([length_aggregator_list(aggregate=i, document_tags=i, accept_tags=punctuation_accept_tags,
                                                             accept_tags_start_with=punctuation_accept_tags_start_with,
                                                             exclude_tags=punctuation_exclude_tags,
                                                             exclude_tags_start_with=punctuation_exclude_tags_start_with) for i in
                                      tags_by_sentence])
    
    count_conjunctions = mean_of_list([length_aggregator_list(aggregate=t, document_tags=t, accept_tags=conjunctions_accept_tags,
                                                              accept_tags_start_with=conjunctions_accept_tags_start_with,
                                                              exclude_tags=conjunctions_exclude_tags,
                                                              exclude_tags_start_with=conjunctions_exclude_tags_start_with)for t in tags_by_sentence])
    
    

    results_stat_sentence_level = [mean_sentence_length, mean_punctuations, count_conjunctions]
    
    # </editor-fold>
    
    # <editor-fold desc="Lexical Sentence Level">
    cont_func_ratio = mean_of_list([content_functional_ratio(document_tags=i, content_tags=content_accept_tags, content_tags_start_with=content_accept_tags_start_with, exclude_content_tags=content_exclude_tags,
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
    results_lex_sent_level = [mean_lexical_diversity, pronoun_noun_ratio, type_token_ratio_nouns,
                              type_token_ratio_verbs, type_token_ratio_adverbs, type_token_ratio_adjectives,
                              type_token_ratio_all_tags, cont_func_ratio]
    
    # </editor-fold>
    
    # <editor-fold desc="Statistics Document Level">

    logical_incidence_score = logical_incidence(aggregate=lemma, document_tags=tags, accept_tags=logical_accept_tags,
                                                          accept_tags_start_with=logical_accept_tags_start_with,
                                                          exclude_tags=logical_exclude_tags,
                                                          exclude_tags_start_with=logical_exclude_tags_start_with)
    
    (mean_connectiven_incidence, negative_connective_incidence, positive_connective_incidence) = connective_incidence(lemma=lemma, connective_dict=connective_dict,
                                                                                                                      name_positive_connective=pos_connective_name,
                                                                                                                      name_negative_connective=neg_connective_name)

    
    FRE = Flescher_Reading_Ease(document_words=words, document_syllables=syllables_list,
                                num_sentences=len(lemma_by_sentence))
    FKGL = Flescher_Kincaid_Grade_Level(document_words=words, document_syllables=syllables_list,
                                        num_sentences=len(lemma_by_sentence))
    
    results_stat_doc_level = [logical_incidence_score, mean_connectiven_incidence, negative_connective_incidence, positive_connective_incidence,
                              FRE, FKGL]
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
    
    # </editor-fold>
    
    mean_semantic_shift, mean_semantic_hitrate = semantic_shift(w2v_model=w2v_model, lemma_by_segment=lemma_by_sentence,
                                                                   tags_by_segment=tags_by_sentence,
                                                                   accept_tags=nouns_accept_tags,
                                                                   accept_tags_start_with=nouns_accept_tags_start_with,
                                                                   exclude_tags=nouns_exclude_tags,
                                                                   exclude_tags_start_with=nouns_exclude_tags_start_with)
    
    affinity_shift_scores = affinity_shift(lemma_by_sent=lemma_by_sentence, affinity_dict=affinity_dict, affinity_label=affinity_score_label)
    
    results_coh_sent_sent = [nouns_overlap, pronouns_overlap, verbs_overlap, adverbs_overlap, adjectives_overlap,
                             mean_semantic_shift, mean_semantic_hitrate]
    results_coh_sent_sent.extend(affinity_shift_scores)
    # </editor-fold>
    
    list_of_results =[results_stat_word_level, results_lex_word_level, results_stat_sentence_level, results_lex_sent_level,
                      results_stat_doc_level, results_coh_sent_sent]
    
    return list_of_results