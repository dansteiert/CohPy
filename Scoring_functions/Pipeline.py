from Helper.Helper_functions import mean_of_list, split_at_charset, split_into_sentences, POS_tagger
from Scoring_functions.Lexical_word_level import concreteness_score
from Scoring_functions.Statistics_word_level import word_length, syllable_count
from Scoring_functions.Statistics_sentence_level import length_aggregator_list
from Scoring_functions.Lexical_sentence_level import type_token_ratio, lexical_diversity, pronoun_resolution, content_functional_ratio
from Scoring_functions.Statistics_document_level import word_frequency, Flescher_Kincaid_Grade_Level, Flescher_Reading_Ease
from Scoring_functions.Cohesion_Sentence_Sentence import tag_overlap, sentiment_shift


def pipeline(text, language, w2v_model, tagger, conc_dict):
    
    if language == "de":
        from Tagsets.Tagset_de import nouns_accept_tags, nouns_accept_tags_start_with, nouns_exclude_tags, nouns_exclude_tags_start_with
        from Tagsets.Tagset_de import pronouns_accept_tags, pronouns_accept_tags_start_with, pronouns_exclude_tags, pronouns_exclude_tags_start_with
        from Tagsets.Tagset_de import verbs_accept_tags, verbs_accept_tags_start_with, verbs_exclude_tags, verbs_exclude_tags_start_with
        from Tagsets.Tagset_de import adverbs_accept_tags, adverbs_accept_tags_start_with, adverbs_exclude_tags, adverbs_exclude_tags_start_with
        from Tagsets.Tagset_de import adjectives_accept_tags, adjectives_accept_tags_start_with, adjectives_exclude_tags, adjectives_exclude_tags_start_with
        from Tagsets.Tagset_de import punctuation_accept_tags, punctuation_accept_tags_start_with, punctuation_exclude_tags, punctuation_exclude_tags_start_with
        from Tagsets.Tagset_de import punctuation_fin_accept_tags, punctuation_fin_accept_tags_start_with, punctuation_fin_exclude_tags, punctuation_fin_exclude_tags_start_with
        from Tagsets.Tagset_de import conjugations_accept_tags, conjugations_accept_tags_start_with, conjugations_exclude_tags, conjugations_exclude_tags_start_with
        from Tagsets.Tagset_de import logical_accept_tags, logical_accept_tags_start_with, logical_exclude_tags, logical_exclude_tags_start_with
        from Tagsets.Tagset_de import count_accept_tags, count_accept_tags_start_with, count_exclude_tags, count_exclude_tags_start_with
    
    elif language == "en":
        from Tagsets.Tagset_en import nouns_accept_tags, nouns_accept_tags_start_with, nouns_exclude_tags, nouns_exclude_tags_start_with
        from Tagsets.Tagset_en import pronouns_accept_tags, pronouns_accept_tags_start_with, pronouns_exclude_tags, pronouns_exclude_tags_start_with
        from Tagsets.Tagset_en import verbs_accept_tags, verbs_accept_tags_start_with, verbs_exclude_tags, verbs_exclude_tags_start_with
        from Tagsets.Tagset_en import adverbs_accept_tags, adverbs_accept_tags_start_with, adverbs_exclude_tags, adverbs_exclude_tags_start_with
        from Tagsets.Tagset_en import adjectives_accept_tags, adjectives_accept_tags_start_with, adjectives_exclude_tags, adjectives_exclude_tags_start_with
        from Tagsets.Tagset_en import punctuation_accept_tags, punctuation_accept_tags_start_with, punctuation_exclude_tags, punctuation_exclude_tags_start_with
        from Tagsets.Tagset_en import punctuation_fin_accept_tags, punctuation_fin_accept_tags_start_with, punctuation_fin_exclude_tags, punctuation_fin_exclude_tags_start_with
        from Tagsets.Tagset_en import conjugations_accept_tags, conjugations_accept_tags_start_with, conjugations_exclude_tags, conjugations_exclude_tags_start_with
        from Tagsets.Tagset_en import logical_accept_tags, logical_accept_tags_start_with, logical_exclude_tags, logical_exclude_tags_start_with
        from Tagsets.Tagset_en import count_accept_tags, count_accept_tags_start_with, count_exclude_tags, count_exclude_tags_start_with
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
    results_stat_word_level = [mean_word_length, mean_syllable_count]
    # </editor-fold>
    
    # <editor-fold desc="Lexical Word Level">
    conc_score_hitrate= [concreteness_score(lemma=l, conc_dict=conc_dict) for l in lemma_by_sentence]
    mean_concreteness = mean_of_list([i[0] for i in conc_score_hitrate])
    hitrate_concreteness = mean_of_list([i[1] for i in conc_score_hitrate])
    results_lex_word_level = [mean_concreteness, hitrate_concreteness]
    
    # </editor-fold>
    
    # <editor-fold desc="Statistical Sentence Level">
    # TODO: Exclude Punctuation?
    mean_sentence_length = mean_of_list([len(i) for i in lemma_by_sentence])
    
    mean_punctuations = mean_of_list([length_aggregator_list(aggregate=i, document_tags=i, accept_tags=punctuation_accept_tags,
                                                             accept_tags_start_with=punctuation_accept_tags_start_with,
                                                             exclude_tags=punctuation_exclude_tags,
                                                             exclude_tags_start_with=punctuation_exclude_tags_start_with) for i in
                                      tags_by_sentence])
    
    count_conjugations = mean_of_list([length_aggregator_list(aggregate=t, document_tags=t, accept_tags=conjugations_accept_tags,
                                                              accept_tags_start_with=conjugations_accept_tags_start_with,
                                                              exclude_tags=conjugations_exclude_tags,
                                                              exclude_tags_start_with=conjugations_exclude_tags_start_with)for t in tags_by_sentence])
    
    
    
    count_logicals = mean_of_list([length_aggregator_list(aggregate=t,document_tags=t, accept_tags=logical_accept_tags,
                                                          accept_tags_start_with=logical_accept_tags_start_with,
                                                          exclude_tags=logical_exclude_tags,
                                                          exclude_tags_start_with=logical_exclude_tags_start_with) for t in tags_by_sentence])
    results_stat_sentence_level = [mean_sentence_length, mean_punctuations, count_conjugations, count_logicals]
    
    # </editor-fold>
    
    # <editor-fold desc="Lexical Sentence Level">
    # TODO: add content_functional_ratio
    
    
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
    
    pronoun_noun_ratio = pronoun_resolution(document_tags=tags, nouns_accept_tags=nouns_accept_tags,
                                            nouns_accept_tags_start_with=nouns_accept_tags_start_with,
                                            nouns_exclude_tags=nouns_exclude_tags,
                                            nouns_exclude_tags_start_with=nouns_exclude_tags_start_with,
                                            pronouns_accept_tags=pronouns_accept_tags,
                                            pronouns_accept_tags_start_with=pronouns_accept_tags_start_with,
                                            pronouns_exclude_tags=pronouns_exclude_tags,
                                            pronouns_exclude_tags_start_with=pronouns_exclude_tags_start_with)
    
    # TODO: Tagsets!!
    cont_func_ratio = content_functional_ratio(document_tags=tags, accept_tags=verbs_accept_tags,
                                               accept_tags_start_with=verbs_accept_tags_start_with,
                                               exclude_tags=verbs_exclude_tags,
                                               exclude_tags_start_with=verbs_exclude_tags_start_with)
    # </editor-fold>
    results_lex_sent_level = [mean_lexical_diversity, pronoun_noun_ratio, type_token_ratio_nouns,
                              type_token_ratio_verbs, type_token_ratio_adverbs, type_token_ratio_adjectives,
                              type_token_ratio_all_tags, cont_func_ratio]
    
    # </editor-fold>
    
    # <editor-fold desc="Statistics Document Level">
    # TODO by groups?
    (count_repeated_words, num_word_repetitions, word_freq_ratio) = word_frequency(document_lemma=lemma, document_tags=tags,
                                                                                   accept_tags=count_accept_tags,
                                                                                   accept_tags_start_with=count_accept_tags_start_with,
                                                                                   exclude_tags=count_exclude_tags,
                                                                                   exclude_tags_start_with=count_exclude_tags_start_with)
    
    
    FRE = Flescher_Reading_Ease(document_words=words, document_syllables=syllables_list,
                                num_sentences=len(lemma_by_sentence))
    FKGL = Flescher_Kincaid_Grade_Level(document_words=words, document_syllables=syllables_list,
                                        num_sentences=len(lemma_by_sentence))
    
    results_stat_doc_level = [count_repeated_words, num_word_repetitions, word_freq_ratio, FRE, FKGL]
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
    
    mean_sentiment_shift, mean_sentiment_hitrate = sentiment_shift(w2v_model=w2v_model, lemma_by_segment=lemma_by_sentence,
                                                                   tags_by_segment=tags_by_sentence,
                                                                   accept_tags=nouns_accept_tags,
                                                                   accept_tags_start_with=nouns_accept_tags_start_with,
                                                                   exclude_tags=nouns_exclude_tags,
                                                                   exclude_tags_start_with=nouns_exclude_tags_start_with)
    
    results_coh_sent_sent = [nouns_overlap, pronouns_overlap, verbs_overlap, adverbs_overlap, adjectives_overlap,
                             mean_sentiment_shift, mean_sentiment_hitrate]
    # </editor-fold>
    
    list_of_results =[results_stat_word_level, results_lex_word_level, results_stat_sentence_level, results_lex_sent_level,
                      results_stat_doc_level, results_coh_sent_sent]
    
    return list_of_results