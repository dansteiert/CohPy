from COhMatrix_scorings import *
from Treetagger import POS_tagger
from w2v_model import *
from Ratio_Scores import *
from Count_Scores import *
from Overlap_Scores import *
from Word_scorings import *


def pipeline(text, language, w2v_model, tagger, list_dict_conc):
    
    
    if language == "de":
        # <editor-fold desc="German Tagset">
        ## Nouns:
        nouns_accept_tags = []
        nouns_accept_tags_start_with = ["N"]
        nouns_exclude_tags = []
        nouns_exclude_tags_start_with = []
        ## Pronouns:
        pronouns_accept_tags = []
        pronouns_accept_tags_start_with = ["P"]
        pronouns_exclude_tags = ["PTK"]
        pronouns_exclude_tags_start_with = []
        ## Verbs:
        verbs_accept_tags = []
        verbs_accept_tags_start_with = ["V"]
        verbs_exclude_tags = []
        verbs_exclude_tags_start_with = []
        ## Adverbs:
        adverbs_accept_tags = ["ADV"]
        adverbs_accept_tags_start_with = []
        adverbs_exclude_tags = []
        adverbs_exclude_tags_start_with = []
        ## adjectives:
        adjectives_accept_tags = ["ADJA", "ADJD"]
        adjectives_accept_tags_start_with = []
        adjectives_exclude_tags = []
        adjectives_exclude_tags_start_with = []
        ## Punctuation:
        punctuation_accept_tags = []
        punctuation_accept_tags_start_with = ["$"]
        punctuation_exclude_tags = []
        punctuation_exclude_tags_start_with = []
        ## Punctuation Sentence Finishing:
        punctuation_fin_accept_tags = ["$."]
        punctuation_fin_accept_tags_start_with = []
        punctuation_fin_exclude_tags = []
        punctuation_fin_exclude_tags_start_with = []
        ## Conjugations:
        conjugations_accept_tags = []
        conjugations_accept_tags_start_with = ["K"]
        conjugations_exclude_tags = []
        conjugations_exclude_tags_start_with = []
        ## Logicals:
        logical_accept_tags = ["KON", "KOKOM"]
        logical_accept_tags_start_with = []
        logical_exclude_tags = []
        logical_exclude_tags_start_with = []
        ## Word_count:
        count_accept_tags = ["AD"]
        count_accept_tags_start_with = []
        count_exclude_tags = []
        count_exclude_tags_start_with = ["A", "C", "I", "K", "P", "T", "$"]
        # </editor-fold>
    elif language == "en":
        # <editor-fold desc="English Tagset">
        ## Tag set discription can be found at: http://www.natcorp.ox.ac.uk/docs/c5spec.html
        
        ## Nouns:
        nouns_accept_tags = []
        nouns_accept_tags_start_with = ["N"]
        nouns_exclude_tags = []
        nouns_exclude_tags_start_with = []
        ## Pronouns:
        pronouns_accept_tags = []
        pronouns_accept_tags_start_with = ["P"]
        pronouns_exclude_tags = ["PU", "PR", "PO"]
        pronouns_exclude_tags_start_with = []
        ## Verbs:
        verbs_accept_tags = []
        verbs_accept_tags_start_with = ["V"]
        verbs_exclude_tags = []
        verbs_exclude_tags_start_with = []
        ## Adverbs:
        adverbs_accept_tags = []
        adverbs_accept_tags_start_with = ["A"]
        adverbs_exclude_tags = ["AJ", "AT"]
        adverbs_exclude_tags_start_with = []
        ## adjectives:
        adjectives_accept_tags = []
        adjectives_accept_tags_start_with = ["A"]
        adjectives_exclude_tags = ["AV", "AT"]
        adjectives_exclude_tags_start_with = []
        ## Punctuation:
        punctuation_accept_tags = []
        punctuation_accept_tags_start_with = ["P", "S"]
        punctuation_exclude_tags = ["PR", "PO", "PN"]
        punctuation_exclude_tags_start_with = []
        ## Punctuation Sentence Finishing:
        punctuation_fin_accept_tags = []
        punctuation_fin_accept_tags_start_with = ["S"]
        punctuation_fin_exclude_tags = []
        punctuation_fin_exclude_tags_start_with = []
        ## Conjugations:
        conjugations_accept_tags = []
        conjugations_accept_tags_start_with = ["C"]
        conjugations_exclude_tags = ["CR"]
        conjugations_exclude_tags_start_with = []
        ## Logicals:
        logical_accept_tags = ["CJC"]
        logical_accept_tags_start_with = []
        logical_exclude_tags = []
        logical_exclude_tags_start_with = []
        ## Word_count:
        count_accept_tags = []
        count_accept_tags_start_with = []
        count_exclude_tags = ["AT0", "CRD", ]
        count_exclude_tags_start_with = ["S", "P", "D", "I", "E", "T", "X", "Z"]
        # </editor-fold>
    else:
        print("no fitting language found")
        return -1

    # print(datetime.datetime.now(), "preprocessing started")
    # <editor-fold desc="Preprocessing">
    segmented = split_at_newline(text=text, sep="\n\n")
    wtl = [POS_tagger(tagger=tagger, document=i) for i in segmented]
    words_by_seg = [i[0] for i in wtl]
    tags_by_seg = [i[1] for i in wtl]
    lemma_by_seg = [i[2] for i in wtl]
    # lemma_by_seg_noun = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=nouns_accept_tags,
    #                                           accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
    #                                           exclude_tags_start_with =nouns_exclude_tags_start_with)]
    # lemma_by_seg_adverb = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=adverbs_accept_tags,
    #                                           accept_tags_start_with=adverbs_accept_tags_start_with, exclude_tags=adverbs_exclude_tags,
    #                                           exclude_tags_start_with =adverbs_exclude_tags_start_with)]
    # lemma_by_seg_adjective = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=adjectives_accept_tags,
    #                                           accept_tags_start_with=adjectives_accept_tags_start_with, exclude_tags=adjectives_exclude_tags,
    #                                           exclude_tags_start_with =adjectives_exclude_tags_start_with)]
    # lemma_by_seg_verb = [i[2] for i in wtl if check_tags(tag=i[1] , accept_tags=verbs_accept_tags,
    #                                           accept_tags_start_with=verbs_accept_tags_start_with, exclude_tags=verbs_exclude_tags,
    #                                           exclude_tags_start_with =verbs_exclude_tags_start_with)]
    # lemma_by_seg_pronoun = [i[2] for i in wtl if check_tags(tag=i[1], accept_tags=pronouns_accept_tags,
    #                                accept_tags_start_with=pronouns_accept_tags_start_with, exclude_tags=pronouns_exclude_tags,
    #                                exclude_tags_start_with =pronouns_exclude_tags_start_with)]

    words = [j for i in wtl for j in i[0]]
    tags = [j for i in wtl for j in i[1]]
    lemma = [j for i in wtl for j in i[2]]

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

    # print(datetime.datetime.now(), "Count Scores started")
    # <editor-fold desc="Count Scores">
    mean_word_length = word_length(document_word=words)
    syllables_list = syllable_count(document_words=words)
    mean_syllables = mean_of_list(syllables_list)
    count_logicals = count_tags(document_tags=tags, accept_tags=logical_accept_tags,
                                accept_tags_start_with=logical_accept_tags_start_with,
                                exclude_tags=logical_exclude_tags,
                                exclude_tags_start_with=logical_exclude_tags_start_with)
    count_conjugations = count_tags(document_tags=tags, accept_tags=conjugations_accept_tags,
                                    accept_tags_start_with=conjugations_accept_tags_start_with,
                                    exclude_tags=conjugations_exclude_tags,
                                    exclude_tags_start_with=conjugations_exclude_tags_start_with)
    mean_sentence_length = mean_of_list([len(i) for i in lemma_by_sentence])
    mean_punctuations = mean_of_list([count_tags(document_tags=i, accept_tags=punctuation_accept_tags,
                                                 accept_tags_start_with=punctuation_accept_tags_start_with,
                                                 exclude_tags=punctuation_exclude_tags,
                                                 exclude_tags_start_with=punctuation_exclude_tags_start_with) for i in
                                      tags_by_sentence])
    mean_lexical_diversity = mean_of_list([lexical_diversity(document_tags=i, accept_tags=[], accept_tags_start_with=[],
                                                             exclude_tags=punctuation_accept_tags,
                                                             exclude_tags_start_with=punctuation_accept_tags_start_with)
                                           for i in tags_by_sentence])
    # print(lemma, tags,count_accept_tags,count_accept_tags_start_with,count_exclude_tags,count_exclude_tags_start_with, sep="\n")
    (count_repeated_words, num_word_repetitions) = word_repetition(document_lemma=lemma, document_tags=tags,
                                                                   accept_tags=count_accept_tags,
                                                                   accept_tags_start_with=count_accept_tags_start_with,
                                                                   exclude_tags=count_exclude_tags,
                                                                   exclude_tags_start_with=count_exclude_tags_start_with)

    # TODO: add POS-Frequency and Connective Words
    # </editor-fold>

    # print(datetime.datetime.now(), "Ratio Scores started")
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

    pronoun_noun_ratio = pronoun_resolution(document_tags=tags, nouns_accept_tags=nouns_accept_tags,
                                            nouns_accept_tags_start_with=nouns_accept_tags_start_with,
                                            nouns_exclude_tags=nouns_exclude_tags,
                                            nouns_exclude_tags_start_with=nouns_exclude_tags_start_with,
                                            pronouns_accept_tags=pronouns_accept_tags,
                                            pronouns_accept_tags_start_with=pronouns_accept_tags_start_with,
                                            pronouns_exclude_tags=pronouns_exclude_tags,
                                            pronouns_exclude_tags_start_with=pronouns_exclude_tags_start_with)

    # TODO: add content_functional_ratio
    # </editor-fold>

    # print(datetime.datetime.now(), "Overlap Scores started")
    # <editor-fold desc="Overlaps">
    nouns_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg,
                                   accept_tags=nouns_accept_tags,
                                   accept_tags_start_with=nouns_accept_tags_start_with, exclude_tags=nouns_exclude_tags,
                                   exclude_tags_start_with=nouns_exclude_tags_start_with)
    ## Not Sure about Pronouns
    # print(datetime.datetime.now(), "Noun Overlap Scores finished")
    # pronouns_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg, accept_tags=pronouns_accept_tags,
    #                                   accept_tags_start_with=pronouns_accept_tags_start_with, exclude_tags=pronouns_exclude_tags,
    #                                   exclude_tags_start_with =pronouns_exclude_tags_start_with)
    # print(datetime.datetime.now(), "Pronoun Overlap Scores finished")

    adverbs_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg,
                                     accept_tags=adverbs_accept_tags,
                                     accept_tags_start_with=adverbs_accept_tags_start_with,
                                     exclude_tags=adverbs_exclude_tags,
                                     exclude_tags_start_with=adverbs_exclude_tags_start_with)
    # print(datetime.datetime.now(), "Adverb Overlap Scores finished")

    adjectives_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg,
                                        accept_tags=adjectives_accept_tags,
                                        accept_tags_start_with=adjectives_accept_tags_start_with,
                                        exclude_tags=adjectives_exclude_tags,
                                        exclude_tags_start_with=adjectives_exclude_tags_start_with)
    # print(datetime.datetime.now(), "Adjective Overlap Scores finished")

    verbs_overlap = overlap_matrix(lemma_by_segment=lemma_by_seg, tags_by_segment=tags_by_seg,
                                   accept_tags=verbs_accept_tags,
                                   accept_tags_start_with=verbs_accept_tags_start_with, exclude_tags=verbs_exclude_tags,
                                   exclude_tags_start_with=verbs_exclude_tags_start_with)
    # print(datetime.datetime.now(), "Verb Overlap Scores finished")

    # </editor-fold>

    # print(datetime.datetime.now(), "Concretness Score started")
    # <editor-fold desc="Concretness Score">
    mean_concretness_score, hitrate_conc = mean_concretness(lemma=lemma, list_dict=list_dict_conc)

    # </editor-fold>

    # print(datetime.datetime.now(), "Sentiment Overlap started")
    # <editor-fold desc="Sentiment Overlap">
    sentiment_overlap, sentiment_hitrate = overlap_matrix_sentiment(w2v_model=w2v_model, lemma_by_segment=lemma_by_seg,
                                                                    tags_by_segment=tags_by_seg,
                                                                    accept_tags=nouns_accept_tags,
                                                                    accept_tags_start_with=nouns_accept_tags_start_with,
                                                                    exclude_tags=nouns_exclude_tags,
                                                                    exclude_tags_start_with=nouns_exclude_tags_start_with)
    ## TODO: w2v model does not load properly! (only the smaller one)
    # </editor-fold>

    # print(datetime.datetime.now(), "Other Scores (3) started")
    # <editor-fold desc="Other Scores">
    # co_ref = co_reference_matrix(document_tag=tags, document_lemma=lemma)
    FRE = Flescher_Reading_Ease(document_words=words, document_syllables=syllables_list,
                                num_sentences=len(lemma_by_sentence))
    FKGL = Flescher_Kincaid_Grade_Level(document_words=words, document_syllables=syllables_list,
                                        num_sentences=len(lemma_by_sentence))
    # </editor-fold>

    # print(datetime.datetime.now(), "Topic Modeling started")
    # <editor-fold desc="Topic Modeling">
    # TODO: redo with LDA Mallet!
    # ######## Model building
    # # TODO: structure for document vs corpus
    # dictionary, doc_freq_matrix, tfidf = preprocessing(corpus_tokens=lemmas)
    # lsa_model = LSA(df_matrix=doc_freq_matrix, dictionary=dictionary)
    # lda_model = LDA(df_matrix=doc_freq_matrix, dictionary=dictionary)
    topic_overlap = None
    # </editor-fold>

    # print(datetime.datetime.now(), "Result output")
    list_of_results = [mean_word_length, mean_syllables, count_logicals, count_conjugations, mean_sentence_length,
                       mean_punctuations, mean_lexical_diversity, type_token_ratio_nouns,
                       type_token_ratio_verbs, type_token_ratio_adverbs, type_token_ratio_adjectives,
                       FRE, FKGL, count_repeated_words, num_word_repetitions, mean_concretness_score, hitrate_conc,
                       nouns_overlap, verbs_overlap, adverbs_overlap, adjectives_overlap, sentiment_overlap, sentiment_hitrate, topic_overlap]

    # list_of_results_names = ["mean_word_length", "mean_syllables", "count_logicals", "count_conjugations",
    #                          "mean_sentence_length",
    #                          "mean_punctuations", "mean_lexical_diversity", "type_token_ratio_nouns",
    #                          "type_token_ratio_verbs", "type_token_ratio_adverbs", "type_token_ratio_adjectives",
    #                          "FRE", "FKGL", "count_repeated_words", "num_word_repetitions", "mean_concretness",
    #                          "hitrate_conc",
    #                          "nouns_overlap", "verbs_overlap", "adverbs_overlap", "adjectives_overlap"]
    # result_dict = {name: val for name, val in zip(list_of_results_names, list_of_results)}
    # return result_dict
    return list_of_results