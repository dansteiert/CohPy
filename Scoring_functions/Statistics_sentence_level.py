from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list


def mean_tags_by_sentence(tagsets_by_doc, tagset_name, document_sentence):
    '''
    Ref: Pitler08 - Elements of lexical cohesion: # Pronouns; # definite articles - only Tagset for articles available though
    Calculate how often a given POS tag occurce in a sentence
    :param tagsets_by_doc: dict, {key=tagset names, value=dict{key=lemma, value=absolute word count}}
    :param tagset_name: str, tagset name for which the mean occurrance is calculated
    :param document_sentence: int, # of sentences in the document
    :return: float,  how often a given POS tag occurce in a sentence
    '''
    tagset = tagsets_by_doc.get(tagset_name, {})
    return tagset.values()/document_sentence
    

def stat_sentence_length(lemma_by_sent):
    '''
    Ref: Pitler08 - Baseline measures
    Calculate the mean sentence length and the maximal sentence length for the document
    :param lemma_by_sent: list, [list=[str=lemma]]
    :return: float, int
    '''
    sent_lengt = [len(i) for i in lemma_by_sent]
    if len(sent_lengt) > 0:
        return mean_of_list(sent_lengt), max(sent_lengt)
    else:
        return 0, 0
