from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list

def mean_tags_by_sentence(tagsets_by_doc, tagset_name):
    '''
    Ref: Pitler08 - Elements of lexical cohesion: # Pronouns; # definite articles - only Tagset for articles available though
    
    :param tagsets_by_doc
    :param tagset_name
    :return
    '''
    tagset = tagsets_by_doc.get(tagset_name, {})
    return mean_of_list(tagset.values())
    


def stat_sentence_length(lemma_by_sent):
    '''
    Ref: Pitler08 - Baseline measures

    :param lemma_by_sent:
    :return:
    '''
    sent_lengt = [len(i) for i in lemma_by_sent]
    return (mean_of_list(sent_lengt), max(sent_lengt))