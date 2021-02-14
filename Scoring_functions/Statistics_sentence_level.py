from Helper.Helper_functions import search_tag_set, to_count_dict, mean_of_list

def length_aggregator_list(aggregate, document_tags, accept_tags=[], accept_tags_start_with=["$"], exclude_tags=[],
               exclude_tags_start_with=[]):
    '''
    Ref: Pitler08 - Elements of lexical cohesion: # Pronouns; # definite articles - only Tagset for articles available though
    Wrapper function, for search_tag_set, returns a length value instead of its list, requirements are from the same function.
    Filter a list of POS-tags and collect associated values within the aggregate list
    :param aggregate: list, list of elements associated to the POS-tags given
    :param tags: list, list of POS-tags, associated to the aggregate list
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: int, the length of the returned aggregator list, by the search_tag_set
    '''
    tag_list = search_tag_set(aggregate=aggregate, tags=document_tags, accept_tags=accept_tags,
                              accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                              exclude_tags_start_with=exclude_tags_start_with)
    return len(tag_list)


def stat_sentence_length(lemma_by_sent):
    '''
    Ref: Pitler08 - Baseline measures

    :param lemma_by_sent:
    :return:
    '''
    sent_lengt = [len(i) for i in lemma_by_sent]
    return (mean_of_list(sent_lengt), max(sent_lengt), sum(sent_lengt))