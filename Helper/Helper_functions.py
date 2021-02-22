import pandas as pd
import numpy as np

################# Helper functions

# TODO: write what functions do
def check_tags(tag, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
    '''
    Desision function to check wether a tag is within the searched for tag set.
    
    :param tag: str, a single POS-tag
    :param accept_tags: list, a set of POS-tag
    :param accept_tags_start_with: list, a set of POS-tag
    :param exclude_tags: list, a set of POS-tag
    :param exclude_tags_start_with: list, a set of POS-tag
    :return: boolean, whether the POS-tag is wanted within the group or not
    '''
    # print("Check Tagset")
    # print(accept_tags, accept_tags_start_with, exclude_tags, exclude_tags_start_with)
    # <editor-fold desc="Looking for accepted Tags, with given exceptions in exclude tags">
    if len(accept_tags_start_with) > 0 and type(accept_tags_start_with) == list:
        # print("accept start with", accept_tags_start_with)
        if tag[0] in accept_tags_start_with:
            for i in exclude_tags:
                if i in tag:
                    return False
            return True

        if tag in accept_tags:
            return True
        return False
    # </editor-fold>
    
    # <editor-fold desc="Looking for Tags to exclude, with some exceptions">
    elif len(exclude_tags_start_with) > 0 and type(exclude_tags_start_with) == list:
        # print("exclude start with", exclude_tags_start_with)

        if tag[0] in exclude_tags_start_with:
            for i in accept_tags:
                if i in tag:
                    return True
            return False
        if tag in exclude_tags:
            return False
        return True
    # </editor-fold>

    # <editor-fold desc="Check Tags in Accepted Tags">
    elif len(accept_tags) > 0 and type(accept_tags) == list:
        # print("accept", accept_tags)

        if tag in accept_tags:
            return True
        return False
    # </editor-fold>

    # <editor-fold desc="Check Tags in Excluded Tags">
    elif len(exclude_tags) > 0 and type(exclude_tags) == list:
        # print("exclude", accept_tags)
        if tag in exclude_tags:
            return False
        return True
    # </editor-fold>

    # <editor-fold desc="Catching Case">
    else:
        # print("No Tagsets given!", accept_tags, accept_tags_start_with, exclude_tags, exclude_tags_start_with)
        return True
    # </editor-fold>


def search_tag_set(aggregate, tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                   exclude_tags_start_with=[]):
    '''
    Filter a list of POS-tags and collect associated values within the aggregate list
    :param aggregate: list, list of elements associated to the POS-tags given
    :param tags: list, list of POS-tags, associated to the aggregate list
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: list, elements within the aggrergate list, filtered by the POS-tag sets
    '''
    aggregate_list = [a for a, t in zip(aggregate, tags)
                      if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                                    exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with)]
    return aggregate_list


def to_count_dict(aggregate_list):
    '''
    Generate a count dictionary from a list of elements
    :param aggregate_list: list, list of elements
    :return: dict, key=list elements, value=occurances in list
    '''
    count_dict = {}
    for i in aggregate_list:
        count_dict[i] = count_dict.get(i, 0) + 1
    return count_dict


def mean_of_list(l):
    '''
    calculate the mean value of a list
    :param l: list, list of digits
    :return:
    '''
    if len(l) > 0:
        try:
            return sum(l) / len(l)
        except:
            print("Could not calculate mean of: ", l)
            return 0
    return 0


def variance_of_list(l):
    '''
    Calculates the Variance of the elements of a list
    :param l: list, list of digits
    :return: variance of lists elements
    '''
    if len(l) > 0:
        m = mean_of_list(l)
        try:
            l_2 = [i ** 2 for i in l]
            return (sum(l_2) - len(l) * m ** 2) / (len(l) - 1)
        except:
            print("Could not calculate variance of: ", l)
            return np.Infinity
    return np.Infinity


def split_into_sentences(aggregator_list, document_tags, accept_tags=["$."], accept_tags_start_with=[], exclude_tags=[],
                         exclude_tags_start_with=[]):
    '''
    Split a list (agreegator_list) by sentence finishing POS-tags
    :param aggregator_list: list, list of elements e.g. lemmata, words, POS-tags
    :param document_tags: list, list of POS-tags
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: list of list, with the aggregator list, separated in sentences
    '''
    
    lemma_list = []
    temp = []
    for t, a in zip(document_tags, aggregator_list):
        if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            # print("tag:", t, "\n", accept_tags, accept_tags_start_with, exclude_tags, exclude_tags_start_with)
            lemma_list.append(temp)
            temp = []
        else:
            temp.append(a)
    lemma_list.append(temp)
    return lemma_list


def split_at_charset(text, sep=[".", ";", "!", "?", ":"]):
    '''
    Split text, by given characters (no POS-tags known yet)
    :param text: string
    :param sep: list, list of possible separators
    :return: list of str, separated by the given separators
    '''
    
    segmented = []
    while len(text) > 0:
        index = -1
        for index_i, i in enumerate(text):
            if i in sep:
                index = index_i
        if index == -1:
            segmented.append(text)
            break
        segmented.append(text[0:index])
        text = text[index + len(sep) - 1:]

    return segmented


def load_score_df(path_to_file, sep, identifier, column):
    '''
    Convert a Pandas Dataframe into a dictionary, with an index column for the key, and a single column as value.
    Only Last duplication is kept!
    :param df: pandas DataFrame
    :param identifier: str, column name of the identifier
    :param column: str, column name of the value column
    :return: dict
    '''
    df = pd.read_csv(path_to_file, sep=sep)
    df_word = df[~df.duplicated(subset=[identifier], keep="last")]
    df_word = df_word.set_index(identifier)
    if type(column) == list and len(column) > 1:
        df_word = df_word[column]
    else:
        df_word = df_word[[column]]
    word_dict = df_word.to_dict(orient="index")
    return word_dict



def POS_tagger(tagger, document):
    pos_tags = tagger.tag_text(document)
    # someelements are not taggeged!
    words = [i.split("\t")[0] for i in pos_tags if len(i.split("\t")) > 1]
    tags = [i.split("\t")[1] for i in pos_tags if len(i.split("\t")) > 1]
    lemmas = [i.split("\t")[-1] for i in pos_tags if len(i.split("\t")) > 1]
    return (words, tags, lemmas)


def load_word_freq(path, sep="\t", header=None, index_col=0, identifier="word", freq_column="freqency"):
    df = pd.read_csv(path, sep=sep, header=header, index_col=index_col, names=[identifier, freq_column], quoting=3)
    df = df.drop_duplicates(subset=[identifier])
    df = df.set_index(identifier)
    df= df[df[freq_column] > 1]
    list_dict = df.to_dict(orient="index")
    return list_dict
    # return df
    

def sort_by_POS_tags(aggregator_by_sent=[], tags_by_sent=[], accept=[], accept_star_with=[], exclude=[],
                     exclude_start_with=[], order_tagsets=[],
                     exclusive_accept=[], exclusive_accept_star_with=[], exclusive_exclude=[],
                     exclusive_exclude_start_with=[], exclusive_order_tagsets=[]):
    '''
    
    :param aggregator_by_sent:
    :param tags_by_sent:
    :param accept:
    :param accept_star_with:
    :param exclude:
    :param exclude_start_with:
    :param order_tagsets:
    :param exclusive_accept:
    :param exclusive_accept_star_with:
    :param exclusive_exclude:
    :param exclusive_exclude_start_with:
    :param exclusive_order_tagsets:
    :return:
    '''
    dict_by_sentence = {}
    full_doc_dict = {}
    
    # Iterate over all sentences, with aggregator list (lemma) and their POS tags
    for agg_sentence, tag_sentence in zip(aggregator_by_sent, tags_by_sent):
        sentence_dict = {}
        
        # Iterate over each element per sentence (lemma and tag)
        for a, t in zip(agg_sentence, tag_sentence):
            # generate tagsets without exclduing elements
            temp_dict = sentence_dict.get("all", {})
            temp_dict[a] = temp_dict.get(a, 0) + 1
            sentence_dict["all"] = temp_dict
    
            temp_dict = full_doc_dict.get("all", {})
            temp_dict[a] = temp_dict.get(a, 0) + 1
            full_doc_dict["all"] = temp_dict
            for at, atsw, et, etsw, tagset in zip(accept, accept_star_with, exclude, exclude_start_with, order_tagsets):
                if check_tags(tag=t, accept_tags=at, accept_tags_start_with=atsw, exclude_tags=et,
                              exclude_tags_start_with=etsw):
                    temp_dict = sentence_dict.get(tagset, {})
                    temp_dict[a] = temp_dict.get(a, 0) + 1
                    sentence_dict[tagset] = temp_dict

                    temp_dict = full_doc_dict.get(tagset, {})
                    temp_dict[a] = temp_dict.get(a, 0) + 1
                    full_doc_dict[tagset] = temp_dict
            for at, atsw, et, etsw, tagset in zip(exclusive_accept, exclusive_accept_star_with, exclusive_exclude, exclusive_exclude_start_with,
                                                  exclusive_order_tagsets):
                if check_tags(tag=t, accept_tags=at, accept_tags_start_with=atsw, exclude_tags=et,
                              exclude_tags_start_with=etsw):
                    temp_dict = sentence_dict.get(tagset, {})
                    temp_dict[a] = temp_dict.get(a, 0) + 1
                    sentence_dict[tagset] = temp_dict
    
                    temp_dict = full_doc_dict.get(tagset, {})
                    temp_dict[a] = temp_dict.get(a, 0) + 1
                    full_doc_dict[tagset] = temp_dict
                    break
        for tagset in ("all", *order_tagsets, *exclusive_order_tagsets):
            temp_list = dict_by_sentence.get(tagset, [])
            temp_list.append(sentence_dict.get(tagset, {}))
            dict_by_sentence[tagset] = temp_list
    return dict_by_sentence, full_doc_dict


def word_frequencies(lemma_by_sent):
    freq_by_sent = []
    doc_freq = {}
    for sent in lemma_by_sent:
        sent_dict = {}
        for lemma in sent:
            sent_dict[lemma] = sent_dict.get(lemma, 0) + 1
            doc_freq[lemma] = doc_freq.get(lemma, 0) + 1
        freq_by_sent.append(sent_dict)
    return freq_by_sent, doc_freq
    
    