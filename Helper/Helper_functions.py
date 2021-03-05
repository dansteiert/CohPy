import pandas as pd
import numpy as np

################# Helper functions

# TODO: write what functions do
def check_tags(tag, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
    """
    Desision function to check whether a tag is within the searched for tag set.
    
    :param tag: str, a single POS-tag
    :param accept_tags: list, a set of POS-tag
    :param accept_tags_start_with: list, a set of POS-tag
    :param exclude_tags: list, a set of POS-tag
    :param exclude_tags_start_with: list, a set of POS-tag
    :return: boolean, whether the POS-tag is wanted within the group or not
    """
    # <editor-fold desc="Looking for accepted Tags, with given exceptions in exclude tags">
    if len(accept_tags_start_with) > 0 and type(accept_tags_start_with) == list:
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
        if tag in accept_tags:
            return True
        return False
    # </editor-fold>

    # <editor-fold desc="Check Tags in Excluded Tags">
    elif len(exclude_tags) > 0 and type(exclude_tags) == list:
        if tag in exclude_tags:
            return False
        return True
    # </editor-fold>

    # <editor-fold desc="Catching Case">
    else:
        print("No Tagsets given!")
        return True
    # </editor-fold>


def search_tag_set(aggregate, tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                   exclude_tags_start_with=[]):
    """
    Filter a list of POS-tags and collect associated values within the aggregate list
    :param aggregate: list, list of elements associated to the POS-tags given
    :param tags: list, list of POS-tags, associated to the aggregate list
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: list, elements within the aggrergate list, filtered by the POS-tag sets
    """
    aggregate_list = [a for a, t in zip(aggregate, tags)
                      if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                                    exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with)]
    return aggregate_list


def to_count_dict(aggregate_list):
    """
    Generate a count dictionary from a list of elements
    :param aggregate_list: list, list of elements
    :return: dict, key=list elements, value=occurances in list
    """
    count_dict = {}
    for i in aggregate_list:
        count_dict[i] = count_dict.get(i, 0) + 1
    return count_dict


def mean_of_list(l):
    """
    calculate the mean value of a list
    :param l: list, list of digits
    :return: mean of the list, or Infinity if not calculatable
    """
    if len(l) > 0:
        try:
            return sum(l) / len(l)
        except:
            print("Could not calculate mean of: ", l)
            return np.Infinity
    return np.Infinity


def variance_of_list(l):
    """
    Calculates the Variance of the elements of a list
    :param l: list, list of digits
    :return: variance of lists elements
    """
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
    """
    Split a list (agreegator_list) by sentence finishing POS-tags
    :param aggregator_list: list, list of elements e.g. lemmata, words, POS-tags
    :param document_tags: list, list of POS-tags
    :param accept_tags: list, a set of POS-tag, needed for check_tags function
    :param accept_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags: list, a set of POS-tag, needed for check_tags function
    :param exclude_tags_start_with: list, a set of POS-tag, needed for check_tags function
    :return: list of list, with the aggregator list, separated in sentences
    """
    
    lemma_list = []
    temp = []
    for t, a in zip(document_tags, aggregator_list):
        if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            lemma_list.append(temp)
            temp = []
        else:
            temp.append(a)
    lemma_list.append(temp)
    return lemma_list


def split_at_charset(text, sep=[".", ";", "!", "?", ":"]):
    """
    Split text, by given characters (no POS-tags known yet)
    :param text: string
    :param sep: list, list of possible separators
    :return: list of str, separated by the given separators
    """
    
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
    """
    Convert a Pandas Dataframe into a dictionary, with an index column for the key, and a single column as value.
    Only Last duplication is kept!
    :param path_to_file: str, path to the score file
    :param sep: the separator used in the score file
    :param identifier: str, column name of the identifier
    :param column: str, column name of the value column
    :return: dict{key=identifier,value=dict{key=column name given in column, value=value saved in the file}}
    """
    if path_to_file is None:
        return None
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
    """
    Assign Part of Spech Tags to the single elements given in the document.
    :param tagger: Instance of a POS Tagger - this is based on the TreeTagger!
    :param document: the raw text
    :return: list, words (unchanged), their POS tags and their lemma are given in their order of appearance, saved in separate files
    """
    pos_tags = tagger.tag_text(document)
    words = [i.split("\t")[0] for i in pos_tags if len(i.split("\t")) > 1]
    tags = [i.split("\t")[1] for i in pos_tags if len(i.split("\t")) > 1]
    lemmas = [i.split("\t")[-1] for i in pos_tags if len(i.split("\t")) > 1]
    return words, tags, lemmas


def load_word_freq(path, sep="\t", header=None, index_col=0, identifier="word", freq_column="freqency"):
    """
    Load the frequency word file, determining the frequency of each word in the background corpus
    :param path: str, path to the word frequency file
    :param sep: argument, passed to pandas read_csv function
    :param header: argument, passed to pandas read_csv function
    :param index_col: argument, passed to pandas read_csv function
    :param identifier: str, column name of the identifier
    :param freq_column: str, column name of the frequency column
    :return: dict{key=identifier,value=dict{key=freq_col, value=frequency value}}
    """
    if path is None:
        return None
    df = pd.read_csv(path, sep=sep, header=header, index_col=index_col, names=[identifier, freq_column], quoting=3)
    df = df.drop_duplicates(subset=[identifier])
    df = df.set_index(identifier)
    df= df[df[freq_column] > 1]
    list_dict = df.to_dict(orient="index")
    return list_dict
    

def sort_by_POS_tags(aggregator_by_sent=[], tags_by_sent=[], accept=[], accept_star_with=[], exclude=[],
                     exclude_start_with=[], order_tagsets=[],
                     exclusive_accept=[], exclusive_accept_star_with=[], exclusive_exclude=[],
                     exclusive_exclude_start_with=[], exclusive_order_tagsets=[]):
    """
    This function does the preprocessing for all POS tag related groupings.
    There are two groups of POS tags. Those which mutually exclude each other (exclusive) and those for which a one to
    many relation exists from tag to POS tagset. This is done to improve runtime a little.
    For each lemma and its associated aggregator, it is decided, whether it fits to a particular POS tag group
    and the aggregator elements themselves are counted, in a count dictionary fashion.
    Those Groupings are then returned for each single sentence separately and the whole document.
    :param aggregator_by_sent: list, each sentence is saved in its own sublist, the to be aggregated value, is usually the lemma
    :param tags_by_sent: list, each sentence is saved in its own sublist, POS tags corresponding to the aggregator_by_sent
    :param accept: list, [tagset_group_a[tags]]
    :param accept_star_with: list, [tagset_group_a[tags]]
    :param exclude: list, [tagset_group_a[tags]]
    :param exclude_start_with: list, [tagset_group_a[tags]]
    :param order_tagsets: list [tagset_group_names]
    :param exclusive_accept: list, [tagset_group_a[tags]]
    :param exclusive_accept_star_with: list, [tagset_group_a[tags]]
    :param exclusive_exclude: list, [tagset_group_a[tags]]
    :param exclusive_exclude_start_with: list, [tagset_group_a[tags]]
    :param exclusive_order_tagsets: [tagset_group_names]
    :return: two dict, by_sentence: {key=tagset_group_name, value=[dict{key=aggregator_element(lemma), value=int (how often it appeared}]}
    full_dict: {key=tagset_group_name, value=dict{key=aggregator_element(lemma), value=int (how often it appeared}}
    """
    dict_by_sentence = {}
    full_doc_dict = {}
    
    # <editor-fold desc="Iterate over all sentences, with aggregator list (lemma) and their POS tags">
    for agg_sentence, tag_sentence in zip(aggregator_by_sent, tags_by_sent):
        sentence_dict = {}
        
        # <editor-fold desc="Iterate over each element per sentence (lemma and tag)">
        for a, t in zip(agg_sentence, tag_sentence):
            # <editor-fold desc="Add elements to the 'all' category">
            temp_dict = sentence_dict.get("all", {})
            temp_dict[a] = temp_dict.get(a, 0) + 1
            sentence_dict["all"] = temp_dict
    
            temp_dict = full_doc_dict.get("all", {})
            temp_dict[a] = temp_dict.get(a, 0) + 1
            full_doc_dict["all"] = temp_dict
            # </editor-fold>
            
            # <editor-fold desc="Check whether the tag fits a Tagset (non exclusive)">
            for at, atsw, et, etsw, tagset in zip(accept, accept_star_with, exclude, exclude_start_with, order_tagsets):
                if check_tags(tag=t, accept_tags=at, accept_tags_start_with=atsw, exclude_tags=et,
                              exclude_tags_start_with=etsw):
                    temp_dict = sentence_dict.get(tagset, {})
                    temp_dict[a] = temp_dict.get(a, 0) + 1
                    sentence_dict[tagset] = temp_dict

                    temp_dict = full_doc_dict.get(tagset, {})
                    temp_dict[a] = temp_dict.get(a, 0) + 1
                    full_doc_dict[tagset] = temp_dict
            # </editor-fold>

            # <editor-fold desc="Check whether the tag fits a Tagset (exclusive) if tagset was found it is terminated early">
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
            # </editor-fold>
        # </editor-fold>

        # <editor-fold desc="Add the dictionaries for each sentence, to the list of dictionaries">
        for tagset in ("all", *order_tagsets, *exclusive_order_tagsets):
            temp_list = dict_by_sentence.get(tagset, [])
            temp_list.append(sentence_dict.get(tagset, {}))
            dict_by_sentence[tagset] = temp_list
        # </editor-fold>
    # </editor-fold>
    
    return dict_by_sentence, full_doc_dict


def word_frequencies(lemma_by_sent):
    """
    Count the frequency of each word, per sentence and within the whole document
    :param lemma_by_sent: list of list, list of sentences, where each sentence has a lemma given
    :return: [{key=lemma, value=cont}] in the sentence version or just {key=lemma, value=cont} for the full document.
    """
    freq_by_sent = []
    doc_freq = {}
    for sent in lemma_by_sent:
        sent_dict = {}
        for lemma in sent:
            sent_dict[lemma] = sent_dict.get(lemma, 0) + 1
            doc_freq[lemma] = doc_freq.get(lemma, 0) + 1
        freq_by_sent.append(sent_dict)
    return freq_by_sent, doc_freq
