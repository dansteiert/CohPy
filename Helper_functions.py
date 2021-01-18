################# Helper functions
def check_tags(tag, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
    ## desicion between keeping and throwing out -> check for start with property:
    ## if not met for either:
    ### check if accept > 0:
    #### if not met check for exclude > 0
    ##### if not found -> return True

    ## Looking for accepted Tags, with some exceptions in exclude tags
    if len(accept_tags_start_with) > 0 and type(accept_tags_start_with) == list:
        if tag[0] in accept_tags_start_with:
            for i in exclude_tags:
                if i in tag:
                    return False
            return True

        if tag in accept_tags:
            return True
        return False
    ## Looking for Tags to exclude, with some exceptions
    elif len(exclude_tags_start_with) > 0 and type(exclude_tags_start_with) == list:
        exclude_flipper = False
        if tag[0] in exclude_tags_start_with:
            exclude_flipper = True
            for i in accept_tags:
                if i in tag:
                    return True
            return False
        if tag in exclude_tags:
            return False
        return True

    elif len(accept_tags) > 0 and type(accept_tags) == list:
        if tag in accept_tags:
            return True
        return False
    elif len(exclude_tags) > 0 and type(exclude_tags) == list:
        if tag in exclude_tags:
            return False
        return True
    else:
        print("else decision case")
        return False

        if tag[0] in accept_tags_start_with:
            # print("True")
            return True

    # print("tag:", tag, "\n", accept_tags, accept_tags_start_with, exclude_tags, exclude_tags_start_with)
    exclude_flipper = False
    ## Exclude
    if len(exclude_tags) > 1:
        for i in exclude_tags:
            if i in tag:
                # print("FALSE")
                return False
    elif tag in exclude_tags:
        # print("FALSE")
        return False
    if len(exclude_tags_start_with) > 1:
        for i in exclude_tags:
            if i == tag[0]:
                exclude_flipper = True
                break
    elif tag[0] == exclude_tags_start_with:
        exclude_flipper = True
    ## Accept
    if len(accept_tags) > 1:
        for i in accept_tags:
            if i in tag:
                # print("True")
                return True

    elif tag in accept_tags:
        # print("True")
        return True

    ## exclude rest
    if exclude_flipper:
        # print("Flipper False")
        return False
    if len(accept_tags) == 0 or len(accept_tags_start_with) == 0:
        return True
    # print("nothing matched False")
    return False


def subtag_matching(tag, subtags):
    for st in subtags:
        if st in tag:
            return False
    return True


def search_tag_set(aggregate, tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[],
                   exclude_tags_start_with=[]):
    aggregate_list = [a for a, t in zip(aggregate, tags)
                      if check_tags(tag=t, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                                    exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with)]
    return aggregate_list


def to_count_dict(aggregate_list):
    count_dict = {}
    for i in aggregate_list:
        count_dict[i] = count_dict.get(i, 0) + 1
    return count_dict


def mean_of_list(l):
    return sum(l) / len(l)


def variance_of_list(l):
    m = mean_of_list(l)
    l_2 = [i ** 2 for i in l]
    return (sum(l_2) - len(l) * m ** 2) / (len(l) - 1)


def split_into_sentences(aggregator_list, document_tags, accept_tags=["$."], accept_tags_start_with=[], exclude_tags=[],
                         exclude_tags_start_with=[]):
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
    return lemma_list


def merge_tagsets(tagset_list):
    new_tagset = []
    for i in tagset_list:
        new_tagset.extend(i)
    new_tagset = list(set(new_tagset))
    return new_tagset


def split_at_newline(text, sep="\n"):
    ## double new_line:
    segmented = []
    while len(text) > 0:
        index = text.find(sep)
        if index == -1:
            segmented.append(text)
            break
        segmented.append(text[0:index])
        text = text[index + len(sep) - 1:]

    return segmented

def list_to_dict(df, identifier, column):
    # might need to check for string values
    list_dict = {}
    for w, c in zip(df[identifier].tolist(), df[column].tolist()):
        list_dict[w] = c
    return list_dict