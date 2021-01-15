
################# Helper functions
def check_tags(tag, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
  if tag in accept_tags or (tag[0] in accept_tags_start_with and subtag_matching(tag=tag, subtags=exclude_tags)):
        return True
  if tag not in exclude_tags or (tag[0] not in exclude_tags_start_with or (subtag_matching(tag=tag, subtags=accept_tags))):
      return True
  return False


def subtag_matching(tag, subtags):
  for st in subtags:
    if st in tag:
      return True
  return False


def search_tag_set(aggregate, tags, accept_tags=[], accept_tags_start_with=[], exclude_tags=[], exclude_tags_start_with=[]):
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
    return sum(l)/len(l)


def variance_of_list(l):
    m = mean_of_list(l)
    l_2 = [i**2 for i in l]
    return (sum(l_2) - len(l) * m**2) / (len(l)-1)


def split_into_sentences(aggregator_list, document_tags, accept_tags=["$."], accept_tags_start_with=[], exclude_tags=[],
                         exclude_tags_start_with=[]):
    lemma_list = []
    temp = []
    for t, a in zip(document_tags, aggregator_list):
        if check_tags(tag=a, accept_tags=accept_tags, accept_tags_start_with=accept_tags_start_with,
                      exclude_tags=exclude_tags, exclude_tags_start_with=exclude_tags_start_with):
            lemma_list.append(temp)
            temp = []
        else:
            temp.append(a)
    return lemma_list

def merge_tagsets(tagset_a, tagset_b):
    tagset_a.extend(tagset_b)
    return tagset_a