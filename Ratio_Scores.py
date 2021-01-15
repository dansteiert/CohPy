from Helper_functions import *


def pronoun_resolution(document_tags, nouns_accept_tags=[], nouns_accept_tags_start_with=["N"], nouns_exclude_tags=[],
                       nouns_exclude_tags_start_with=[],
                       pronouns_accept_tags=["ADJA", "ADJD", "ADV"], pronouns_accept_start_with=["P"],
                       pronouns_exclude_tags=["PTK"],
                       pronouns_exclude_tags_start_with=[]):
    # This Task, wants to map pronouns with their respectiv "owners"
    # more complex task!
    # Pronoun density  consists  of  the  proportion  of  noun  phrases(NPs, as defined  by  a  syntactic
    # parser, which  will  be  described  later) that  are  captured by  pronouns(as defined  by  the  Brill  POS  tagger).

    tag_list_nouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=noun_tags,
                                    accept_tags_start_with=noun_tags_start_with, exclude_tags=exclude_noun_tags,
                                    exclude_tags_start_with=exclude_noun_tags_start_with)
    tag_list_pronouns = search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=pronoun_tags,
                                       accept_tags_start_with=pronoun_tags_start_with,
                                       exclude_tags=exclude_pronoun_tags,
                                       exclude_tags_start_with=exclude_pronoun_tags_start_with)
    if len(tag_list_pronouns) > 0:
        return len(tag_list_nouns) / len(tag_list_pronouns)
    return len(tag_list_nouns)


def content_functional_ratio(document_tags, content_tags=[], content_tags_start_with=["N"], exclude_content_tags=[],
                             exclude_content_tags_start_with =[], functional_tags=["READUP!"],
                             functional_tags_start_with=[], exclude_functional_tags=[],
                             exclude_functional_tags_start_with =[]):
  tag_list_content= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=content_tags,
                                    accept_tags_start_with=content_tags_start_with, exclude_tags=exclude_content_tags,
                                    exclude_tags_start_with=exclude_content_tags_start_with)
  tag_list_functional= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=functional_tags,
                                    accept_tags_start_with=functional_tags_start_with, exclude_tags=exclude_functional_tags,
                                    exclude_tags_start_with=exclude_functional_tags_start_with)

  if len(tag_list_functional) > 0:
    return len(tag_list_content)/len(tag_list_functional)
  return len(tag_list_content)


def type_token_ratio(document_tags, accept_tags=[], accept_tags_start_with=["N"], exclude_tags=[],
                             exclude_tags_start_with =[]):
  # count unique words against their repetitions.
  # split into Nouns and non-Noun content words
  tag_list_nouns= search_tag_set(aggregate=document_tags, tags=document_tags, accept_tags=accept_tags,
                                    accept_tags_start_with=accept_tags_start_with, exclude_tags=exclude_tags,
                                    exclude_tags_start_with=exclude_tags_start_with)
  count_dict = to_count_dict(aggregate_list=tag_list_nouns)
  if len(count_dict) == 0:
    ratio = 0
  else:
    ratio = len(count_dict)/sum(count_dict.values())

  return ratio