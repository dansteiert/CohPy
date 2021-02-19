from Helper.Helper_functions import mean_of_list


def affective_conc_score(lemma_dict_by_sent, df_affective, affective_conc_label, size_of_document):
    '''
    Check a list of conretness scores for the mean score of the lemma list
    :param lemma: list, of lemma
    :param conc_dict: dictionary of concreteness scores
    :return: set(dict{affective label: list[sentence list[affective values per lemma]]}, int, hitrate
    '''
    hitrate = 0
    affinities_by_sent = {}
    
    # Iterate over all Sentences
    for dictionary in lemma_dict_by_sent:
        temp_affinities = {} # dict of affective_conc_label as keys and list of affective values as dict values
        
        # Iterate over all lemma within the sample and search in reference dictionary for values
        for lemma, count in dictionary.items():
            # try:
            #     temp_row = df_affective.query(expr="index == '%s'" % lemma)
            # except:
            #     continue
            temp_dict = df_affective.get(lemma, None)
            if temp_dict is not None:
            # if temp_row.shape[0] > 0:
                # get for each affective value for the lemma
                hitrate += 1
                for aff_lab in affective_conc_label:
                    # add to affinitie
                    try:
                        temp_aff = temp_affinities.get(aff_lab, [])
                        # temp_aff.append(temp_row.loc[lemma, aff_lab])
                        temp_aff.extend([temp_dict.get(aff_lab, 0) for _ in range(0, count)])
                        temp_affinities[aff_lab] = temp_aff
                    except:
                        print(aff_lab, lemma)
        
        # for each affective_conc_label, create a list of list of affective values
        for aff_lab in affective_conc_label:
            temp_aff = affinities_by_sent.get(aff_lab, [])
            temp_aff.append(temp_affinities.get(aff_lab, []))
            affinities_by_sent[aff_lab] = temp_aff
    hitrate /= size_of_document
    return (affinities_by_sent, hitrate)

def mean_concreteness(concreteness_label, affective_conc_dict):
    '''
    Ref: Grasser 2004
    :param concreteness_label:
    :param affective_conc_dict:
    :return:
    '''
    concreteness_list = affective_conc_dict.get(concreteness_label, [])
    concreteness_score = mean_of_list([mean_of_list(i) for i in concreteness_list])
    return concreteness_score