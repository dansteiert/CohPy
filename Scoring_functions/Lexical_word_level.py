from Helper.Helper_functions import mean_of_list


def affinity_conc_score(lemma_by_sent, affinity_conc_dict, affinity_conc_label, size_of_document):
    '''
    Check a list of conretness scores for the mean score of the lemma list
    :param lemma: list, of lemma
    :param conc_dict: dictionary of concreteness scores
    :return: mean concreteness score and the hitrate
    '''
    hitrate = 0
    affinities_by_sent = {}
    
    # Iterate over all Sentences
    for sentence in lemma_by_sent:
        temp_affinities = {} # dict of affinity_conc_label as keys and list of affinity values as dict values
        
        # Iterate over all lemma within the sample and search in reference dictionary for values
        for lemma in sentence:
            temp_dict = affinity_conc_dict.get(lemma, None)
            if temp_dict is not None:
                
                # get for each affinity value for the lemma
                hitrate += 1
                for aff_lab in affinity_conc_label:
                    # add to affinitie
                    temp_aff = temp_affinities.get(aff_lab, [])
                    temp_aff.append(temp_dict.get(aff_lab, 0))
                    temp_affinities[aff_lab] = temp_aff
        
        # for each affinity_conc_label, create a list of list of affinity values
        for aff_lab in affinity_conc_label:
            temp_aff = affinities_by_sent.get(aff_lab, [])
            temp_aff.append(temp_affinities.get(aff_lab, []))
            affinities_by_sent[aff_lab] = temp_aff
    hitrate /= size_of_document
    return (affinities_by_sent, hitrate)

def mean_concreteness(concreteness_label, affinity_conc_dict):
    '''
    Ref: Grasser 2004
    :param concreteness_label:
    :param affinity_conc_dict:
    :return:
    '''
    concreteness_list = affinity_conc_dict.get(concreteness_label, [])
    concreteness_score = mean_of_list([mean_of_list(i) for i in concreteness_list])
    return concreteness_score