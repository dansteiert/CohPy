from Helper.Helper_functions import mean_of_list


def concreteness_score(lemma, conc_dict):
    '''
    Check a list of conretness scores for the mean score of the lemma list
    :param lemma: list, of lemma
    :param conc_dict: dictionary of concreteness scores
    :return: mean concreteness score and the hitrate
    '''
    hitrate = 0
    conc = []
    for index, i in enumerate(lemma):
        temp = conc_dict.get(i, None)
        if temp is not None:
            conc.append(temp)
            hitrate += 1
    hitrate /= len(lemma)
    mean_concretness = mean_of_list(conc)
    return (mean_concretness, hitrate)

