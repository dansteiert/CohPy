def POS_tagger(tagger, document):
    pos_tags = tagger.tag_text(document)
    # someelements are not taggeged!
    words = [i.split("\t")[0] for i in pos_tags if len(i.split("\t")) > 1]
    tags = [i.split("\t")[1] for i in pos_tags if len(i.split("\t")) > 1]
    lemmas = [i.split("\t")[-1] for i in pos_tags if len(i.split("\t")) > 1]


    return (words, tags, lemmas)
