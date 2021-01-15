def POS_tagger(tagger, document):
  pos_tags = tagger.tag_text(document)
  words = [i.split("\t")[0] for i in pos_tags]
  tags = [i.split("\t")[1] for i in pos_tags]
  lemmas = [i.split("\t")[-1] for i in pos_tags]
  return (words, tags, lemmas)








