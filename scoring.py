'''
Here the scoring should be done:
Meaning that each model is scored, by a given scoring method (cosine similarity)

For LSA and word2vec, similarity scores were determined using the cosine similarity between the vectors corresponding to
the compared segments (e.g., sentences, paragraphs, or documents), whichwere obtained from summing the vector weights for
each word in a particular segment. Average vector weights were calculated by considering the square root of the sum of
the squares of vector weights. The final cosine similarity score consisted of the sum of the products of the summed vector
weights from both segments, divided by the product of the average vector weights from both segments. LDA relatedness scores
were calculated as the inverse of the Jensen–Shannon divergence between the normalized summed vector weights for the
words in each segment


scoring is done in TAACO with adjacent Segments and with the next two segments (1,2), (2,3) or (1, 2+3), (2, 3+4)
'''