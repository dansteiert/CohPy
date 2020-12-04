'''
KEywords:
Any word or n-gram that occurs at least twice in the source text and that occurs more frequently in the source text than
in the reference corpus (using normed frequency counts) is preliminarily selected as a keyword or n-gram

Idea: for each document, find those "keywords" meaning terms that occure more often in the document, than in the rest
of the dataset. And select the top "x" percent to represent the keywords of the document, comparing those keywords with
other documents to attain another measure of connectedness by keyword overlap, between documents.
This can be used for global cohesion.
'''