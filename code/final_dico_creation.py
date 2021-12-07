'''
create our final dictionary that takes Authors_id as keys
and for each has a single value associated : the concatenation of
all abstracts strings (in case this author had several papers) 
after they have been preprocessed to remove stopwords
'''
#used when we truncate the extra words (input for BERT)
def dictionary_concatenation(dico_paperid_to_abs, dico_authorid_to_paperid):
	dico_author_to_concatAbs = {}
	for author in dico_authorid_to_paperid:
		concatenated_abs = ''
		for paperid in dico_authorid_to_paperid[author]:
			if (int(paperid) in dico_paperid_to_abs.keys()):
				concatenated_abs+= dico_paperid_to_abs[int(paperid)]
		dico_author_to_concatAbs[author] = concatenated_abs
	return dico_author_to_concatAbs

#used when we remove random words to match BERT input size
def dictionary_concatenation2(dico_paperid_to_abs, dico_authorid_to_paperid):
	dico_author_to_concatAbs = {}
	for author in dico_authorid_to_paperid:
		concatenated_abs = []
		for paperid in dico_authorid_to_paperid[author]:
			if (int(paperid) in dico_paperid_to_abs.keys()):
				concatenated_abs.append(dico_paperid_to_abs[int(paperid)])
		dico_author_to_concatAbs[author] = concatenated_abs
	return dico_author_to_concatAbs
