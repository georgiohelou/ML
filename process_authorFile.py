'''
process the author_papers.txt file into a dictionary
'''
import os
import sys

def process_file_to_dico():
	dico_authors_to_paperid = {}
	with open(os.path.join(sys.path[0], "author_papers.txt"), "r") as file1:
		while True:
		    # Get next line from file
		    line = file1.readline()
		    if not line:
		        break
		    author_id, all_paper_ids = line.split(":")
		    papers_ids = all_paper_ids.split("-")
		    dico_authors_to_paperid[author_id] = papers_ids
		 
		file1.close()
	return dico_authors_to_paperid

if __name__ == "__main__" :
	dic = process_file_to_dico()
	print(dic)