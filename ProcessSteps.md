## Steps #Baseline

1. loading **training** and **test** csv data, loading the graph
2. computing graph features:
        - computing core_number (`nx.core_number(G)`)
        - computing centrality (`nx.eigenvector_centrality(G)`)
        - computing clustering coefficients (`nx.clustering(G)`)
        - computing pagerank (`nx.pagerank`)
        - computing DeepWalk (`dw=DpWalk(H)`) - after relabeling nodes using mapping
3. generating word embeddings from the abstracts as text [^1] and without stopwords (using **Bert**)
2. Most English stopwords are removed from the abstracts (see list of words in ``baseline_draft.py`` or ``textfeatures_randomRemovals.ipynb``)

[^1]: the abstracts, initially in *InvertedIndex* format, were processed and converted to pure *text* format (`processing_abstract.py`)
  

## Text Features preprocessing
The original text data comes in 2 forms: 
- a list of authors (``author_IDs``) and the IDs of their top-cited papers (at most 5, can be 0) ➡️ ``author_papers.txt``
- a list containing all papers (``paper_IDs``) with for each paper its abstract stored in InvertedIndex format ➡️ ``abstracts.txt`` <br />
In order to prepare the text features, the following preprocessing steps are completed:
1. the abstracts of all papers (``abstracts.txt``) are processed in order to convert them from InvertedIndex format to their actual text. This is done in/by ``processing_abstract.py`` and the results are saved in ``abstracts_data.pkl`` (paper_ID : paper_abstract_as_text...).
2. Process all authors (``author_papers.txt``) to associate each with the abstracts of his/her top-cited papers in plain text format, i.e. *author_ID: paper1's abstract, paper2's abstract...* <br /> This is done in/by:
        - ``process_authorFile.py`` which creates a dictionary with all authors and their paper_IDs (from ``author_papers.txt``).
        - ``final_dico_creation.py`` which takes as inputs the results of the first preprocessing step and the dictionary of authors and their paper_IDs and returns a dictionary in which the keys are the author_IDs and the values are concatenations of the processed abstracts of the authors. 
        - target stopwords are then removed from this dictionary (done in ``baseline_draft.py`` or ``textfeatures_randomRemovals.ipynb``) and the result is saved in ``DictForAuthor_new.pkl`` (dictionary format) and ``sentences.pkl`` (list format).

3. the list of authors with their concatenated abstracts (``DictForAuthor_new.pkl``, ``sentences.pkl``) is then passed through BERT to generate an embedding for each researcher. This is done in ``sentence_generator.py`` which parallelizes the process (in our case, 37 computers were used in parallel). The results are then put back together, which is done in ``master_pickle.py``.
4. Finally, the entire pipeline, loading the above features, creating the datasets, loading the model, training and testing, is visible in ``MLP_pipeline_with_feature_loading.py``.