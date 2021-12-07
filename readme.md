# INF554 2021 Data Challenge: H-index Prediction
### Ecole Polytechnique - INF 554 Machine and Deep Learning - Data Challenge: H-index Prediction  
**Georgio El Helou, Rebecca Jaubert, Nathan Pollet.** 

#### Dependencies
Add the following dependencies inside of your environment (if not already installed): ``Pandas``, ``numpy``, ``networkx``, ``sklearn``, ``pickle``, ``nltk``, ``torch``, ``gensim``, ``transformers`` (recent versions recommended). 
We have used conda to install packages when available but please note that some packages such as ``transformers`` can only be installed via ``pip``.
In case you receive an ntlk warning please run the following command from your terminal: ``python -m nltk.downloader punkt``. The last parameter, “punkt”, is to be changed to any undownloaded file mentioned in the warning message.
#### Project structure and file details
Note: ``ProcessSteps.md`` describes the main steps of our analysis.
* The **``code_all_in_one``** folder contains the files necessary to repeat our experiments simply by placing the original data files in it and running ``ML_allInOne.py``. However, due to the nature of the data processing tasks, this would take an enormous amount of time to complete and would require excessive (unrealistic) memory requirements if not properly parallelized. <span style="color:blue">*We advice not to attempt this method and to use the feature loading approach instead*</span> (details below).
* The **``code``** folder contains all the scripts developed for the analysis. All tasks are covered: data loading, data processing, feature generation, feature saving and loading, model initialization, model training, model testing and graphing of the results.
To repeat the experiments, two options are available: **Option 1**: process the data, generate the required features and run the experiments, **Option 2**: load the required processed data and features and run the experiments. 
**Note:** for both options, a **``data``** directory and a **``features``** directory should exist alongside the **``code``** directory, as initially provided.
    - **Option 1**
        1. run ``MLP_pipeline1_with_feature_generation.py``. This generates several pickle data files (see details in the script) and results in the generation of ``sentences_cleaned_random_max450words.pkl``. This file is to be used to obtain the BERT text embeddings.
        2. Send ``sentences_cleaned_random_max450words.pkl`` to your school computer system (or similar architecture allowing for parallelization) along with ``sentence_generator.py`` and ``master_pickle.py``.
        3. Run ``sentence_generator.py`` on separate sessions (e.g. on different machines via ``ssh``), incrementally increasing the ``bucket ID`` until it reaches 37. This code divides the BERT Embedding task into 37 subtasks which can be ran independently.
        4. Once all 37 "bucket embeddings" generated, run ``master_pickle.py`` to reassemble them into a single pickle file: ``fullEmbeddings_random.pkl``.
        5. Place this file in the **``features``** folder.
        6. Run ``MLP_pipeline2_with_feature_generation.py``.
    - **Option 2** <span style="color:blue">(*recommended*)</span>.
        1. Go [here](https://drive.google.com/drive/folders/1CDvV8U3VG0SWPUb1hqMhVmUhCs3TEucH) and download the **``data``** and **``features``** folders
        2. Place these folders alongside ``code`` (overwrite if necessary)
        3. Run ``MLP_pipeline_with_feature_loading.py``

Do not hesitate to email us if there are any difficulties :). 

nathan.pollet@polytechnique.edu, georgio.el-helou@polytechnique.edu, rebecca.jaubert@polytechnique.edu
