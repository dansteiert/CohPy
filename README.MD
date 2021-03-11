# CohPy
CohPy is an easy-to-use pure python pipeline, to measure textual features and with them assess 
textual comprehensibility. Currently CohPy is usable for two languages, english and german.
It supplies a crawl feature for the gutenberg project, and two categories of additional texts.
The great advantage is its open source code and thereby enabling additional implementations/modifications
of scores and languages.

## Target group
The aim of this project, is to supply everybody, working with texts and text comprehension/cohesion,
to have a tool at their disposal.

## Get Started:
- Some of the scores require additional dependencies. If those dependencies are not available, 
the remaining workflow is still operational!
    - V2W model of sentiment scores: we received the SentiArt model (Jacobs, A. M., Kinder, A. (2019). Computing the Affective-Aesthetic Potential of Literary Texts)
    - Affective Norm scores: Available at the University of Stuttgarts Institute for Natural Language Processing
    (https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/twitter-norms/)
    - Word Frequencies: Available at the University of Leipzig Leipzig Corpora Collection (https://wortschatz.uni-leipzig.de)
    - List of Connective words, by categories - A preliminary self compiled list is available at data/Score_files
- The installation of the TreeTagger is also necessary (https://cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
    - Alternatively, a different POS Tagger and Lemmatizer can be used, by changing the implementation slightly.
    The relevant function can be found at Helper/Helper_function.py, called POS_tagger.
- Clone github repository
- install all necessary dependencies (conda env create -f environment.yml)
- running the application:
    - It is recommended to keep the folder structure suggested below.
        - Otherwise quite some paths have to be adapted
    - make sure that your current working directory is at the same level as the main.py script
    - As mentioned earlier, this tool provides a download possibility for the gutenberg project,
    although this is not possible from a german Network address! (at least in March 2021)
        -   Note that this process requires 10h + 
    - with bool variables, it can be checked, which folders should be processed.
    - If some of the dependencies are not given, set them to None (for each language where it is applicable)
- for specifics, each function is labeled as what its input is and what it does    

    
    
## Folder Structure for the data folder:
- save additional dependency files in the Score_files folder
    - Affective_Norm_*LANGUAGE_IN_2_LETTER_CODE*.csv
    - Word_Frequency_*LANGUAGE_IN_2_LETTER_CODE*.csv
    - Connectives_*LANGUAGE_IN_2_LETTER_CODE*.csv
    - Sentiment_v2w.vec - using SentiArt, contains english and german words
- If no changes are made, the Gutenberg data is saved to a folder Gutenberg, containing the 
metadata file and a subfolder txt_files, where the Gutenberg Project Books are saved into, with their
ID as identifier.
- Extra_books and New_documents folders contain a subfolder for each language (also with *LANGUAGE_IN_2_LETTER_CODE*)
- ML Results is generated to save all results of the Classification/Regression Task
- The Result files of the scoring pipeline are saved at the target_paths for the categories 
(can be set at the main function) default location is data/


## Adding new Scores:
 - Implement the score
 - import into the Pipeline.py script
 - place at an appropriate position/end
 - add the results to the result_dict
 
## Adding new Languages:
- Add new Tagsets: see Tagset_*LANGUAGE_IN_2_LETTER_CODE*.py
- Import Tagsets in the Pipeline (at "Load Tagsets from Tagset_LANGUAGE.py" )
- Add the dependency for the treeTagger
- call the main function, with the respective additional language specific dependencies
