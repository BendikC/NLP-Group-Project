# NLP-Group-Project

Setting up the project:

1. Download the data files and put them into the data/ directory, can be downloaded from here: https://drive.google.com/drive/folders/1qIZcNcU2wtiJNr3BUyX2GIUtnHEfbQDi?usp=sharing

2. Structure the files such that 'wiki_musique_corpus.json' is in the 'data' directory, whereas the 'dev.json', 'train.json' and 'test.json' are in a subdirectory of data called 'qa', so you should make a new folder in 'data/qa' to put them in. The filepaths are thus 'data/wiki_musique_corpus.json', 'data/qa/dev.json', 'data/qa/train.json' and 'data/qa/test.json'

3. Make a conda-env with conda, using ``` conda env create -f environment.yml ``` to install the required dependencies (make sure to add/remove dependencies while we work, as we most likely will need to use more dependencies). Note: you might have to delete and create the environment again. If dexter-cqa is a dependency you might need to install: https://visualstudio.microsoft.com/visual-cpp-build-tools/ . During installation, make sure to select "Desktop development with C++". 

4. Get a huggingface token (from https://huggingface.co/) and an openai token (from their developer platform https://platform.openai.com/docs/overview) and put them in a .env file in the root directory of the project, as such: 
- huggingface_token=[insert huggingace token] 
- OPENAI_KEY=[key here]

5. Run `python corpus_management/encode_corpus.py` to encode the corpus into dense embeddings. This process will take a while (potentially several hours depending on your hardware) and will save the embedded corpus as a memmap file in data/embeddings. This step is necessary for efficient retrieval later.

6. Run experiment and hope for the best.

