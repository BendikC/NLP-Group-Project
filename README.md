# NLP-Group-Project
This project is done as part of the DSAIT4090 Natural Language Processing course at TU Delft. It investigates how negatives and har negatives affect RAG performance. In both their inclusion in IR training and the input contexts for the LLM.

## Setting up the project:

1. Download the data files and put them into the data/ directory, can be downloaded from here: https://drive.google.com/drive/folders/1qIZcNcU2wtiJNr3BUyX2GIUtnHEfbQDi?usp=sharing

2. Structure the files such that 'wiki_musique_corpus.json' is in the 'data' directory, whereas the 'dev.json', 'train.json' and 'test.json' are in a subdirectory of data called 'qa', so you should make a new folder in 'data/qa' to put them in. The filepaths are thus 'data/wiki_musique_corpus.json', 'data/qa/dev.json', 'data/qa/train.json' and 'data/qa/test.json'

3. Make a conda-env with conda, using ``` conda env create -f environment.yml ``` to install the required dependencies (make sure to add/remove dependencies while we work, as we most likely will need to use more dependencies). Note: you might have to delete and create the environment again. If dexter-cqa is a dependency you might need to install: https://visualstudio.microsoft.com/visual-cpp-build-tools/ . During installation, make sure to select "Desktop development with C++". 

4. Get a huggingface token (from https://huggingface.co/) and an openai token (from their developer platform https://platform.openai.com/docs/overview) and put them in a .env file in the root directory of the project, as such: 
- huggingface_token=[insert huggingace token] 
- OPENAI_KEY=[key here]

5. Run `python corpus_management/encode_corpus.py` to encode the corpus into dense embeddings. This process will take a while (potentially several hours depending on your hardware) and will save the embedded corpus as a memmap file in data/embeddings. This step is necessary for efficient retrieval later.

## Training ADORE-Contreiver 
This can take a while and you might need to configure the training parameters in `train_adore.py` based on your system's capabilities. You can first check with ``` python setup_analysis/check_gpu_availability.py ``` . Then once the parameters are configured apprpriately you can run ```python train_adore.py``` which will generate model weights for each of 6 epochs in `model_checkpoint/`.

## Run Experiment to collect results
Then in `relevant_contexts_experiment.py` choose the experiment you want and run the file with ``` python relevant_contexts_experiment.py```. This will generate results in `results/` when complete.


