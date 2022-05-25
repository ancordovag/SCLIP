***SCLIP***

This project mix  SBERT and CIP to get better results of tfor CLIP for multiple languages.

**Steps**
0) To start, download the data:
    - Europarl en-es plain text from https://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-es.txt.zip
    - COCO 2017 Training and Validation Images and Annotations from https://cocodataset.org/#download
1) Prepare the data
    - Europarl: sentence_preprocessing.py separates the data located in the directory europarl in train, test and validation sets, whoses sizes are according to the configuration file config.yml. These sets can be found  
    - COCO: coco_preprocessing selects up to certain number of annotations to be processed. Then, pairs.py creates a dictionary with pairs annotations and images identifiers in a separate file. translate.ipynb translate the captions of all those anotations to all the adminisbles languages in the config file, creating new files with those pairs.
2) Train the NN
    2.0) The NN structure is to be loaded from networks.py. It is already being called by the other classes.
    2.1) Find which is the best NN structure playing with SCLIP.ipynb
    2.1) Train the best NN in best_sclip.ipynb. This saves the trained model in the folder models. This training takes parameters of the config file.
3) Compare
    experiment.ipynb 
    3.1) loads the best NN model already trained. 
    3.2) Then it goes file for file, for every language, and encodes the text with:
        - SBERT + the best model (NN)
        - CLIP
         At the same time encodes the images. Then compares the vector of each image with all the possible captions in that language, giving a probability score. With this, it calculates the Mean Reciprocal Rank.
    3.3) It outputs the results for each language, for SBERT + NN and for CLIP. In a table and ina barplot.
    
**Results**
Preliminary results show that CLIP is much better at encoding text in english but for most of other languages SBERT performs better. 
