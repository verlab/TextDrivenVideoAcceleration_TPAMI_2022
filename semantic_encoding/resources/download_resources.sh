#!/bin/bash

# Download VaTeX JSON data
wget https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json
wget https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json

# Download the Pretrained GloVe Embeddings
wget -O glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip -j glove.6B.zip glove.6B.300d.txt
rm glove.6B.zip