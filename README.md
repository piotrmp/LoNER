## Where Does It End? Long Named Entity Recognition for Propaganda Detection


This repository contains the code for our solution for identifying propaganda techniques in text. 

The source code was prepared within a study [Where Does It End? Long Named Entity Recognition for Propaganda Detection and Beyond](TODO), presented at the [Workshop on NLP applied to Misinformation] (NLP-Misinfo-2023), organised at the [SEPLN 2023](http://sepln2023.sepln.org) Conference, held in Jaén, Spain on the 26th of September, 2023. Please consult [the paper](TODO) for details on the work.

The research was done within the [HOMADOS](https://homados.ipipan.waw.pl/) project at [Institute of Computer Science](https://ipipan.waw.pl/), Polish Academy of Sciences.

## Overview

Propaganda detection is usually defined and solved as a Named Entity Recognition (NER) task. However, the instances of propaganda techniques (text spans) are usually much longer than typical NER entities (e.g. person names) and can include dozens of words. In this work, we investigate how the extensive span lengths affect the recognition of propaganda, showing that the task difficulty indeed increases with the span length. We systematically evaluate several common approaches to the task, measuring how well they recover the length distribution of true spans. We also propose a new solution, specifically aimed to perform NER for such long entities.


## Quick Installation

- `python3`
- `fastprogress==1.0.2`
- `keras==2.6.0`
- `scikit-learn==1.0.1`
- `spacy==3.0.1`
- `scipy==1.8.0`
- `tensorflow==2.6.0`
- `tensorflow-addons==0.15.0`
- `tensorflow-hub==0.12.0`
- `tensorflow-text==2.6.0`

Run the following script to install the dependencies,
```
pip3 install -r requirements.txt
```

## Data 
Data processing scripts are invoked inside model running scripts. Please unzip PICO_dataset.zip for _ebmnlp_ scripts or PTC_dataset.zip for other scripts.

## Download Models

Download and unzip Bert base uncased model from [here](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)
. You can also try different sizes of Bert available on Tensorflow Hub.

## How to Run?

Run either of these scripts inside your favourite Python IDE. Remember to replace file paths with your own:

-bert_raw_tags.py (Bert coupled with linear classification layer on top)

-bert_bio_tags.py (The same as above, labels are encoded according to Begin-Inside-Outside scheme)

-bert_iobes_tags.py (The same as above, labels are encoded according to Inside-Outside-Begin-End-Single scheme)

-bert_crf_raw_tags.py (Bert coupled with CRF layer on top)

-bert_crf_bio_tags.py (The same as above, labels are encoded according to Begin-Inside-Outside scheme)

-bert_crf_iobes_tags.py (The same as above, labels are encoded according to Inside-Outside-Begin-End-Single scheme)

-bert_LoNER_systematic_45.py (Bert with adaptive convolutional layer)

## Licence

LoNER code is released under the [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) licence.



