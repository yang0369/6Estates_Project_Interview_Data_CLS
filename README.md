# About The Project

This project is about an interview assignment with a binary sentiment analysis problem.

To address the stated problem, I have tried with a few traditional NLP models.

The models that I have tried are:

- [ ] BERT model
- [ ] TF-IDF + logistic regression
- [ ] TF-IDF + random forest

# Project Structure
```
📦6Estates_Project_Interview_Data_CLS
 ┣ 📂.polyaxon
 ┃ ┣ 📜.polyaxongroup
 ┃ ┣ 📜.polyaxonproject
 ┃ ┗ 📜.polyaxonxp
 ┣ 📂data
 ┃ ┣ 📜dev.json
 ┃ ┣ 📜glove.840B.300d.txt
 ┃ ┣ 📜test.json
 ┃ ┗ 📜train.json
 ┣ 📂model
 ┃ ┣ 📜config.json
 ┃ ┣ 📜history.csv
 ┃ ┣ 📜predicted.csv
 ┃ ┣ 📜tf_model.h5
 ┃ ┗ 📜info.log
 ┃ ┗ 📜training_logs
 ┣ 📂polyaxon
 ┃ ┣ 📂docker
 ┃ ┃ ┣ 📜experiment.df
 ┃ ┃ ┗ 📜notebook.df
 ┃ ┣ 📜experiment.yml
 ┃ ┗ 📜notebook.yml
 ┣ 📂scripts
 ┃ ┗ 📜link_workspace.sh
 ┣ 📂src
 ┃ ┣ 📜config_6estate.yml
 ┃ ┣ 📜datapipeline_6estate.py
 ┃ ┣ 📜experiment_6estate.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜.polyaxonignore
 ┣ 📜analysis.ipynb
 ┣ 📜conda.yml
 ┣ 📜README.md
 ┣ 📜skaffold.yaml
 ┗ 📜__init__.py
```

# Details About The Directories

 ┣ 📂.polyaxon

> contains files related to Polyaxon experiment 

┣ 📂data

> Data folder, contains all the train, dev, test datasets, and one word embedding file (which I did not use in this project). data folder was not uploaded to Git due to size limit

 ┣ 📂model

> All files generated during model training, including:  
>
> 				- 📜config.json: contains all the trained weights/parameters
>
>    - 📜history.csv: contains the loss/accuracy during training and validation, used for plotting graph
>    - 📜predicted.csv: contains all the predicted labels for test dataset
>    - 📜tf_model.h5: the trained model in .h5 format (not uploaded due to size limit)
>    - 📜info.log: records all the hyperparameters/parameters during training process
>    - 📜training_logs: records all the training epochs, train/val accuracy and loss etc.

 ┣ 📂polyaxon

> Dockerfiles for setting up the running environment for polyaxon experiment

┣ 📂scripts

> A script to setup persistent data storage on polyaxon

 ┣ 📂src

> The source code folder, which contains the main python scripts for this project
> - 📜config_6estate.yml: passing path and hyperparameters to the script, which makes the hyperparameter tuning easier 
> - 📜datapipeline_6estate.py: the one-stop pipeline to run data cleaning, data preprocessing and model training, evaluation. Please refer to "User Guide" section on how to run this pipeline.
> - 📜experiment_6estate.py: this script is created for training BERT model on Polyaxon platform, the main reason why I trained model on Polyaxon compared to local machine is the super fast training speed. It takes only about 15 minutes by the GPU on Polyaxon!

📜analysis.ipynb 

> The main analysis notebook, where a simple EDA was performed. Based on this notebook, we can know what kind of dataset out there, the basic statistics about the dataset, and also what data cleaning process should be applied. In addition, I have performed the Model result analysis on this notebook as well.

 📜conda.yml

> Contains all the required libraries for running this project

# User guide 

For the purpose of simplicity, this project has one and only one python script to run the data processsing pipeline and modelling pipeline.

Just simply go to your command prompt(windows) or terminal(linux), change the working directory to the project parent folder. e.g. in my case, I saved the project martials in the directory of "D:\personal_git\6Estates_Project_Interview_CLS\6Estates_Project_Interview_Data_CLS"

```bash
 cd D:\personal_git\6Estates_Project_Interview_Data_CLS\
```

Then type the python command shown below to run the pipeline:

```python
python -m src.datapipeline_6estate --config_path src/config_6estate.yml
```

That's it, the pipeline will run itself locally and generate all the modelling information, parameters, artefacts and trained model automatically, and all of them will be stored at  ┣ 📂model folder.

Please take note that my model was trained on Polyaxon platform, which is why I can train the model with a very huge memory space and super fast compute speed. However, if you need to run it locally, you may need to adjust the batch size, Number of epochs etc to avoid the out of memory(OOM) issue.

# About analysis.ipynb 

The purpose of this notebook is to explore with the raw dataset, and to think of the idea or method on data processing, model building based on the findings. I will explain the works part by part below:

First, some basic statistics and useful findings from "EDA" part: 

- train size：8636
- test size: 1066
- val size: 960
- Each dataset has two columns: "sentence" is the text data for classification; "label" is the response variable with 1 and 0 only.
- Seems label 1 represents positive comment, while label 0 represents negative comment.
- The machine learning model shall have 1 label only
- Basically the dataset is quite clean, e.g. perfect balanced labels, we don't have to deal with class imbalance problem. Also we can use binary accuracy to evaluate our model performance.
- There is no missing values and duplicates at all. 
- Seems all texts in lower case by simple eyeballing.
- There are some phrase contractions found -> we can expand them to improve the model's performance.
- There are some non-English words -> we need to remove them to avoid noise interference.

Second, some basic data cleaning steps performed on "Data Cleaning" part:

- Remove non-English examples by python library "langdetect" + eyeballing e.g. 51 examples removed from train set.
- Expand contractions by regular expression. I did not use package to expand as most of the packages requires long processing time.
- Process data by "DataPipeline", please refer to the "DataPipeline" section for more details

Third, I tried TF-IDF model to compare it's performance with NN model.

TF-IDF is an simple and time efficient way of analyzing text data. The drawbacks is very obvious - it cannot analyze the relationship between words, also it ignores the sequence, context and grammar completely. since it is time efficient, we can take it as the benchmark of this project.

I have tried two of the most commonly used model, and below are their performance (accuracy):

- TF-IDF + logistic regression: 0.51
- TF-IDF + random forest: 0.53

Lastly, I have performed the result analysis for the trained BERT model, please refer to the "BERT Model Result Analysis" section for more details.

# About datapipeline_6estate.py

This is the core pipeline for data cleaning and modelling.

There are two classes that can be imported to use: "DataPipeline" and "Model"

- "DataPipeline" is a one-stop cleaning pipeline built for general pre-processing of train, test, val or any other datasets from raw text data into cleaned text data.

- "Model" is a wrapper for HuggingFace distilled BERT model. I have chosen the distilled BERT model as it is the most advanced bidirectional NLP model with outstanding performance in language understanding. The benefit of BERT is that it can understand the grammar, context, sequence and punctuations apart from word itself. As the dataset is not big in size, a smaller/distilled version of the model is sufficient.

# About experiment_6estate.py

Due to the limited compute resource that my laptop can render, I have to train the BERT model on cloud - Polyaxon platform, built on kubernetes cluster.

The "experiment_6estate.py" is the instruction code to run Polyaxon training experiment.

The basic steps involved in this experiment:

- Setup logger and storing path according to current timing.
- Run DataPipeline and model
- Save training results, running information and trained model

In addition, to run Polyaxon experiment, we also need the config files below:

- conda.yml - to install required libraries

- experiment.df - dockerfile to setup running environment

- experiment.yml - pass config parameters

- link_workspace.sh - connect persistent storage on Polyaxon

- config_6estate.yml - pass all runtime parameters for DataPipeline and Model

# BERT Model Result Analysis

The trained BERT model has the best performance of ***0.88*** binary accuracy, which is much better than our benchmark model's performance.

<img src=".\train_graph.png" alt="loss and accuracy graph during model training" style="zoom:50%;" />

Based on the graph above, we can see the BERT model learns the training data quite fast, train and val graph crosses at epoch 2, and then the validation accuracy becomes quite stable. By test and error, I have figured out the best set of hyperparameters as below (partially shown here):

- [ ] model: "bert-base-uncased"

- [ ] number of labels: 1

- [ ]  batch size: 128

- [ ]  learning rate: 0.00005

- [ ]  learning rate decay: 0.9

- [ ]  number of epochs : 17

- [ ]  max sentence length: 512

Based on the trained model, we can get the confusion matrix as below(refer to the analysis.ipynb for more details)

<img src=".\confusion_matrix.png" alt="confusion matrix" style="zoom:50%;" />

In addition, the recall is 0.9 and precision is 0.86, both of them are acceptable as expected.



# Reference

The reference links that I have referred:

[Huggingface transformer]: https://huggingface.co/transformers/custom_datasets.html
[freeze specific BERT layers]: https://colab.research.google.com/drive/1EAVhQGdVvXbCu8gGq0lZ9dOnN4jJtvAj?usp=sharing

