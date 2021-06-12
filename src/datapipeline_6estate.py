import os
import yaml
import demjson
import pandas as pd
import re
import numpy 
from numpy import savetxt
import logging
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import BinaryAccuracy
from transformers import AutoTokenizer
from transformers import TFDistilBertForSequenceClassification

AUTOTUNE = tf.data.experimental.AUTOTUNE

logger = logging.getLogger("6estates")
logging.basicConfig(level=logging.INFO)

class DataPipeline():
    """
    DataPipe processes the input text data and return processed data for model training
    1.Load data
    2.Clean data and generate labels
    3.Return cleaned text and labels
    """

    def __init__(self, flag_contraction=True, flag_digits=True):
        self.__flag_contraction = flag_contraction
        self.__flag_digits = flag_digits
        logger.info("Datapipeline is initiated")
    
    @staticmethod
    def expand_contractions(word:str) -> str:
        """expand some of the contracted words

        Returns:
            string -- word without contractions
        """
        word = re.sub(r"won\'t", "will not", word)
        word = re.sub(r"can\'t", "can not", word)
        word = re.sub(r"n\'t", " not", word)
        word = re.sub(r"\'re", " are", word)
        word = re.sub(r"\'d", " would", word)
        word = re.sub(r"\'ll", " will", word)
        word = re.sub(r"\'t", " not", word)
        word = re.sub(r"\'ve", " have", word)
        word = re.sub(r"\'m", " am", word)
        return word
    
    @staticmethod
    def to_lower(word:str) -> str:
        """convert all text to lower case

        Arguments:
            word {str} -- each word

        Returns:
            str -- [description]
        """
        word = word.lower()
        return word

    @staticmethod
    def remove_digits(word:str) -> str:
        patten = '[0-9]'
        word = re.sub(patten, '', word)
        return word

    @staticmethod
    def drop_na(df:pd.DataFrame) -> pd.DataFrame:
        """drop rows with na values

        Arguments:
            df {dataframe} -- input dataframe

        Returns:
            dataframe -- dataframe with no na values
        """
        logger.info("performed dropping missing values")
        return df.dropna() 

    @staticmethod
    def drop_duplicates(df:pd.DataFrame) -> pd.DataFrame:
        """drop duplicated rows when both label and sentences are the same

        Arguments:
            df {dataframe} -- input dataframe

        Returns:
            dataframe -- dataframe without duplicates
        """
        logger.info("performed dropping duplicates")
        return df.drop_duplicates(ignore_index=True)


    @staticmethod
    def load_data(data_path:str) -> pd.DataFrame:
        """load input data input dataframe

        Arguments:
            input_path {str} -- where is the data saved
        """
        df = pd.DataFrame([demjson.decode(line) for line in open(data_path, 'r')])
        return df

    def transform_data(self, data_path:str) -> pd.DataFrame:
        """perform one-stop tranforming for data pre-processing

        Arguments:
            sentence {str} -- [text data that needs to be cleaned]
        """
        df = self.load_data(data_path)
        df = self.drop_na(df)
        df = self.drop_duplicates(df)
        if self.__flag_contraction:
            df.loc[:, "sentence"] = df.sentence.apply(lambda x: self.expand_contractions(x))
        if self.__flag_digits:
            df.loc[:, "sentence"] = df.sentence.apply(lambda x: self.remove_digits(x))
        df.loc[:, "sentence"] = df.sentence.apply(lambda x: self.to_lower(x))
        logger.info("finished transforming data")        
        return df

class Model():
    """
    DataPipe processes the input text data and return processed data for model training
    1.Load data
    2.Clean data and generate labels
    3.Return cleaned text and labels
    """

    def __init__(self, config, path):
        self.__model_path = path
        self.__tokenizer = AutoTokenizer.from_pretrained(config["model"]["MODEL"], use_fast=True)
        self.__model = TFDistilBertForSequenceClassification.from_pretrained(config["model"]["MODEL"], num_labels=config["model"]["NUM_LABELS"])
        self.__batch = config["model"]["BATCH_SIZE"]
        self.__learning_rate = config["model"]["LEARNING_RATE"]
        self.__epochs = config["model"]["EPOCHS"]
        self.__max_length = config["model"]["MAX_LENGTH"]
        logger.info("initiating model")

    def preprocess_data(self, train, val, test):
        train_encodings = self.__tokenizer(train.sentence.tolist(), truncation=True, padding=True, max_length = self.__max_length, return_tensors="tf")
        val_encodings = self.__tokenizer(val.sentence.tolist(), truncation=True, padding=True, max_length = self.__max_length, return_tensors="tf")
        test_encodings = self.__tokenizer(test.sentence.tolist(), truncation=True, padding=True, max_length = self.__max_length, return_tensors="tf")

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train.label.values
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val.label.values
        ))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test.label.values
        ))
        self.__steps_per_epoch = train_dataset.cardinality().numpy()//self.__batch 
        self.__validation_steps = val_dataset.cardinality().numpy()//self.__batch 

        dataset_train_tf = train_dataset.shuffle(train_dataset.cardinality().numpy()).batch(self.__batch)
        dataset_val_tf = val_dataset.shuffle(val_dataset.cardinality().numpy()).batch(self.__batch)
        dataset_test_tf = test_dataset.shuffle(test_dataset.cardinality().numpy()).batch(self.__batch)

        self.__train_dataset = dataset_train_tf.repeat(self.__epochs).prefetch(buffer_size=AUTOTUNE)
        self.__test_dataset = dataset_test_tf.prefetch(buffer_size=AUTOTUNE)
        self.__validation_dataset = dataset_val_tf.prefetch(buffer_size=AUTOTUNE)
        return self.__train_dataset, self.__test_dataset, self.__validation_dataset

    def build_model(self):
        for layer in self.__model.layers:
            if layer.name == "distilbert":
                layer.trainable = False

        self.__model.compile(optimizer=Adam(learning_rate=self.__learning_rate),
                             loss=BinaryCrossentropy(from_logits=True),
                             metrics=BinaryAccuracy(), 
                            ) 

    def fit_model(self):
        reduce_lr = ReduceLROnPlateau(
            factor=0.5,
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=4,
            min_lr=0.00000001)

        early_stop = EarlyStopping(
            verbose=1,
            patience=13,
            mode='auto',
            monitor='val_loss',
            restore_best_weights=True)

        history = self.__model.fit(
            x=self.__train_dataset,
            validation_data=self.__validation_dataset,
            steps_per_epoch=self.__steps_per_epoch,
            epochs=self.__epochs,
            validation_steps=self.__validation_steps,
            callbacks=[early_stop, reduce_lr])

        return history

    def evaluate_model(self):
        """model evaluation on test set

        Returns:
            CategoricalAccuracy -- return test accuracy
        """
        loss, binary_acc = self.__model.evaluate(self.__test_dataset)
        logger.info('Test Loss: %s', loss)
        logger.info('Test Accuracy: %s', binary_acc)
        return binary_acc

    def save_model(self):
        """save model on polyaxon 
        """
        logger.info(f"Saving model in {self.__model_path}")
        self.__model.save_pretrained(self.__model_path)
    
    def save_predicted(self):
        predicted = tf.round(tf.sigmoid(self.__model.predict(self.__test_dataset)["logits"]))
        logger.info(f"predicted values for test_dataset: {predicted.numpy()}")
        savetxt(os.path.join(self.__model_path, "predicted.csv"), predicted.numpy(), delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', action='store', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path) as file:
        config = yaml.safe_load(file)

    # declare drop list
    drop_list = config["datapipe"]["drop_list"]

    # initiate pipeline
    dp = DataPipeline()
    train = dp.transform_data("./data/train.json")
    # remove non-English sentences as noise
    train = train.drop(drop_list, axis=0)
    test = dp.transform_data("./data/test.json")
    val = dp.transform_data("./data/dev.json")

    train_size = train.shape[0]
    val_size = val.shape[0]
    test_size = test.shape[0]
    logger.info(f"train size:{train_size}")
    logger.info(f"test size:{test_size}")
    logger.info(f"val size:{val_size}")
    
    # initiating modelling
    # set the path to store trained model
    path = "./model"
    Bert = Model(config, path)

    Bert.preprocess_data(train, val, test)
    logger.info("Finished preprocessing input data for Bert")
    
    Bert.build_model()
    logger.info("Finished building Bert")
    
    logger.info("Starting training")
    history = Bert.fit_model()
    # save history for local plots
    pd.DataFrame.from_dict(history.history).to_csv(os.path.join(path, 'history.csv'), index=False)

    logger.info("Start testing")
    test_accuracy = Bert.evaluate_model()
    
    Bert.save_model()
    Bert.save_predicted()