from .datapipeline_6estate import DataPipeline, Model
import logging
import argparse
import yaml
import os
import time
import pandas as pd
from polyaxon_client.tracking import Experiment

def run_experiment(config, path):   
    experiment = Experiment() 
    logger.info("Starting experiment...")

    dp = DataPipeline()
    train = dp.transform_data(config["dataset"]["training_path"])
    # remove non-English sentences as noise
    train = train.drop(config["datapipe"]["drop_list"], axis=0)
    test = dp.transform_data(config["dataset"]["test_path"])
    val = dp.transform_data(config["dataset"]["val_path"])
    logger.info("Finished loading data")

    train_size = train.shape[0]
    val_size = val.shape[0]
    test_size = test.shape[0]
    logger.info(f"train size:{train_size}")
    logger.info(f"test size:{test_size}")
    logger.info(f"val size:{val_size}")

    logger.info("Starting Modelling...")
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

    experiment.log_metrics(test_acc=test_accuracy)
    
    Bert.save_model()
    Bert.save_predicted()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', action='store', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    # create time stamp
    t = time.localtime()
    timestamp = time.strftime('%Y%m%d%H%M', t)
    # create folder to save model and logs 
    folder_name = "6estate" + timestamp 
    path = os.path.join(config["model"]["PATH"], folder_name) # where we store the model
    os.mkdir(path)
    store_logs = os.path.join(path, "info.log") # where we store the logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=store_logs,
        filemode='w',
        format='%(asctime)s , %(name)s - %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S', 
        level=logging.INFO
    )
    logger = logging.getLogger("6estate")
    logger.info(f"parameters are:{config}")

    run_experiment(config, path)
