import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import pandas as pd

from model import *
from data import *
from train import *
from eval import *

import json 
import time
import os
import datetime

class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

        def __init__(self, name='accuracy', dtype=tf.float32):
            super().__init__(name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
            y_true = tf.reshape(y_true, [-1, 1])
            y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
            return super().update_state(y_true, y_pred, sample_weight)

# Initialize random seed
random.seed(10)

with open("/configs/configtest.json", "r") as f:
    tmp_dict = json.load(f)
    print(tmp_dict)

    for config in tmp_dict:
        # Specify configuration
        experiment_name = tmp_dict[config]["name"]
        dataset = tmp_dict[config]["dataset"]
        model = tmp_dict[config]["model"]
        clients = tmp_dict[config]["clients"]
        SEQ_LENGTH = tmp_dict[config]["SEQ_LENGTH"]
        BATCH_SIZE = tmp_dict[config]["BATCH_SIZE"]
        BUFFER_SIZE = tmp_dict[config]["BUFFER_SIZE"]
        NUM_ROUNDS = tmp_dict[config]["NUM_ROUNDS"]
        Number_of_Batches = tmp_dict[config]['Number_of_Batches']  
        scheme = tmp_dict[config]["scheme"]
        local_steps = tmp_dict[config]['local_steps']
        #lr = tmp_dict[config]['learning_rate']

        # Adjust the ammount of data to be equal no matter number of clients
        # and epochs to be equal depending on the local steps
        Number_of_Batches = int(Number_of_Batches/clients)
        NUM_ROUNDS = int(NUM_ROUNDS/local_steps)


        # Create empty directory to save things in
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(os.path.exists(f"../results/{experiment_name}/"))
        if not os.path.exists(f"../results/{experiment_name}/"):
            os.mkdir(f"../results/{experiment_name}/")
            print('created new folder for saving at')
        
        dirpath = f"../results/{experiment_name}/{timestamp}/"
        os.mkdir(dirpath)

        # Start time for config
        t0 = time.time()

        # Specify scheme
        client_optimizer = lambda: tf.keras.optimizers.SGD(learning_rate = 0.5)
        #server_opt = lambda: tf.keras.optimizers.SGD(learning_rate=lr)

        if model == 'simple_rnn':
            # Length of the vocabulary in StringLookup Layer
            vocab_size = len(list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'))

            # The embedding dimension
            embedding_dim = 256

            # Number of RNN units
            rnn_units = 256
            keras_model = RNNModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units=rnn_units)
        elif model == 'lstm':
            # Length of the vocabulary in StringLookup Layer
            vocab_size = len(list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'))

            # The embedding dimension
            embedding_dim = 256

            # Number of RNN units
            rnn_units = 256
            keras_model = LSTMModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                rnn_units=rnn_units)

        # Compile model with loss and optimizer
        keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.5),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=[FlattenedCategoricalAccuracy()])

        # Load train, validation and test_data
        train_data, val_data, test_data = load_data(dataset)

        val_metrics = {}
        train_metrics = {}
        for i in range(NUM_ROUNDS):
            val_metrics[str(i)] = {'val_loss': 0, 'val_accuracy': 0} 
            train_metrics[str(i)] = {'train_loss': 0, 'train_accuracy': 0}


        for i in range(NUM_ROUNDS):
            print(i)
            # Select x clients and generate data to train and validate on, put it all in centralized dataset
            temp_clients = client_generator(clients,train_data)
            preprocessed_central = tf.data.Dataset.from_tensor_slices(
                preprocess_text(model,dataset,local_steps,train_data,temp_clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches)).flat_map(lambda x: x)
            val_clients = client_generator(clients,val_data)
            preprocessed_val = tf.data.Dataset.from_tensor_slices(
                preprocess_text(model,dataset,local_steps,val_data,val_clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches)).flat_map(lambda x: x)
            history = keras_model.fit(preprocessed_central,
                                      validation_data=preprocessed_val)
            train_metrics[str(i)]['train_loss'] = history.history['loss']
            train_metrics[str(i)]['train_accuracy'] = history.history['accuracy']
            val_metrics[str(i)]['val_loss'] = history.history['val_loss']
            val_metrics[str(i)]['val_accuracy'] = history.history['val_accuracy']


        # Stopping time
        t1 = time.time()
        deltat = t1 - t0
        print('Total time for '+ config +": " + str(deltat))
        
        # Convert metrics to pandas and save to csv
        val = pd.DataFrame(val_metrics)
        train = pd.DataFrame(train_metrics)
        print(val)
        print(train)
        res = pd.concat([train, val])
        print(res)
        res['time'] = deltat
        res.to_csv(dirpath + 'results')

        # Save config to file
        print(tmp_dict[config])
        df = pd.DataFrame(tmp_dict[config], index=[0])
        df.to_csv(dirpath + 'config')

        # Save TensorBoard, Model, Settings and Performance in directory
        file_path = os.path.join(dirpath,"model")

        # Save model weights
        keras_model.save(file_path)
        
