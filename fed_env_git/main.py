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

# Initialize random seed
random.seed(10)


# Specify Configs
with open("/configs/configlstmlocal.json", "r") as f:
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

        # Load initial train data
        federated_train_data, federated_val_data, federated_test_data = make_federated_data(model,dataset,clients,local_steps, SEQ_LENGTH, BATCH_SIZE, BUFFER_SIZE, Number_of_Batches, True)
        
        # Create federated scheme object
        fed_scheme = create_fed_scheme(model, dataset, federated_train_data, scheme, client_optimizer, BATCH_SIZE)
        
        # Train model on client train data
        final_state, val_metrics, train_metrics = train_scheme(local_steps,model, NUM_ROUNDS, fed_scheme, federated_train_data, federated_val_data, dataset,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches, dirpath, experiment_name)

        # Evaluate the final model by testing it on the full test set
        test_loss, test_accuracy = evaluate_state(federated_test_data, fed_scheme, final_state, model, BATCH_SIZE)

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
        
        # Send server state over on model
        if model == 'dickens':
            temp_model = load_dickens_model(BATCH_SIZE)
            temp_model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[FlattenedCategoricalAccuracy()])
            model_weights = fed_scheme.get_model_weights(final_state) 
            model_weights.assign_weights_to(temp_model)
            # Save model weights
            temp_model.save(file_path)
        else:
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics_function = [FlattenedCategoricalAccuracy()]
            
            input_spec = federated_train_data[0].element_spec
            temp_model = create_keras_model(model, loss_function, metrics_function, BATCH_SIZE)
            temp_model.build(input_spec[0].shape)
            temp_model.summary()
            save_model = tff.learning.from_keras_model(
                            temp_model,
                            input_spec=input_spec,
                            loss=loss_function,
                            metrics=metrics_function)
            model_weights = fed_scheme.get_model_weights(final_state)
            # Write over the saved model weights on model 
            model_weights.assign_weights_to(save_model)
            

            # Save model weights
            tff.learning.models.save(save_model,file_path)
        
        tf.keras.backend.clear_session()
