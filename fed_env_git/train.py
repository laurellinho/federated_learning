import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import os
from model import *
import json
import datetime
# Intitalize the serverer state and returns it
def initialize(fed_scheme):
    return fed_scheme.initialize()

# Perform one local step on all the clients from current server state
# and update server state
def train_epoch(state, fed_scheme,clients):
    result = fed_scheme.next(state, clients)
    train_metrics = result.metrics['client_work']['train']
    print('Training: loss={l:.3f}, accuracy={a:.3f}'.format(l=train_metrics['loss'], a=train_metrics['accuracy']))
    return result

def train_scheme(local_steps, model, num_of_rounds, fed_scheme, train, val, dataset,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches, dirpath, experiment_name):
    val_metrics = {}
    train_metrics = {}
    for i in range(num_of_rounds):
        val_metrics[str(i)] = {'val_loss': 0, 'val_accuracy': 0} 
        train_metrics[str(i)] = {'train_loss': 0, 'train_accuracy': 0}

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_log_dir = "/tmp/logs/scalars/" + experiment_name + '/' + current_time + '/training'
    test_log_dir = "/tmp/logs/scalars/" + experiment_name + '/' + current_time + '/validation'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    
    state = initialize(fed_scheme)



    for i in range(num_of_rounds):
        print('Epoch: ' + str(i+1) + '/' + str(num_of_rounds))
        # Training model on clients data and updating the state
        result = train_epoch(state, fed_scheme, train)
        state = result.state

        train_result = result.metrics['client_work']['train']
        train_metrics[str(i)]['train_loss'] = train_result['loss']
        train_metrics[str(i)]['train_accuracy'] = train_result['accuracy']
        with train_summary_writer.as_default():
            for name, value in result.metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=i)
        val_state = fed_scheme.next(state, val)
        val_result = val_state.metrics['client_work']['train']
        val_metrics[str(i)]['val_loss'] = val_result['loss']
        val_metrics[str(i)]['val_accuracy'] = val_result['accuracy']
        print('Validation: loss={l:.3f}, accuracy={a:.3f}'.format(l=val_result['loss'], a=val_result['accuracy']))
        with test_summary_writer.as_default():
            for name, value in val_state.metrics['client_work']['train'].items():
                tf.summary.scalar(name, value, step=i)

        # Select new train & test data, comment this line to train on same clients each round
        train, val, temp = make_federated_data(model,dataset,local_steps,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches, False)
    return state, val_metrics, train_metrics