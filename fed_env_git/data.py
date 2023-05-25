import random
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import os


def make_federated_data(model,dataset, local_steps, NUM_CLIENTS,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches, test = True):
    # Load train, validation and test_data
    train_data, val_data, test_data = load_data(dataset)
    # Select clients and generate data to train and validate on
    clients = client_generator(NUM_CLIENTS,train_data)
    preprocessed_train = preprocess_text(model,dataset,local_steps,train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches)
    clients = client_generator(NUM_CLIENTS,val_data)
    preprocessed_val = preprocess_text(model,dataset,local_steps,val_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches)
    # for test we generate a full data set for all test clients to evaluate final model on
    if test:
        clients = client_generator(NUM_CLIENTS, test_data)
        preprocessed_test = tf.data.Dataset.from_tensor_slices(
        preprocess_text(model,dataset,local_steps,test_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches)).flat_map(lambda x: x)
    else:
        preprocessed_test = None

    return preprocessed_train, preprocessed_val, preprocessed_test



def load_data(dataset):

    if dataset == 'stackoverflow':
        train_data, val_data, test_data = tff.simulation.datasets.stackoverflow.load_data()
    elif dataset == 'shakespeare':
        train_data, test_data = tff.simulation.datasets.shakespeare.load_data()
        val_data = None
    else:
        print('that dataset is not yet implemented')
    return train_data, val_data, test_data


def client_generator(NUM_CLIENTS,train_data):

    clients = [None] * NUM_CLIENTS

    rng = random.sample(range(0, len(train_data.client_ids)), NUM_CLIENTS)
    clients = [train_data.client_ids[i] for i in rng]

    return clients


def preprocess_text(model,dataset, local_steps, train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches):
    if model == 'simple_rnn' or model == 'lstm':
        if dataset == 'stackoverflow':
            return preprocess_text_dickens_stackoverflow(local_steps,train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches)
        elif dataset == 'shakespeare':
            return preprocess_text_dickens_shakespeare(local_steps,train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches)
    elif model == 'dickens':
        if dataset == 'stackoverflow':
            return preprocess_text_dickens_stackoverflow(local_steps,train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches)
        elif dataset == 'shakespeare':
            return preprocess_text_dickens_shakespeare(local_steps,train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches)
    elif model == 'gpt-2':
        return preprocess_text_dickens_stackoverflow(local_steps,train_data,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE,Number_of_Batches)


def preprocess_text_dickens_stackoverflow(local_steps,dataset,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches):

    vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

    # Construct a lookup table to map string chars to indexes,
    # using the vocab loaded above:
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.constant(list(range(len(vocab))),
                                        dtype=tf.int64)),
        default_value=0)

    def to_ids(x):
        s = tf.reshape(x['tokens'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids


    def split_input_target(chunk):
        input_text = tf.map_fn(lambda x: x[:-1], chunk)
        target_text = tf.map_fn(lambda x: x[1:], chunk)
        return (input_text, target_text)


    def preprocess(dataset):
        return (
            # Map ASCII chars to int64 indexes using the vocab
            dataset.repeat(local_steps).map(to_ids)
            # Split into individual chars
            .unbatch()
            # Form example sequences of SEQ_LENGTH +1
            .batch(SEQ_LENGTH + 1, drop_remainder=True)
            # Shuffle and form minibatches
            .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            # And finally split into (input, target) tuples,
            # each of length SEQ_LENGTH.
            .map(split_input_target))

    return [
        preprocess(dataset.create_tf_dataset_for_client(x)).take(Number_of_Batches)
        for x in clients]

def preprocess_text_dickens_shakespeare(local_steps,dataset,clients,SEQ_LENGTH,BATCH_SIZE,BUFFER_SIZE, Number_of_Batches):

    vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

    # Construct a lookup table to map string chars to indexes,
    # using the vocab loaded above:
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.constant(list(range(len(vocab))),
                                        dtype=tf.int64)),
        default_value=0)


    def to_ids(x):
        s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids


    def split_input_target(chunk):
        input_text = tf.map_fn(lambda x: x[:-1], chunk)
        target_text = tf.map_fn(lambda x: x[1:], chunk)
        return (input_text, target_text)


    def preprocess(dataset):
        return (
            # Map ASCII chars to int64 indexes using the vocab
            dataset.repeat(local_steps).map(to_ids)
            # Split into individual chars
            .unbatch()
            # Form example sequences of SEQ_LENGTH +1
            .batch(SEQ_LENGTH + 1, drop_remainder=True)
            # Shuffle and form minibatches
            .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            # And finally split into (input, target) tuples,
            # each of length SEQ_LENGTH.
            .map(split_input_target))

    return [
        preprocess(dataset.create_tf_dataset_for_client(x)).take(Number_of_Batches)
        for x in clients]

