import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import os
from rnn_model import *
from lstm_model import *

from transformers import TFGPT2Model

def load_dickens_model(batch_size):
    urls = {1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
    8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
    assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
    url = urls[batch_size]
    local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)
    return tf.keras.models.load_model(local_file, compile=False)



def create_keras_model(parent_model, loss_function, metrics_function, BATCH_SIZE):
    if parent_model == 'dickens':
        keras_model = load_dickens_model(batch_size=BATCH_SIZE)
        keras_model.compile(loss=loss_function
                            ,metrics=metrics_function)
    elif parent_model == 'simple_rnn':
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
    elif parent_model == 'lstm':
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
    elif parent_model == 'gpt-2':
        keras_model = TFGPT2Model.from_pretrained('gpt2')

    else:
        print('no such model implemented yet')
        keras_model = 0
    return keras_model




class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

        def __init__(self, name='accuracy', dtype=tf.float32):
            super().__init__(name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
            y_true = tf.reshape(y_true, [-1, 1])
            y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
            return super().update_state(y_true, y_pred, sample_weight)



def create_fed_scheme(model, dataset, train_data, scheme, client_optimizer, BATCH_SIZE,server_optimizer = None, client_weighting = None, model_distributor = None,  model_aggregator = None, metrics_aggregator = None):
    
    class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

        def __init__(self, name='accuracy', dtype=tf.float32):
            super().__init__(name, dtype=dtype)

        def update_state(self, y_true, y_pred, sample_weight=None):
            vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
            y_true = tf.reshape(y_true, [-1, 1])
            y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
            return super().update_state(y_true, y_pred, sample_weight)

    # Select loss and metrics function
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics_function = [FlattenedCategoricalAccuracy()]

    keras_model = create_keras_model(model, loss_function, metrics_function, BATCH_SIZE)


    def model_fn1():
        input_spec = train_data[0].element_spec
        keras_model = create_keras_model(model, loss_function, metrics_function, BATCH_SIZE)
        keras_model.build(input_spec[0].shape)
        keras_model.summary()
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()])
    
    # Clone the keras_model inside `create_tff_model()`, which TFF will
    # call to produce a new copy of the model inside the graph that it will
    # serialize. Note: we want to construct all the necessary objects we'll need# _inside_ this method.
    def model_fn2():
        # TFF uses an `input_spec` so it knows the types and shapes  
        # that your model expects.
        input_spec = train_data[0].element_spec
        keras_model_clone = tf.keras.models.clone_model(keras_model)
        keras_model_clone.summary()

        return tff.learning.from_keras_model(
            keras_model_clone, 
            input_spec=input_spec, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            metrics=[FlattenedCategoricalAccuracy()])

    if model == 'dickens':
        model_builder = model_fn2
    else:
        model_builder = model_fn1
    
    if scheme == 'fedavg':
        fed_scheme = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn = model_builder,
        client_optimizer_fn=client_optimizer
        #server_optimizer_fn=server_optimizer,
        #client_weighting=client_weighting,
        #model_distributor = model_distributor,
        #model_aggregator = model_aggregator,
        #metrics_aggregator = metrics_aggregator
        )
    if scheme == 'fedsgd':
        fed_scheme = tff.learning.algorithms.build_fed_sgd(
        model_fn=model_builder,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate = 1),
        #model_distributor = model_distributor,
        #model_aggregator = model_aggregator,
        #metrics_aggregator = metrics_aggregator
        )

    print(fed_scheme.initialize.type_signature.formatted_representation())
    return fed_scheme
    
