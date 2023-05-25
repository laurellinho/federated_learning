import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import pandas as pd
from model import *


def evaluate_state(test_data, fed_scheme, state, model, BATCH_SIZE):
    # Extract model weights
    weights = fed_scheme.get_model_weights(state)
    # loss and metrics functions
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics_function = [FlattenedCategoricalAccuracy()] 
    test_model = create_keras_model(model, loss_function, metrics_function, BATCH_SIZE)
    test_model.compile(loss=loss_function
                        ,metrics=metrics_function)
    
    input_spec = test_data.element_spec

    test_model.build(input_spec[0].shape)

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics_function = [FlattenedCategoricalAccuracy()]
    # Send weights to model
    weights.assign_weights_to(test_model)
    # Evaluate using standard keras method
    print('Final model valuation:')
    loss, accuracy = test_model.evaluate(test_data, steps=2, verbose=0)
    print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
    return loss, accuracy

