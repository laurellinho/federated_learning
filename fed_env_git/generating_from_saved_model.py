import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import pandas as pd
from model import *
from data import *
from train import *
import json 
import time
import os
import datetime

vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
idx2char = np.array(vocab)
print(idx2char[48])
print(idx2char[41])
# Construct a lookup table to map string chars to indexes,
# using the vocab loaded above:
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                    dtype=tf.int64)),
    default_value=0)

def to_ids(x):
    s = tf.reshape(x, shape=[1])
    chars = tf.strings.bytes_split(s).values
    ids = table.lookup(chars)
    return ids

def split_input_target(chunk):
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    target_text = tf.map_fn(lambda x: x[1:], chunk)
    return (input_text, target_text)

model_path = "/results/test_local/2023-05-11_23-44-15/model"

model = tff.learning.models.load(model_path)

print(model.input_spec)

input = "from what i have read i can not see how to use tensorflow federated-learning (tff) for a real world application: datasets on multiple hardware clients. it all looks like its meant only for simulating federated learning."

input = "Global trade, inflation, supply chains, policies, and innovation shape the economy. Governments strive for growth and stability amidst challenges. arr"
input = input.lower()
print(input)
input = input[-100:]
print(input)
ids = to_ids(input)

ids = tf.expand_dims(ids, 0)

print(ids)

ids = tf.tile(ids,[8,1])

print(ids)
num_char = 50
for i in range(num_char):
    preds = model.predict_on_batch(ids, training = False)
    temperature = 1.0
    predictions = preds[0]
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(
        predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    pred = idx2char[predicted_id]

    input = input + pred

    ids = input[len(input)-100:]
    ids = to_ids(ids)
    ids = tf.expand_dims(ids, 0)
    ids = tf.tile(ids,[8,1])

print(input)