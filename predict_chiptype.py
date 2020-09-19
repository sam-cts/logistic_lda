"""
Copyright 2019 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0

Evaluation script for both supervised and unsupervised models.

"""

import csv
import json
import os
import pickle
import pprint
import sys
from argparse import ArgumentParser
from collections import defaultdict

import tensorflow as tf
from PIL import Image, ImageOps

import data_iter_batched
import data_iter_online
import embeddings
import models
import utils


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--experiment_name', type=str, default='ng20_unsupervised_debug')
    parser.add_argument('--evaluate_unsupervised', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--metadata_dir', type=str, default='metadata')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--hidden_units', type=int, nargs='+', default=None)
    parser.add_argument('--vocab_file', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--topic_bias_regularization', type=float, default=None)
    parser.add_argument('--author_topic_weight', type=float, default=None)
    parser.add_argument('--model_regularization', type=float, default=None)
    parser.add_argument('--num_valid', type=float, default=None)
    parser.add_argument('--items_per_author', type=int, default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--n_unsupervised_topics', type=int, default=None)
    parser.add_argument('--n_author_topic_iterations', type=int, default=None)
    parser.add_argument('--embedding', type=str, default=None)
    parser.add_argument('--use_author_topics', type=int, default=None)
    parser.add_argument('--output_predictions_file', type=str, default='predictions.csv')
    parser.add_argument('--cache', action='store_true', help='Cache data for faster iterations')

    args = parser.parse_args()

    model_dir = os.path.join(args.metadata_dir, args.experiment_name)

    # stores parameters used during training
    args_file = os.path.join(model_dir, 'args.json')
    args_dict = {}
    if tf.gfile.Exists(args_file):
        with tf.gfile.GFile(args_file, 'r') as handle:
            args_dict = json.load(handle)

    for key, value in vars(args).items():
        if value is None:
            if not args_dict:
                print('Could not find hyperparameters used during training.', end=' ', file=sys.stderr)
                print('Please specify `{}` manually.'.format(key), file=sys.stderr)
                sys.exit(1)
            elif key not in args_dict:
                print('Could not find `{}` in hyperparameters.'.format(key), end=' ', file=sys.stderr)
                print('Please specify manually.', file=sys.stderr)
                sys.exit(1)
            else:
                setattr(args, key, args_dict[key])

    pprint.pprint(vars(args))
