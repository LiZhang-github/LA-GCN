from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from scipy import sparse
from models import GCN_MASK
import scipy.io as scio
import pandas as pd
import pickle
import pdb


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_mask', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('seed',6, 'define the seed.')
flags.DEFINE_float('train_percentage', 0.1 , 'define the percentage of training data.')
flags.DEFINE_integer('fastgcn_setting', 0, 'define the training setting for gcn or fastgcn setting')
flags.DEFINE_integer('start_test', 80, 'define from which epoch test')
flags.DEFINE_integer('train_jump', 0, 'define whether train jump, defaul train_jump=0')
flags.DEFINE_integer('attack_dimension', 0, 'define how many dimension of the node feature to attack')


# Set random seed
seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)


k_att = FLAGS.train_percentage
test_result_gather = []



# Load data
add_all, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.fastgcn_setting, 
                                                                                        FLAGS.dataset,
                                                                                        k_att, FLAGS.attack_dimension,
                                                                                        FLAGS.train_jump)

# Some preprocessing

features = preprocess_features(features,adj) ## type(features) is tuple

if FLAGS.model == 'gcn_mask':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_MASK
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not usedouts
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(add_all, placeholders, input_dim=features[2][1], logging=True)
sess = tf.Session()
all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(all_variables)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.mask], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

train_gcnmask_gather = []
test_gcnmask_gather = []
val_gcnmask_gather = []

best_test_result = 0

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)

    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.mask], feed_dict=feed_dict)
    ## the last layer of nn output is model.outputs
    cost, acc, duration, val_gcnmask = evaluate(features, support, y_val, val_mask, placeholders)

    cost_val.append(cost)  ##transpose to numpy and reshape, then write to txt
    train_loss.append(outs[1])
    train_accuracy.append(outs[2])
    val_loss.append(cost)
    val_accuracy.append(acc)

    
    train_gcnmask_gather.append(outs[4])
    val_gcnmask_gather.append(val_gcnmask)
    # set the preserved values
    np.set_printoptions(precision=3)


    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("aaa=========",epoch)
        print("Early stopping...")
        break

    if epoch > FLAGS.start_test:

        test_cost, test_acc, test_duration, test_gcnmask = evaluate(features, support, y_test, test_mask, placeholders)

        if test_acc > best_test_result:
            best_test_result = test_acc

        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        
feed_dict_test = construct_feed_dict(features, support, y_test, test_mask, placeholders)
test_cost, test_acc, test_duration, test_gcnmask = evaluate(features, support, y_test, test_mask, placeholders)
test_gcnmask_gather = test_gcnmask

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_result_gather.append([k_att, best_test_result])

