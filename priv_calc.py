import tensorflow as tf
from privacy_accountant import accountant, utils
import argparse
import time

import pickle as pkl

def calc_priv(noise, epochs, training_size, batch_size):
    privacy_history = []
    with tf.Session() as sess:
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)

        num_batches = epochs * (training_size / batch_size)
        target_eps = [0.125,0.25,0.5,1,2,4,8]
        priv_accountant = accountant.GaussianMomentsAccountant(training_size)

        print('accum privacy, batches: ' + str(num_batches))
        priv_start_time = time.clock()
        privacy_accum_op = priv_accountant.accumulate_privacy_spending(
          [None, None], args.noise, batch_size)
        tf.global_variables_initializer().run()
        for index in range(num_batches):
            with tf.control_dependencies([privacy_accum_op]):
                spent_eps_deltas = priv_accountant.get_privacy_spent(
                    sess, target_eps=target_eps)
                privacy_history.append(spent_eps_deltas)
            sess.run([privacy_accum_op])

        print('priv time: ', time.clock() - priv_start_time)

        if spent_eps_deltas[-3][1] > 0.0001:
            raise Exception('spent privacy')

    pkl.dump(privacy_history, open('./privacy/' + str(noise) + '_' +
             str(epochs) + '_' + str(training_size) + '_' + str(batch_size) +
             '.p', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--training_size", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    calc_priv(args.noise, args.epochs, args.training_size, args.batch_size)
