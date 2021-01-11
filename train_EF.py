# encoding=utf-8
# author: liang xinyan
# email: liangxinyan48@163.com
import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import tensorflow as tf
# https://stackoverflow.com/questions/60130622/warningtensorflow-with-constraint-is-deprecated-and-will-be-removed-in-a-future
tf.get_logger().setLevel('ERROR')
import numpy as np
from os.path import join as pjoin
import random
from sklearn import metrics
from multiprocessing import Pool
import time
import argparse
from data_utils.data_uitl import get_views
import config
from code2net import code2net
import population_init
import gen_offspring
import utils


paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']
batch_size = paras['batch_size']
epochs = paras['epochs']
classes = paras['classes']
pop_size = paras['pop_size']
nb_iters = paras['nb_iters']
data_name = paras['data_name']

# ['add', 'mul', 'cat', 'max', 'avg']
# Only load all view once
data_base_dir = os.path.join('..', data_name)
view_data_dir = os.path.join(data_base_dir, 'view')
view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir=view_data_dir)


def train_individual(individual_code, result_save_dir='.', gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    nb_view = (len(individual_code) + 1) // 2  # 视图个数
    view_train_xx, view_test_xx = [], []
    for i in individual_code[:nb_view]:
        view_train_xx.append(view_train_x[i])
        view_test_xx.append(view_test_x[i])
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    nb_feats = [i.shape[1] for i in view_train_x]
    model = code2net(individual_code=individual_code, nb_feats=nb_feats)
    # print(model.summary())
    # print('*' * 20, f'individual code:{individual_code_str}')
    adam = tf.keras.optimizers.Adam()
    topk = tf.keras.metrics.top_k_categorical_accuracy
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', topk])
    checkpoint_filepath = os.path.join(result_save_dir, individual_code_str + '.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=paras['patience'])
    csv_filepath = os.path.join(result_save_dir, individual_code_str + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_filepath)

    model.fit(view_train_xx, train_y, batch_size=batch_size, epochs=epochs,
              verbose=0, validation_data=(view_test_xx, test_y),
              callbacks=[csv_logger, early_stop, checkpoint])
    model_best = tf.keras.models.load_model(checkpoint_filepath)
    pre_y = model_best.predict(view_test_xx)
    pre_y = np.argmax(pre_y, axis=1)
    true_y = np.argmax(test_y, axis=1)
    acc = metrics.accuracy_score(true_y, pre_y)
    return individual_code_str + ',' + str(acc)


def find_same_code_acc(individual_code, result_save_dir='.'):
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    return individual_code_str


def record_code(individual_code, result_save_dir='.'):
    individual_code_str = '-'.join([str(ind) for ind in individual_code])
    return individual_code_str


def list2str(list1):
    return '-'.join([str(i) for i in list1])


def multi_proccess_train(i_iter, Q_t, shared_code_sets):
    gpu_list = paras['gpu_list']
    gpus = len(gpu_list)
    gpu_idx = 0
    pool = Pool(gpus)
    individual_code_str = []
    pop_size1 = len(Q_t)
    for ind_i in np.arange(0, pop_size1):
        print(len(Q_t), '==========', ind_i+1)
        code_str = list2str(Q_t[ind_i])
        utils.write_result_file(','.join([str(i_iter+1), code_str]),
                                fn=os.path.join(result_save_dir, 'history.csv'))
        if code_str not in shared_code_sets:
            shared_code_sets.add(code_str)
            individual_code_str.append(
                pool.apply_async(func=train_individual,
                                 args=(Q_t[ind_i], result_save_dir, str(gpu_idx)))) #GPU +1
            gpu_idx += 1

        if gpu_idx == gpus or ind_i == (pop_size1-1):
            pool.close()
            pool.join()
            for ss in individual_code_str:
                utils.write_result_file(ss.get(), fn=os.path.join(result_save_dir, 'result.csv'))
            pool = Pool(gpus)
            gpu_idx = 0
            individual_code_str = []


def train():
    shared_code_sets = set()
    # 1. population initialization
    print(f'The number of views: {len(view_train_x)}')
    ini_population = population_init.generate_population(
        views=len(view_train_x), pop_size=pop_size, verbose=0)

    # 2.init population fitness
    start = time.time()
    P_t = ini_population
    multi_proccess_train(i_iter=-1, Q_t=P_t, shared_code_sets=shared_code_sets)

    # 3. gen_offspring
    for i in tqdm(range(paras['nb_iters'])):
        print(f'==================={i+1}/', paras['nb_iters'])
        Q_t = gen_offspring.gen_offspring(P_t)
        multi_proccess_train(i_iter=i, Q_t=Q_t, shared_code_sets=shared_code_sets)
        P_t = gen_offspring.selection(P_t, Q_t)
        print('=' * 60, i+1, 'End.')

    print(f'Total time is :{time.time()-start}')
    utils.write_result_file(str(time.time()-start), fn=os.path.join(result_save_dir, 'history.csv'))


if __name__ == '__main__':
    result_save_dir = pjoin(data_name+'_view_result', paras['result_save_dir'])
    print(result_save_dir)
    print(data_name, fused_nb_feats)
    os.makedirs(result_save_dir, exist_ok=True)
    train()
    print(result_save_dir)
    print(data_name, fused_nb_feats)
