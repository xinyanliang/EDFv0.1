#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
def get_configs():
    paras = {  # 'fusion_ways': [ 'mul', 'cat', 'max'],
        'data_name': 'Chembl',
        'fusion_ways': ['add', 'mul', 'cat', 'max', 'avg'],
        # 'fused_nb_feats': 256,
        'fused_nb_feats': 64,
        'nb_view': 5,
        'pop_size': 32,
        'nb_iters': 20,
        # training parameter settings
        'result_save_dir': 'EFv2Ture_' + '64-5' + 'avg_result',
        'gpu_list': [0, 1, 2, 3, 4, 5, 6, 7],
        # 'gpu_list': [0, 1, 2, 3, 5, 6],
        'epochs': 100,
        'batch_size': 64,
        'patience': 10,
        # EF
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'is_remove': False,
        'noisy': True,
        'max_len': 40,
        # data set information
        'image_size': {
            'w': 230, 'h': 230, 'c': 1},
        'classes': 10000,
    }
    return paras
