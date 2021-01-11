#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com

def get_configs():
    paras = {  # 'fusion_ways': [ 'mul', 'cat', 'max'],
        # 'data_name': 'tiny-imagenet200',
        'data_name': 'ChemBook',
        # 'data_name': 'Chembl',
        # 'data_name': 'PubChem',
        'fusion_ways': ['add', 'mul', 'cat', 'max', 'avg'],
        'fused_nb_feats': 128,
        'nb_view': 5,
        'pop_size': 28,
        'nb_iters': 20,
        # training parameter settings
        'result_save_dir': 'EDF-True' + '-128-5' + 'result',
        'gpu_list': [0, 1, 2, 3, 4, 5, 6],
        'epochs': 100,
        'batch_size': 64,
        'patience': 10,
        # EF
        'is_remove': False,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'noisy': True,
        'max_len': 40,
        # data set information
        'image_size': {
            'w': 230, 'h': 230, 'c': 3},
        # 'classes': 10000,
        'classes': 200,
    }
    return paras
