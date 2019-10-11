from easydict import EasyDict


def get_config():
    conf = EasyDict()

    conf.arch = "multistream_6_01"
    conf.model = "MultiHDR_01"
    conf.model_name = conf.arch + "_l1"

    conf.save_results = False
    conf.hdrvdp = False

    conf.is_train = False
    conf.gpu_ids =  [0]
    conf.use_cpu = False

    conf.dataset_dir = "/home/sicheng/data/hdr/multi_ldr_hdr_test/"
    conf.exp_path = "/home/sicheng/data/hdr/multi_ldr_hdr_test/exp.json"
    conf.dataset_name = 'Multi_LDR_HDR_01'
    conf.batch_size = 8
    conf.c_dim = 3
    conf.num_shots = 3
    conf.n_workers = 8
    conf.use_shuffle = True

    conf.use_tb_logger = True
    conf.experiments_dir = "../../experiments/" + conf.model_name

    conf.save_freq = 2000
    conf.print_freq = 200

    conf.best_model = 306000
    conf.pretrained = '/home/sicheng/program/High_Dynamic_Range/BasicHDR/experiments/' + conf.model_name + \
                      '/models/' + str(conf.best_model) + '_G.pth'

    conf.results_dir = '../../results/' + conf.model_name
    conf.start_model = 100000
    conf.end_model = 500001

    conf.need_resize = False
    conf.size = (720, 480)


    return conf