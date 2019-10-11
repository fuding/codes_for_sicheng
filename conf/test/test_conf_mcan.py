from easydict import EasyDict


def get_config():
    conf = EasyDict()

    conf.arch = "mcan_s"
    conf.model = "MultiHDR"
    conf.model_name = conf.arch + ""

    conf.is_train = False
    conf.gpu_ids =  [2]
    conf.use_cpu = False

    conf.dataset_dir = "/home/sicheng/data/hdr/multi_ldr_hdr_test/"
    conf.exp_path = "/home/sicheng/data/hdr/multi_ldr_hdr_test/exp.json"
    conf.dataset_name = 'Multi_LDR_HDR'
    conf.batch_size = 8
    conf.c_dim = 3
    conf.num_shots = 3
    conf.n_workers = 4
    conf.use_shuffle = True

    conf.use_tb_logger = True
    conf.experiments_dir = "../../experiments/" + conf.model_name

    conf.save_freq = 2000
    conf.print_freq = 200

    conf.pretrained = '/home/sicheng/program/High_Dynamic_Range/BasicHDR/experiments/mcan_s/models/282000_G.pth'

    conf.save_results = False
    conf.results_dir = '../../results/' + conf.model_name
    conf.start_model = 100000
    conf.end_model = 306001

    conf.need_resize = False
    conf.size = (720, 480)


    return conf