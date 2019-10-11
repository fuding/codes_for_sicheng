from easydict import EasyDict



def get_config():
    conf = EasyDict()

    conf.arch = "awsrn_sd"
    conf.model = "MultiHDR"
    conf.model_name = conf.arch + ""

    conf.use_cpu = False
    conf.is_train = True
    conf.gpu_ids = [0]
    conf.epoch = 400
    conf.start_epoch = 0
    conf.learning_rate = 0.0002
    conf.beta1 = 0.5
    conf.loss = 'l2'  # l1 or l2
    conf.lr_scheme = "MultiStepLR"
    conf.lr_steps = [100*2387]
    conf.lr_gamma = 0.1

    conf.dataset_dir = "/home/sicheng/data/hdr/multi_ldr_hdr_patch/"
    conf.exp_path = "/home/sicheng/data/hdr/multi_ldr_hdr_patch/exp.json"
    conf.dataset_name = 'Multi_LDR_HDR'
    conf.batch_size = 8
    conf.load_size = 256
    conf.fine_size = 256
    conf.c_dim = 3
    conf.num_shots = 3
    conf.n_workers = 4
    conf.use_shuffle = True

    conf.use_tb_logger = True
    conf.experiments_dir = "../../experiments/" + conf.model_name
    conf.log_dir = "../../tb_logger/" + conf.model_name
    conf.save_freq = 2000
    conf.print_freq = 200

    conf.pretrained = '/home/sicheng/program/High_Dynamic_Range/BasicHDR/experiments/awsrn_sd/models/142000_G.pth'
    conf.resume = '/home/sicheng/program/High_Dynamic_Range/BasicHDR/experiments/awsrn_sd/training_state/142000.state'

    return conf