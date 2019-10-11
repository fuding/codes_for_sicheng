from easydict import EasyDict


def get_config():
    conf = EasyDict()

    conf.arch = "expandnet"
    conf.model = "SingleHDR"
    conf.model_name = conf.arch + ""

    conf.use_cpu = False
    conf.is_train = True
    conf.gpu_ids = [0]
    conf.epoch = 5000
    conf.start_epoch = 0
    conf.learning_rate = 1e-4
    conf.beta1 = 0.5
    conf.loss = 'l2'  # l1 or l2
    conf.lr_scheme = "MultiStepLR"
    conf.lr_steps = [1000 * 773 / 8, 2000 * 773 / 8]
    conf.lr_gamma = 0.1
    conf.loss2 = None

    conf.dataroot_ref = "/home/sicheng/data/fivek/train/HDR"
    conf.dataroot_inp = "/home/sicheng/data/fivek/train/LDR_HR"
    conf.data_type = 'img'
    conf.dataset_name = 'FiveK'
    conf.preprocess = 'expandnet'
    conf.type_in = 'uint8'
    conf.type_out = 'uint16'
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

    conf.resume_step = 0
    if conf.resume_step == 0:
        conf.pretrained = None
        conf.resume = None
    else:
        conf.pretrained = '/home/sicheng/program/High_Dynamic_Range/BasicHDR/experiments/' + conf.model_name + '/models/' + str(
            conf.resume_step) + '_G.pth'
        conf.resume = '/home/sicheng/program/High_Dynamic_Range/BasicHDR/experiments/' + conf.model_name + '/training_state/' + str(
            conf.resume_step) + '.state'

    return conf