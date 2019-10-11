import torch
import torch.nn as nn
import functools
from torch.nn import init

def create_model(conf):
    model = conf.model

    if model == 'MultiHDR':
        from .multihdr_model import MultiHDR_Model as M
    elif model == 'MultiHDR_01':
        from .multihdr_model_01 import MultiHDR_Model as M
    elif model == 'SingleHDR':
        from .singlehdr_model import SingleHDR_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    m = M(conf)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

def def_network(conf):
    network = conf.arch

    if network == 'deephdr':
        from .networks.deephdr import DeepHDR as N
        net = N()
    elif network == 'carn':
        from .networks.carn import CARN as N
        net = N()
    elif network == 'carn_m':
        from .networks.carn_m import CARN_M as N
        net = N(group=4)
    elif network == 'ahdrnet':
        from .networks.ahdrnet import AHDRNet as N
        net = N()
    elif network == 'awsrn_sd':
        from .networks.awsrn import AWSRN_SD as N
        net = N()
    elif network == 'awsrn_sd_ia':
        from .networks.awsrn_ia import AWSRN_SD_IA as N
        net = N()
    elif network == 'mcan_s':
        from .networks.mcan_s import MCAN_S as N
        net = N()
    elif network == 'awsrn_sd_ed':
        from .networks.awsrn_sd_ed import AWSRN_SD_ED as N
        net = N()
    elif network == 'awsrn_sd_ed_aw':
        from .networks.awsrn_sd_ed_aw import AWSRN_SD_ED_AW as N
        net = N()
    elif network == 'awsrn_sd_ed_c':
        from .networks.awsrn_sd_ed_c import AWSRN_SD_ED_C as N
        net = N()
    elif network == 'awsrn_sd_ed_aw_ia':
        from .networks.awsrn_sd_ed_aw_ia import AWSRN_SD_ED_AW_IA as N
        net = N()
    elif network == 'awsrn_ed_aw_mn':
        from .networks.awsrn_ed_aw_mn import AWSRN_ED_AW_MN as N
        net = N(n_feats=32, num_blocks=4)
    elif network == 'awsrn_ed_aw_mn_ca':
        from .networks.awsrn_ed_aw_mn_ca import AWSRN_ED_AW_MN_CA as N
        net = N(n_feats=32, num_blocks=4, reduction=8)
    elif network == 'awsrn_ed_aw_mn_scse':
        from .networks.awsrn_ed_aw_mn_scse import AWSRN_ED_AW_MN_scSE as N
        net = N(n_feats=32, num_blocks=4)
    elif network == 'awsrn_ed_aw_mn_scse_2':
        from .networks.awsrn_ed_aw_mn_scse_2 import AWSRN_ED_AW_MN_scSE_2 as N
        net = N(n_feats=32, num_blocks=4)
    elif network == 'awsrn_ed_aw_mn_cbam':
        from .networks.awsrn_ed_aw_mn_cbam import AWSRN_ED_AW_MN_CBAM as N
        net = N(n_feats=32, num_blocks=4)
    elif network == 'awsrn_ed_aw_mn_ram':
        from .networks.awsrn_ed_aw_mn_ram import AWSRN_ED_AW_MN_RAM as N
        net = N(n_feats=32, num_blocks=4, reduction=8)
    elif network == 'awsrn_ed_aw_mn_aw':
        from .networks.awsrn_ed_aw_mn_aw import AWSRN_ED_AW_MN_AW as N
        net = N(n_feats=32, num_blocks=4)
    elif network == 'mcan_s_ed':
        from .networks.mcan_s_ed import MCAN_S_ED as N
        net = N()
    elif network == 'awsrn_sd_ed_aw_aspp':
        from .networks.awsrn_sd_ed_aw_aspp import AWSRN_SD_ED_AW_ASPP as N
        net = N()
    elif network == 'dense_1':
        from .networks.dense_1 import Dense_1 as N
        net = N()
    elif network == 'dense_2':
        from .networks.dense_2 import Dense_2 as N
        net = N()
    elif network == 'dense_3':
        from .networks.dense_3 import Dense_3 as N
        net = N()
    elif network == 'dense_4':
        from .networks.dense_4 import Dense_4 as N
        net = N()
    elif network == 'dense_5':
        from .networks.dense_5 import Dense_5 as N
        net = N()
    elif network == 'dense_6':
        from .networks.dense_6 import Dense_6 as N
        net = N(n_feats=32)
    elif network == 'dense_3_awms':
        from .networks.dense_3_awms import Dense_3_awms as N
        net = N()
    elif network == 'awsrn_sd_ed_c_carafe':
        from .networks.awsrn_sd_ed_c_carafe import AWSRN_SD_ED_C_CARAFE as N
        net = N()
    elif network == 'dense_2_awms':
        from .networks.dense_2_awms import Dense_2_AWMS as N
        net = N()
    elif network == 'dense_6_awms':
        from .networks.dense_6_awms import Dense_6_AWMS as N
        net = N()
    elif network == 'dense_6_ram':
        from .networks.dense_6_ram import Dense_6_RAM as N
        net = N()
    elif network == 'dense_2_ram':
        from .networks.dense_2_ram import Dense_2_RAM as N
        net = N()
    elif network == 'dense_6_ca':
        from .networks.dense_6_ca import Dense_6_CA as N
        net = N(n_feats=32, reduction=4)
    elif network == 'dense_4_ram':
        from .networks.dense_4_ram import Dense_4_RAM as N
        net = N()
    elif network == 'awsrn_sd_ed_aw_ram':
        from .networks.awsrn_sd_ed_aw_ram import AWSRN_SD_ED_AW_RAM as N
        net = N()
    elif network == 'dense_6_c':
        from .networks.dense_6_c import Dense_6_C as N
        net = N()
    elif network == 'dense_7':
        from .networks.dense_7 import Dense_7 as N
        net = N()
    elif network == 'dense_8':
        from .networks.dense_8 import Dense_8 as N
        net = N()
    elif network == 'dense_9':
        from .networks.dense_9 import Dense_9 as N
        net = N()
    elif network == 'dense_6_ca_c':
        from .networks.dense_6_ca_c import Dense_6_CA_C as N
        net = N()
    elif network == 'dense_6_ca_c_sp':
        from .networks.dense_6_ca_c_sp import Dense_6_CA_C_SP as N
        net = N()
    elif network == 'dense_6_ca_c_uc':
        from .networks.dense_6_ca_c_uc import Dense_6_CA_C_UC as N
        net = N()
    elif network == 'dense_6_ca_c_dr':
        from .networks.dense_6_ca_c_dr import Dense_6_CA_C_DR as N
        net = N()
    elif network == 'dense_6_ca_c_sp_dr':
        from .networks.dense_6_ca_c_sp_dr import Dense_6_CA_C_SP_DR as N
        net = N()
    elif network == 'dense_6_ca_sp':
        from .networks.dense_6_ca_sp import Dense_6_CA_SP as N
        net = N()
    elif network == 'dense_6_ca_uc':
        from .networks.dense_6_ca_uc import Dense_6_CA_UC as N
        net = N()
    elif network == 'dense_6_ca_c_sp_dr_64':
        from .networks.dense_6_ca_c_sp_dr_64 import Dense_6_CA_C_SP_DR_64 as N
        net = N()
    elif network == 'dense_6_ca_2':
        from .networks.dense_6_ca_2 import Dense_6_CA_2 as N
        net = N()
    elif network == 'dense_6_ca_c_uc_dr':
        from .networks.dense_6_ca_c_uc_dr import Dense_6_CA_C_UC_DR as N
        net = N()
    elif network == 'dense_6_ram_mobile':
        from .networks.dense_6_ram_mobile import Dense_6_RAM_Mobile as N
        net = N()
    elif network == 'dense_6_ca_ca':
        from .networks.dense_6_ca_ca import Dense_6_CA_CA as N
        net = N()
    elif network == 'dense_6_ca_basic':
        from .networks.dense_6_ca_basic import Dense_6_CA_Basic as N
        net = N()
    elif network == 'dense_6_ca_c_uc_dr_bi':
        from .networks.dense_6_ca_c_uc_dr_bi import Dense_6_CA_C_UC_DR_BI as N
        net = N()
    elif network == 'dense_6_full':
        from .networks.dense_6_full import Dense_6_full as N
        net = N(n_feats=16)
    elif network == 'dense_10_ca':
        from .networks.dense_10_ca import Dense_10_CA as N
        net = N()
    elif network == 'dense_10_ca_2':
        from .networks.dense_10_ca_2 import Dense_10_CA_2 as N
        net = N()
    elif network == 'dense_6_ca_3':
        from .networks.dense_6_ca_3 import Dense_6_CA_3 as N
        net = N()
    elif network == 'dense_6_ca_relu':
        from .networks.dense_6_ca_relu import Dense_6_CA_relu as N
        net = N()
    elif network == 'dense_6_ca_lrelu':
        from .networks.dense_6_ca_lrelu import Dense_6_CA_lrelu as N
        net = N()
    elif network == 'dense_6_ca_4':
        from .networks.dense_6_ca_4 import Dense_6_CA_4 as N
        net = N()
    elif network == 'dense_11_ca':
        from .networks.dense_11_ca import Dense_11_CA as N
        net = N()
    elif network == 'dense_11_ca_2':
        from .networks.dense_11_ca_2 import Dense_11_CA_2 as N
        net = N()
    elif network == 'dense_6_ca_5':
        from .networks.dense_6_ca_5 import Dense_6_CA_5 as N
        net = N()
    elif network == 'dense_6_ca_6':
        from .networks.dense_6_ca_6 import Dense_6_CA_6 as N
        net = N()
    elif network == 'dense_6_ca_noaw':
        from .networks.dense_6_ca_noaw import Dense_6_CA_noAW as N
        net = N()
    elif network == 'dense_6_ca_ca_2':
        from .networks.dense_6_ca_ca_2 import Dense_6_CA_CA_2 as N
        net = N()
    elif network == 'dense_6_ca_ca_3':
        from .networks.dense_6_ca_ca_3 import Dense_6_CA_CA_3 as N
        net = N()
    elif network == 'dense_6_ca_ca_4':
        from .networks.dense_6_ca_ca_4 import Dense_6_CA_CA_4 as N
        net = N()
    elif network == 'dense_6_ca_ca_5':
        from .networks.dense_6_ca_ca_5 import Dense_6_CA_CA_5 as N
        net = N()
    elif network == 'dense_6_ca_ca_5_2':
        from .networks.dense_6_ca_ca_5_2 import Dense_6_CA_CA_5_2 as N
        net = N()
    elif network == 'dense_6_ca_5x5':
        from .networks.dense_6_ca_5x5 import Dense_6_CA_5x5 as N
        net = N()
    elif network == 'dense_6_ca_ca_5_new':
        from .networks.dense_6_ca_ca_5_new import Dense_6_CA_CA_5_new as N
        net = N()
    elif network == 'dense_6_ca_new':
        from .networks.dense_6_ca_new import Dense_6_CA_new as N
        net = N()
    elif network == 'dense_12_ca_ca_5':
        from .networks.dense_12_ca_ca_5 import Net as N
        net = N()
    elif network == 'multistream_1':
        from .networks.multistream_1 import Net as N
        net = N()
    elif network == 'multistream_2':
        from .networks.multistream_2 import Net as N
        net = N()
    elif network == 'multistream_3':
        from .networks.multistream_3 import Net as N
        net = N()
    elif network == 'multistream_4':
        from .networks.multistream_4 import Net as N
        net = N()
    elif network == 'multistream_5':
        from .networks.multistream_5 import Net as N
        net = N()
    elif network == 'multistream_6':
        from .networks.multistream_6 import Net as N
        net = N()
    elif network == 'multistream_7':
        from .networks.multistream_7 import Net as N
        net = N()
    elif network == 'multistream_8':
        from .networks.multistream_8 import Net as N
        net = N()
    elif network == 'multistream_9':
        from .networks.multistream_9 import Net as N
        net = N()
    elif network == 'ahdrnet_official':
        from .networks.ahdrnet_official import AHDR as N
        net = N()
    elif network == 'multistream_6_01':
        from .networks.multistream_6_01 import Net as N
        net = N()
    elif network == 'expandnet':
        from .networks.expandnet import ExpandNet as N
        net = N()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(network))

    print('Network [{:s}] is created.'.format(net.__class__.__name__))
    if conf.is_train:
        init_weights(net, init_type='kaiming', scale=0.1)
    if conf.gpu_ids and not conf.use_cpu:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)

    return net


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)