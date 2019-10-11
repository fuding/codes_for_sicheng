from thop import profile
import torch

from models.networks.deephdr import DeepHDR
from models.networks.carn import CARN
from models.networks.carn_m import CARN_M
from models.networks.ahdrnet import AHDRNet
from models.networks.awsrn import AWSRN_SD
from models.networks.awsrn_ia import AWSRN_SD_IA
from models.networks.mcan_s import MCAN_S
from models.networks.awsrn_sd_ed import AWSRN_SD_ED
from models.networks.awsrn_sd_ed_aw import AWSRN_SD_ED_AW
from models.networks.awsrn_sd_ed_c import AWSRN_SD_ED_C
from models.networks.awsrn_sd_ed_aw_ia import AWSRN_SD_ED_AW_IA
from models.networks.awsrn_ed_aw_mn import AWSRN_ED_AW_MN
from models.networks.awsrn_ed_aw_mn_ca import AWSRN_ED_AW_MN_CA
from models.networks.awsrn_ed_aw_mn_scse import AWSRN_ED_AW_MN_scSE
from models.networks.awsrn_ed_aw_mn_scse_2 import AWSRN_ED_AW_MN_scSE_2
from models.networks.awsrn_ed_aw_mn_cbam import AWSRN_ED_AW_MN_CBAM
from models.networks.awsrn_ed_aw_mn_ram import AWSRN_ED_AW_MN_RAM
from models.networks.mcan_s_ed import MCAN_S_ED
from models.networks.dense_1 import Dense_1
from models.networks.awsrn_sd_ed_aw_aspp import AWSRN_SD_ED_AW_ASPP
from models.networks.awsrn_ed_aw_mn_nobn import AWSRN_ED_AW_MN_noBN
from models.networks.dense_2 import Dense_2
from models.networks.dense_3 import Dense_3
from models.networks.dense_4 import Dense_4
from models.networks.dense_5 import Dense_5
from models.networks.dense_6 import Dense_6
from models.networks.dense_3_awms import Dense_3_awms
from models.networks.awsrn_sd_ed_c_carafe import AWSRN_SD_ED_C_CARAFE
from models.networks.dense_2_awms import Dense_2_AWMS
from models.networks.dense_7 import  Dense_7
from models.networks.dense_8 import  Dense_8
from models.networks.dense_9 import  Dense_9
from models.networks.dense_6_c import Dense_6_C
from models.networks.dense_6_ca import Dense_6_CA
from models.networks.dense_6_ca_c_uc import Dense_6_CA_C_UC
from models.networks.dense_6_ca_c_sp import Dense_6_CA_C_SP
from models.networks.dense_6_ca_c import Dense_6_CA_C
from models.networks.dense_6_ca_c_dr import  Dense_6_CA_C_DR
from models.networks.dense_6_ca_c_sp_dr import Dense_6_CA_C_SP_DR
from models.networks.dense_6_ca_sp import Dense_6_CA_SP
from models.networks.dense_6_ca_c_sp_dr_64 import Dense_6_CA_C_SP_DR_64
from models.networks.dense_6_full import Dense_6_full
from models.networks.dense_6_ca_c_uc_dr import Dense_6_CA_C_UC_DR
from models.networks.dense_6_ca_uc import Dense_6_CA_UC
from models.networks.dense_6_ca_ca import Dense_6_CA_CA
from models.networks.dense_6_ca_c_uc_dr_bi import Dense_6_CA_C_UC_DR_BI
from models.networks.dense_10_ca import Dense_10_CA
from models.networks.dense_10_ca_2 import Dense_10_CA_2
from models.networks.dense_11_ca import Dense_11_CA
from models.networks.dense_11_ca_2 import Dense_11_CA_2
from models.networks.dense_6_ca_ca_5 import Dense_6_CA_CA_5
from models.networks.dense_6_ca_5 import Dense_6_CA_5
from models.networks.dense_6_ca_ca_5_2 import Dense_6_CA_CA_5_2

# model = DeepHDR()  # 63.259885568 16.601731
# model = CARN() # 79.373008896 0.972803
# model = CARN_M() # 36.67918848 0.376259
# model = AHDRNet()  # 96.561266688 1.286851
# model = AWSRN_SD() # 29.226958848 0.407049
# model = AWSRN_SD_IA() # 45.341474816 0.628489
# model = MCAN_S() # 19.228786688 0.249065
# model = AWSRN_SD_ED() # 5.72932096 0.952841
# model = AWSRN_SD_ED_AW()
# model = AWSRN_SD_ED_C() # 9.430286336 1.081865
# model = AWSRN_SD_ED_AW_IA() # 13.785595904 2.059535
# model = AWSRN_ED_AW_MN(n_feats=32, num_blocks=4) # 6.015483904 1.176213
# model = AWSRN_ED_AW_MN_CA(n_feats=32, num_blocks=4) # 6.019219456 1.471125
# model = AWSRN_ED_AW_MN_scSE(n_feats=32, num_blocks=4) # 6.022365184 1.474197
# model = AWSRN_ED_AW_MN_scSE_2(n_feats=32, num_blocks=4) # 6.022365184 1.474197
# model = AWSRN_ED_AW_MN_CBAM(n_feats=32, num_blocks=4) # 6.020534272 1.472693
# model = AWSRN_ED_AW_MN_RAM(n_feats=32, num_blocks=4) # 6.047531008 1.498741
# model = MCAN_S_ED() # 5.560008704 0.783353
# model = AWSRN_ED_AW_MN_CA(reduction=8) # 5.911896064 1.190451
# model = AWSRN_ED_AW_MN(n_feats=16) # 5.492736 0.743781
# model = Dense_1(n_feats=32) # 5.760024576 0.922089
# model = AWSRN_SD_ED_AW_ASPP() # 5.69434112 0.919627
# model = Dense_2(n_feats=32) # 5.79371008 0.938601
# model = Dense_3(n_feats=32) # 5.858820096 0.970441
# model = Dense_4(n_feats=32) # 6.044975104 1.190645
# model = Dense_5(n_feats=32) # 5.79371008 0.938601
# model = Dense_6(n_feats=32) # 5.88398592 0.982729
# model = Dense_3_awms(n_feats=32) # 6.034980864 1.138637
# model = AWSRN_SD_ED_C_CARAFE() # 5.4367232 1.272545
# model = Dense_2_AWMS() # 5.969870848 1.106797
# model = Dense_7() # 5.818875904 0.950889
# model = Dense_8()
# model = Dense_9() # 5.88398592 0.982729
# model = Dense_6(n_feats=16) # 5.460017152 0.695433
# model = Dense_6_C() # 9.660465152 1.130185
# model = Dense_6_CA(n_feats=16) #5.461737472 0.769161
# model = Dense_6_CA(n_feats=32) # 5.887721472 1.277641
# model = Dense_6_CA_C_UC() # 10.495721472 1.425091
# model = Dense_6_CA_C_SP() # 10.495721472 2.310595
# model = Dense_6_CA_C() # 9.664200704 1.425097
# model = Dense_6_CA_C_DR() # 5.838995456 1.302345
# model = Dense_6_CA_C_SP_DR() # 6.200754176 1.745475
# model = Dense_6_CA_SP() # 6.301384704 1.720777
# model = Dense_6_CA_C_SP_DR_64() # 6.574178304 2.077251
# model = Dense_6_full(n_feats=16)  # 15.91752704 0.278659
# model = Dense_6_full(n_feats=32)  # 40.265908224 0.796355
# model = Dense_6_CA_C_UC_DR() # 6.200754176 1.302339
# model = Dense_6_CA_UC() # 6.301384704 1.277641
# model = Dense_6_CA_CA() # 5.89106176 1.342601
# model =  Dense_6_CA_C_UC_DR_BI()
# model = Dense_10_CA() # 5.924593664 1.355561
# model = Dense_10_CA_2() # 5.93514496 1.360745
# model = Dense_11_CA() # 6.179565568 1.721993
# model = Dense_11_CA_2() # 6.19853824 1.731305
# model = Dense_6_CA_CA_5() # 5.890997248 1.309897
# model = Dense_6_CA_5() # 6.076465152 1.351369
# model = Dense_6_CA_CA_5_2() # 6.07981568 1.388233

# from models.networks.dense_6_ca_5x5 import Dense_6_CA_5x5
# model = Dense_6_CA_5x5() # 6.192992256 1.659401

# from models.networks.dense_12_ca_ca_5 import Net
# from models.networks.multistream_1 import Net
# from models.networks.multistream_2 import Net
# from models.networks.multistream_3 import Net
# from models.networks.multistream_4 import Net
from models.networks.multistream_5 import Net

model = Net()

x1 = torch.randn(1, 3, 256, 256)
x2 = torch.randn(1, 3, 256, 256)
x3 = torch.randn(1, 3, 256, 256)
x4 = torch.randn(1, 3, 256, 256)
x5 = torch.randn(1, 3, 256, 256)
x6 = torch.randn(1, 3, 256, 256)


flops, params = profile(model, inputs=([x1,x2,x3], [x4,x5,x6]))

print(flops/1000000000, params/1000000)






