import os
class Config():
    def __init__(self):
        self.mode="dynamic" # MODES: "maintenance" | "dynamic"
        self.data_dir = ""
        self.save_path = os.environ.get("HAND_DT_DATA_ROOT", "/nas/data_EMG/data_DT/")
        self.fs=2048
        self.segmentation_length=.25 # in second
        self.overlap_ratio=.5 # in percentage
        self.discard_length=.25 # in second
        self.os_dir= os.getcwd()
        self.train_dir_name={
            "resample171_data_half": "PR_v1_resample171_half_unslice",
            "smooth_rms_half": "PR_v1_smooth_half",
            "unslice_features_half": "PR_v1_features_half_unslice_repair",
            "stft_half": "PR_v1_stft_half",
            "StftFeature": "PR_v1_StftFeature",
        }
        self.train_dir={
            # "origin_data_half":"/nas/data_EMG/data_DT/PR_v1_original_half_unslice/", # 原始数据, 去前.5, feature_dim=1
            # "resample512_data_half":"/nas/data_EMG/data_DT/PR_v1_resample512_half_unslice/", # 原始数据，去前.5, 降采样为1/2, features=range(64*4), feature_dim=1
            "resample171_data_half": os.path.join(self.save_path, self.train_dir_name["resample171_data_half"]), # 原始数据，去前.5, 降采样为1/6(向上取整), features=range(171), feature_dim=1
            "smooth_rms_half": os.path.join(self.save_path, self.train_dir_name["smooth_rms_half"]), # smooth with rms, slice, 去前.5, features=range(240), feature_dim=1
            # "slice_features":'/nas/data_EMG/data_DT/PR_v1_features_slice/', # 60特征, 滑窗 1-> 5, feature_dim=5
            # "unslice_features":'/nas/data_EMG/data_DT/PR_v1_features_unslice/', # 60特征, 未滑窗, 未去前.25, feature_dim=1
            "unslice_features_half": os.path.join(self.save_path, self.train_dir_name["unslice_features_half"]), # 60特征, 未滑窗, 去前.5, 坏道修复, feature_dim=1
            "stft_half": os.path.join(self.save_path, self.train_dir_name["stft_half"]), # stft, win=256, overlap=0, dowsample=4, 去前.5, feature_dim=1
            # "stft_triple_half":'/nas/data_EMG/data_DT/PR_v1_stft_triple_half',
            # "stft_half_half":'/nas/data_EMG/data_DT/PR_v1_stft_half_half/', # stft, win=256, overlap=0, dowsample=4, 取一半频谱（正）, 去前.5, feature_dim=1
            # "resample_stft_half":'/nas/data_EMG/data_DT/PR_v1_stft_resample_half/', # stft, win=256, overlap=50%, dowsample=4, 去前.5, features=range(64*4), feature_dim=1
            "StftFeature": os.path.join(self.save_path, self.train_dir_name["StftFeature"]),
        }
        self.num_class=10
        self.num_sub=20
        self.adjust_map=[ 63,  62,  61,  60,  59,  58,  57,  56, 191, 190, 189, 188, 187, 186, 185, 184, 
                          55,  54,  53,  52,  51,  50,  49,  48, 183, 182, 181, 180, 179, 178, 177, 176,
                          47,  46,  45,  44,  43,  42,  41,  40, 175, 174, 173, 172, 171, 170, 169, 168,
                          39,  38,  37,  36,  35,  34,  33,  32, 167, 166, 165, 164, 163, 162, 161, 160,
                          31,  30,  29,  28,  27,  26,  25,  24, 159, 158, 157, 156, 155, 154, 153, 152,
                          23,  22,  21,  20,  19,  18,  17,  16, 151, 150, 149, 148, 147, 146, 145, 144,
                          15,  14,  13,  12,  11,  10,   9,   8, 143, 142, 141, 140, 139, 138, 137, 136,
                           7,   6,   5,   4,   3,   2,   1,   0, 135, 134, 133, 132, 131, 130, 129, 128,
                         127, 126, 125, 124, 123, 122, 121, 120, 255, 254, 253, 252, 251, 250, 249, 248, 
                         119, 118, 117, 116, 115, 114, 113, 112, 247, 246, 245, 244, 243, 242, 241, 240, 
                         111, 110, 109, 108, 107, 106, 105, 104, 239, 238, 237, 236, 235, 234, 233, 232, 
                         103, 102, 101, 100,  99,  98,  97,  96, 231, 230, 229, 228, 227, 226, 225, 224, 
                          95,  94,  93,  92,  91,  90,  89,  88, 223, 222, 221, 220, 219, 218, 217, 216,
                          87,  86,  85,  84,  83,  82,  81,  80, 215, 214, 213, 212, 211, 210, 209, 208,  
                          79,  78,  77,  76,  75,  74,  73,  72, 207, 206, 205, 204, 203, 202, 201, 200,  
                          71,  70,  69,  68,  67,  66,  65,  64, 199, 198, 197, 196, 195, 194, 193, 192]
     
