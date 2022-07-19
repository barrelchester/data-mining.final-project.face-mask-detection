import torch


class Config():
    def __init__(self):
        self.ims_path1 = '../data/real/WWMR-DB - Part 1'
        self.ims_path2 = '../data/real/WWMR-DB - Part 2'
        self.labs_path = '../data/real/WWMR-DB - Labels/Labels/YOLO'
        self.tensor_path = '../data/real'

        self.im_size = 128

        self.lab2idx_subj = {'0001': 0, '0002': 1, '0003': 2, '0004': 3, '0005': 4, '0006': 5, '0007': 6,
                             '0008': 7, '0009': 8, '0010': 9, '0011': 10, '0012': 11, '0013': 12, '0014': 13,
                             '0015': 14, '0016': 15, '0017': 16, '0018': 17, '0019': 18, '0020': 19, '0021': 20,
                             '0022': 21, '0023': 22, '0024': 23, '0025': 24, '0026': 25, '0027': 26, '0028': 27,
                             '0029': 28, '0030': 29, '0031': 30, '0032': 31, '0033': 32, '0034': 33, '0035': 34,
                             '0036': 35, '0037': 36, '0038': 37, '0039': 38, '0040': 39, '0041': 40, '0042': 41}
        self.lab2idx_mask_stat = {'MRCW': 0, 'MRFH': 1, 'MRHN': 2, 'MRNC': 3, 'MRNN': 4, 'MRNW': 5, 'MRTN': 6, 'MSFC': 7}
        self.lab2idx_mask_type = {'DRNV': 0, 'DRWV': 1, 'NMDM': 2, 'NONE': 3, 'SRGM': 4}
        self.lab2idx_angle = {'0000_NA': 0, '0045_B': 1, '0045_F': 2, '0045_NA': 3, '0090_B': 4, '0090_F': 5, '0090_NA': 6}

        self.idx2lab_subj = {v:k for k,v in self.lab2idx_subj.items()}
        self.idx2lab_mask_stat = {v:k for k,v in self.lab2idx_mask_stat.items()} 
        self.idx2lab_mask_type = {v:k for k,v in self.lab2idx_mask_type.items()} 
        self.idx2lab_angle = {v:k for k,v in self.lab2idx_angle.items()}

        self.mask_stat_orig = {'MRCW':'Mask Or Respirator Correctly Worn',
                          'MRFH':'Mask Or Respirator On The Forehead', 
                          'MRHN':'Mask Or Respirator Hanging From An Ear', 
                          'MRNC':'Mask Or Respirator Under The Chin', 
                          'MRNN':'Mask Or Respirator Under The Nose', 
                          'MRNW':'Mask Or Respirator Not Worn', 
                          'MRTN':'Mask Or Respirator On The Tip Of The Nose', 
                          'MSFC':'Mask Folded Above The Chin'}

        self.mask_type_orig = {'DRNV':'Disposable Respirator Without Valve',
                          'DRWV':'Disposable Respirator With Valve',
                          'NMDM':'Non-Medical Mask', 
                          'NONE':'No Mask', 
                          'SRGM':'Surgical Mask'}

        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406])
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225])
    
    