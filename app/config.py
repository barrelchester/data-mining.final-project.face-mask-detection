import torch, pickle, os


class Config():
    def __init__(self):
        self.im_size = 128
        
        self.wwmr_path = '../data/real'
        self.wwmr_ims_path1 = '%s/WWMR-DB - Part 1' % self.wwmr_path
        self.wwmr_ims_path2 = '%s/WWMR-DB - Part 2' % self.wwmr_path
        self.wwmr_labs_path = '%s/WWMR-DB - Labels/Labels/YOLO' % self.wwmr_path

        self.fmd_path = '../data/face_mask_detection'
        self.fmd_ims_path = '%s/images' % self.fmd_path
        self.fmd_annots_path = '%s/annotations' % self.fmd_path

        self.mfn_path = '../data/masked_face_net'
        self.mfn_ims_path = '%s/images' % self.mfn_path

        self.gan_path = '../data/gan_faces'
        self.gan_ims_path = '%s/images' % self.gan_path
        
        self.combined_data_path = '../data/combined_3class'

        self.lab2idx = {'without_mask':0, 'with_mask':1, 'mask_weared_incorrect':2}
        if not os.path.exists('%s/lab2idx.pkl' % self.combined_data_path):
            with open('%s/lab2idx.pkl' % self.combined_data_path, 'wb') as o:
                pickle.dump(self.lab2idx, o)
                
        self.idx2lab = {v:k for k,v in self.lab2idx.items()}

        self.mask_stat_details = {'MRCW':'Mask Or Respirator Correctly Worn',
                          'MRFH':'Mask Or Respirator On The Forehead', 
                          'MRHN':'Mask Or Respirator Hanging From An Ear', 
                          'MRNC':'Mask Or Respirator Under The Chin', 
                          'MRNN':'Mask Or Respirator Under The Nose', 
                          'MRNW':'Mask Or Respirator Not Worn', 
                          'MRTN':'Mask Or Respirator On The Tip Of The Nose', 
                          'MSFC':'Mask Folded Above The Chin'}

        #self.mask_type_orig = {'DRNV':'Disposable Respirator Without Valve',
        #                  'DRWV':'Disposable Respirator With Valve',
        #                  'NMDM':'Non-Medical Mask', 
        #                  'NONE':'No Mask', 
        #                  'SRGM':'Surgical Mask'}

        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406])
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225])
    
    