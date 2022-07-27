import os, random, pickle, json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt



class ImageProcessor():
    def __init__(self, config):
        self.config = config
        self.mask_stats = self.config.lab2idx_mask_stat.keys()
        self.mask_types = self.config.lab2idx_mask_type.keys()
        self.head_angles = set(['0000', '0045', '0090'])
        
        
    def process_images(self):
        if os.path.exists('%s/x_%d.pt' % (self.config.tensor_path, self.config.im_size)):
            print('Images already processed, delete %s/x_%d.pt to reprocess' % (self.config.tensor_path, self.config.im_size))
            return 
        
        if os.path.exists('%s/im_data.json' % self.config.tensor_path):
            with open('%s/im_data.json' % self.config.tensor_path, 'r') as f:
                im_data = json.load(f)
        else:
            print('Loading image data...')
            im_data = self.__get_image_data(self.config.ims_path1)
            im_data.update(self.__get_image_data(self.config.ims_path2))

            print('Getting bounding box info...')
            self.__get_bounding_boxes(im_data)

            print('Writing image data to %s/im_data.json' % self.config.tensor_path)
            with open('%s/im_data.json' % self.config.tensor_path, 'w') as o:
                json.dump(im_data, o)
            
        print('Cropping and resizing and storing tensors...')
        self.__crop_and_resize(im_data)
        
        print('Done!')
        
        
    def __get_image_data(self, ims_path):
        '''Get a dictionary of image sizes mapped to list of image paths'''
        im_data = {}

        i=0
        # image name encodes the following (if present): subjnum_maskstat_masktype_headangle_maskinbackgroundforeground.jpg
        for path, _, fns in os.walk(ims_path):
            for fn in fns:
                if fn.lower().endswith('.jpg'):
                    im_name = fn[:-4]
                elif fn.lower().endswith('.jpeg'):
                    im_name = fn[:-5]
                else:
                    continue

                items = im_name.split('_')

                if len(items)<3 or len(items)>5:
                    print(items)
                    return {}

                subj = items[0]
                items = items[1:]
                mask_stat = ''
                head_angle = ''
                mask_type = 'NONE'
                back_fore = 'NA'

                for item in items:
                    if item in self.mask_stats:
                        mask_stat = item
                    elif item in self.mask_types:
                        mask_type = item
                    elif item in self.head_angles:
                        head_angle = item
                    elif item=='B' or item=='F':
                        back_fore = item

                if not mask_stat or not head_angle:
                    print('missing mask stat or head angle', im_name)
                    continue

                fp = '%s/%s' % (path, fn)

                #read the image with opencv2
                try:
                    im = cv2.imread(fp)
                except Exception as ex:
                    print('couldnt open %s: %s' % (fp, str(ex)))
                    continue

                if im is None:
                    print('couldnt open %s' % (fp))
                    continue

                #get and store the image size
                sz = im.shape

                im_data[im_name] = {'path':fp, 
                                      'size':sz,
                                      'subject':subj, 
                                      'mask_status':mask_stat, 
                                      'mask_type':mask_type, 
                                      'head_angle':head_angle, 
                                      'mask_background_foreground':back_fore}

                i+=1
                if i%100==0:
                    print('\n', i, im_data[im_name])

        print('%s files processed' % i)

        return im_data
    
    
    def __get_bounding_boxes(self, im_data):
        for path, _, fns in os.walk(self.config.labs_path):
            for fn in fns:
                if not fn.endswith('.txt'):
                    continue
                if fn.startswith('classes'):
                    continue

                im_name = fn[:-4]

                if not im_name in im_data:
                    print('%s not in im sizes' % im_name)
                    continue

                fp = '%s/%s' % (path, fn)

                x, y, w, h = 0,0,0,0
                with open(fp, 'r') as f:
                    for line in f:
                        if not line.startswith('0'):
                            continue

                        _, x, y, w, h = line.replace('\n','').split(' ')
                        break

                if not w:
                    print('full bounding box not found for %s' % fn)
                    continue

                im_data[im_name]['bb'] = {'x':x, 'y':y, 'w':w, 'h':h}
                    
                    
    def __crop_and_resize(self, im_data):
        im_arrays = []
        im_names = []
        y_binary = []
        y_subj = []
        y_mask_stat = []
        y_mask_type = []
        y_angle = []

        i=0
        for im_name, im_dict in im_data.items():
            i+=1
            if i%100==0:
                print(i)
                
            im = cv2.cvtColor(cv2.imread(im_dict['path']), cv2.COLOR_BGR2RGB)
            sz = im.shape

            h = int(sz[0]*float(im_dict['bb']['h']))
            w = int(sz[1]*float(im_dict['bb']['w']))

            y1 = int(sz[0]*float(im_dict['bb']['y'])) - h//2
            x1 = int(sz[1]*float(im_dict['bb']['x'])) - w//2
            x2 = x1 + w
            y2 = y1 + h

            crop_w = x2-x1
            crop_h = y2-y1

            if crop_w < crop_h:
                pad = (crop_h - crop_w)//2
                x1 = max(0, x1-pad)
                x2 = min(x2+pad, sz[1]-1)
            elif crop_w > crop_h:
                pad = (crop_w - crop_h)//2
                y1 = max(0, y1-pad)
                y2 = min(y2+pad, sz[0]-1)

            im_crop = im[y1:y2, x1:x2, :]

            #resize the image
            im_new = cv2.resize(im_crop, (self.config.im_size, self.config.im_size), interpolation = cv2.INTER_LINEAR)

            im_arrays.append(im_new)

            im_names.append(im_name)

            y_binary.append(1 if im_dict['mask_status']=='MRCW' else 0)
            y_subj.append(self.config.lab2idx_subj[im_dict['subject']])
            y_mask_stat.append(self.config.lab2idx_mask_stat[im_dict['mask_status']])
            y_mask_type.append(self.config.lab2idx_mask_type[im_dict['mask_type']])

            ang = '%s_%s' % (im_dict['head_angle'], im_dict['mask_background_foreground'])
            y_angle.append(self.config.lab2idx_angle[ang])

        #shape (3 x H x W), where H and W are expected to be at least 224.
        # The images have to be loaded in to a range of [0, 1] and then normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        im_arrays = np.reshape(im_arrays, (-1, self.config.im_size, self.config.im_size, 3))

        im_arrays = torch.from_numpy(im_arrays)
        #normalize
        im_arrays = ((im_arrays/255) - self.config.imnet_mean) / self.config.imnet_std
        #need to swap the RGB dim from last to second index
        im_arrays = torch.swapaxes(im_arrays, 3, 1)
        print(im_arrays.shape)

        print('saving %s/x_%d.pt' % (self.config.tensor_path, self.config.im_size))
        torch.save(im_arrays, '%s/x_%d.pt' % (self.config.tensor_path, self.config.im_size))
        
        if not os.path.exists('%s/y_binary.pt' % (self.config.tensor_path)):
            torch.save(torch.LongTensor(y_binary), '%s/y_binary.pt' % (self.config.tensor_path))
            torch.save(torch.LongTensor(y_subj), '%s/y_subj.pt' % (self.config.tensor_path))
            torch.save(torch.LongTensor(y_mask_stat), '%s/y_mask_stat.pt' % (self.config.tensor_path))
            torch.save(torch.LongTensor(y_mask_type), '%s/y_mask_type.pt' % (self.config.tensor_path))
            torch.save(torch.LongTensor(y_angle), '%s/y_angle.pt' % (self.config.tensor_path))

            with open('%s/im_names.txt' % self.config.tensor_path, 'w') as o:
                o.write('\n'.join(im_names))
                
                
    def flip_augment_class(self, im_tensors, y, class_idx_to_flip=1):
        #get target class instances
        trg_class_idx = (y==class_idx_to_flip).nonzero(as_tuple=True)[0]
        trg_class_tensors = im_tensors[trg_class_idx]
        
        print('Flipping %d instances of class index %d' % (trg_class_tensors.size(0), class_idx_to_flip))
        
        #shape (3 x H x W), flip W, index 2
        trg_class_flipped_tensors = torch.flip(trg_class_tensors, dims=(2,))
        
        #add flipped images and targets
        im_tensors = torch.cat((im_tensors, trg_class_flipped_tensors), 0)
        y = torch.cat((y, torch.ones(trg_class_flipped_tensors.size(0)) * class_idx_to_flip))
        print('New dataset size: %s' % im_tensors.size(0))
        
        #shuffle all
        print('Shuffling...')
        idx = torch.randperm(im_tensors.size(0))
        im_tensors = im_tensors[idx]
        y = y[idx]

        return im_tensors, y
    
    
    def balance_classes(self, im_tensors, y):
        class_labs, cts = np.unique(y, return_counts=True)
        minority_ct = cts.min()
        print('Truncating all class counts to min class count %d' % minority_ct)
        
        class_x = []
        class_y = []
        for class_lab in class_labs.tolist():
            trg_class_idx = (y==class_lab).nonzero(as_tuple=True)[0][:minority_ct]
            class_x.append(im_tensors[trg_class_idx])
            class_y.append(torch.ones(trg_class_idx.size(0)) * class_lab)
            
        balanced_x = torch.cat(class_x, 0)
        balanced_y = torch.cat(class_y, 0)
        
        print('Shuffling...')
        idx = torch.randperm(balanced_x.size(0))
        balanced_x = balanced_x[idx]
        balanced_y = balanced_y[idx]
        
        print('X: %s, y: %s' % (balanced_x.size(0), balanced_y.size(0)))
        
        return balanced_x, balanced_y
            
            
    def load_data(self):
        im_arrays = torch.load('%s/x_%d.pt' % (self.config.tensor_path, self.config.im_size))

        with open('%s/im_names.txt' % self.config.tensor_path, 'r') as f:
            im_names = f.read().split('\n')

        y_binary = torch.load('%s/y_binary.pt' % (self.config.tensor_path)).type(torch.LongTensor)
        y_subj = torch.load('%s/y_subj.pt' % (self.config.tensor_path)).type(torch.LongTensor)
        y_mask_stat = torch.load('%s/y_mask_stat.pt' % (self.config.tensor_path)).type(torch.LongTensor)
        y_mask_type = torch.load('%s/y_mask_type.pt' % (self.config.tensor_path)).type(torch.LongTensor)
        y_angle = torch.load('%s/y_angle.pt' % (self.config.tensor_path)).type(torch.LongTensor)

        return im_arrays, im_names, y_binary, y_subj, y_mask_stat, y_mask_type, y_angle
            
            
    def show_image(self, im_index, im_arrays, im_names, y_binary, y_subj, y_mask_stat, y_mask_type, y_angle):
        self.show_image(im_arrays[im_index], im_names[im_index], y_binary[im_index], y_subj[im_index], 
                        y_mask_stat[im_index], y_mask_type[im_index], y_angle[im_index])
        
        
    def show_image(self, im_tensor, im_name, y_binary, y_subj, y_mask_stat, y_mask_type, y_angle):
        worn_correctly = y_binary.item()
        subject = self.config.idx2lab_subj[y_subj.item()]
        mask_status = self.config.idx2lab_mask_stat[y_mask_stat.item()]
        mask_type = self.config.idx2lab_mask_type[y_mask_type.item()]
        head_angle = self.config.idx2lab_angle[y_angle.item()]

        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255

        print('Image name: %s' % im_name)
        print('Mask worn correctly?: %s' % bool(worn_correctly))
        print('Subject: %s' % subject)
        print('Mask wearing status: %s' % self.config.mask_stat_orig[mask_status])
        print('Mask type: %s' % self.config.mask_type_orig[mask_type])
        print('Head angle and foreground/background: %s' % head_angle)

        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
        
    def show_image(self, im_tensor):
        #unswap the color channel from index 0 (needed for model) back to index 2
        im_tensor = torch.swapaxes(im_tensor, 0, 2)
        #denormalize
        im_tensor = ((im_tensor * self.config.imnet_std) + self.config.imnet_mean) * 255
        plt.imshow(im_tensor.numpy().astype(np.int32))
        plt.show()
                    
                    