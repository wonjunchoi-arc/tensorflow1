#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__() # super의__init__()에 있는 클래스 변수들을 가지고 올 수 있습니다. 그렇지 않으면 현재 class의 init으로 덮어써버려짐
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        # self.transform = transforms.Compose([
        #         transforms.ToTensor(),   \
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))]) #위에 to tensor를 정규화 0~1사이라고 생가갛고 노말라이즈가 -1~1범위로 표준화하여 아웃라이어를 제거해주는 것
        
        # load data list
        im_names = []
        c_names = []
        # print('경로',opt.dataroot)
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                #print(line): 019595_0.jpg 019595_1.jpg
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name))
        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'warp-mask', c_name))
     
        c = self.transform(c)  # [-1,1]
        # cm = self.transform(cm)  # [-1,1]  torch.Size([1, 256, 192])

        # print(c.shape) #torch.Size([3, 256, 192])
        cm_array = np.array(cm)
        # print(cm_array.shape) #(256, 192)
        cm_array = (cm_array >= 128).astype(np.float32)
        # print(cm_array.shape) #(256, 192)
        '''
        x = np.array([[ -1 , 1 ],[ 3 , -2 ]])
        y = x > 0 (x array에서 0보다 큰 것!)
        y = array([ False , True ], [ True , False ])


        [False False False ... False False False]
        '''

        cm = torch.from_numpy(cm_array) # [0,1]
        #print(cm.shape) #torch.Size([256, 192])
        #2)torch.from_numpy()  tensor로 변환할 때, 원래 메모리를 상속받는다. (=as_tensor()) 즉 원래의 넘파이를 텐서로 변환시키는것
        cm.unsqueeze_(0) # 언스퀴즈(Unsqueeze) - 특정 위치에(0차원에 ) 1인 차원을 추가
        # print(cm.shape) #torch.Size([1, 256, 192])



        # person image 
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im) # [-1,1]
        # print(im.size())
        
        
        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        # print(type(im_parse))

        parse_array = np.array(im_parse)
        # print(parse_array.shape) #(256, 192)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
       
        '''
        해당 조건을 만족하는 값들은 true로 1로 반환 
        나머지는 False 0로 반환
        '''


        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8)) #넘파이 배열을 이미지로 변환
        # parse_shape.show()흰색 검은색으로 이미지 테두리 출력
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        # print(parse_shape.size)(12, 16)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        # print(parse_shape.size) #(192, 256)
        shape = self.transform(parse_shape) # [-1,1] torch.Size([1, 256, 192])
        # print(shape.shape)
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts 나머지부분은 1이되고 채워져있는 부분은 0을 더해주네
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts 얼굴만 나온다 둥둥


        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            # print(pose_label)
            '''
            {'version': 1.0, 'people': [{'face_keypoints': [], 
            'pose_keypoints': [79.1181102362205, 40.7272727272727, 0.907790213823318, 87.6850393700787, 91.2290909090909, 0.654397651553154, 49.3858267716535, 93.5563636363636, 0.493277914822102, 45.8582677165354, 153.6, 0.713929876685143, 29.9842519685039, 178.501818181818, 0.802610471844673, 127.748031496063, 87.5054545454545, 0.526449844241142, 142.614173228346, 159.650909090909, 0.792855083942413, 119.181102362205, 201.076363636364, 0.495145117864013, 42.8346456692913, 214.341818181818, 0.244836983649293, 0, 0, 0, 0, 0, 0, 92.7244094488189, 222.254545454545, 0.226115419587586, 0, 0, 0, 0, 0, 0, 70.0472440944882, 31.1854545454545, 0.96220988035202, 89.9527559055118, 31.8836363636364, 0.945377916097641, 58.7086614173228, 36.7709090909091, 0.656627222895622, 103.307086614173, 40.0290909090909, 0.842052668333054], 
            'hand_right_keypoints': [], 
            'hand_left_keypoints': []}]}
            '''
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            # print(pose_data.shape) #(54,)
            pose_data = pose_data.reshape((-1,3))
            # print(pose_data.shape) #(18, 3)

        point_num = pose_data.shape[0] #18
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height)) #L (8-bit pixels, black and white)

        pose_draw = ImageDraw.Draw(im_pose)
        
        # pose_draw.show()
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            # im_pose.show()
            one_map = self.transform(one_map)
            # print(one_map.shape) #torch.Size([1, 256, 192])
            pose_map[i] = one_map[0]
        # print(pose_map.shape) #torch.Size([18, 256, 192])

        # im_pose.show()
        
        # just for visualization
        im_pose = self.transform(im_pose)
        
        # cloth-agnostic representation
        agnostic = torch.cat([shape, im_h, pose_map], 0) 

        if self.stage == 'GMM':
            im_g = Image.open('sergeywong/grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "../viton_resize")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    # from IPython import embed; embed()

