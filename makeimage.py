import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
import torchvision.utils as utils
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from imageio import imwrite
from testdataset import *
from datetime import datetime
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="NTIRE DeBlur Challenge")
parser.add_argument("--num_of_layers", type=int, default=16, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='val_blur', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Build model
    print('Loading model ...\n')
    net = EACD(channels=3)
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model = net.cuda()
    load = torch.load(os.path.join(opt.logdir, 'EACD_348_0.0328.pth'))
    model.load_state_dict(load)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    dataset_val = Dataset(train=False).data
    files_source = glob.glob(os.path.join('data/val_submit', opt.test_data, '*.jpg'))
    files_source.sort()

    print('Model Test Start!')

    for f in files_source:
        # image
        blur = Image.open(f)
        # blur = cv2.imread(f)
        blur = np.transpose(blur, (2, 0, 1))
        blur = normalize(np.float32(blur))
        blur = np.expand_dims(blur, 0)
        blurimg = torch.Tensor(blur)

        with torch.no_grad():  # this can save much memory
            blurimg = Variable(blurimg.cuda())
            out, edge = model(blurimg)
            Out = torch.clamp(out, 0., 1.)

        # Tensor to Image.
        blur_img = utils.make_grid(blurimg.data, nrow=8, normalize=True, scale_each=True)
        deblur_img = utils.make_grid(Out.data, nrow=8, normalize=True, scale_each=True)

        result_img = torch.clamp(deblur_img * 255, 0, 255)
        result_img = np.uint8(result_img.cpu())
        name = f.split('/')[-1]
        name = name.split('.')[0]
        name = name + '.png'
        imwrite('./data/val_out/' + name , np.transpose(result_img, (1,2,0)))
        print(name + ' saved!!')

if __name__ == "__main__":
    main()
