import torch
import torch.nn as nn
import numpy as np
import math
import os
from scipy import ndimage as nd
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops, label

class UNet_Seg(nn.Module):
    """
    Class for UNet meant for segmentation.
    """

    def __init__(self,  base = 32, num_classes=21, model_location=None, mode='baseline', pre_task=None):
        super(UNet_Seg, self).__init__()
        print('Generating Segmentation 32 UNet')
        base= base
        self.mode = mode
        self.pretask = pre_task

        self.encoder_block1 = make_layers([base, base]) #256x256 16 16
        base = base * 2
        self.encoder_block2 = make_layers(['M', base, base], c_in=base//2) #128x128 32 32
        base = base * 2 # 32
        self.encoder_block3 = make_layers(['M',base, base], c_in=base//2) #64x64
        base = base * 2 #64
        self.encoder_block4 = make_layers(['M',base, base], c_in=base//2) #32x32 128 128
        base = base * 2 #128
        self.encoder_block5 = make_layers(['M', base, base, 'U'], c_in=base//2, upsample_in=base, upsample_out=base//2) #16x16 512 512
        base= base//2
        self.decoder_block1 = make_layers([base,base,'U'], c_in=base*2, upsample_in=base, upsample_out=base//2) #64x64
        base = base // 2
        self.decoder_block2 = make_layers([base,base,'U'], c_in=base*2, upsample_in=base, upsample_out=base//2)
        base = base // 2
        self.decoder_block3 = make_layers([base, base, 'U'], c_in=base * 2, upsample_in=base, upsample_out=base // 2)
        base = base // 2
        self.decoder_block4 = make_layers([base,base], c_in=base*2)
        self.classifier = nn.Conv2d(base, num_classes, kernel_size=1, stride=1)

        if model_location != None:
            self.transferWeights(model_location)

    def forward(self, x):
        B, C, H, W = x.size()

        #Encode
        E1 = self.encoder_block1(x)
        E2 = self.encoder_block2(E1)
        E3 = self.encoder_block3(E2)
        E4 = self.encoder_block4(E3)
        x = self.encoder_block5(E4)

        #Decode
        if E4.shape[-3:] != x.shape[-3:]:
            x = self.equalize_dimensions(x, E4)
        x = torch.cat((x, E4), dim=1)
        x = self.decoder_block1(x)

        if E3.shape[-3:] != x.shape[-3:]:
            x = self.equalize_dimensions(x, E3)
        x = torch.cat((x, E3), dim=1)
        x = self.decoder_block2(x)

        if E2.shape[-3:] != x.shape[-3:]:
            x = self.equalize_dimensions(x, E2)
        x = torch.cat((x, E2), dim=1)
        x = self.decoder_block3(x)

        if E1.shape[-3:] != x.shape[-3:]:
            x = self.equalize_dimensions(x, E1)
        x = torch.cat((x, E1), dim=1)
        x = self.decoder_block4(x)

        x=self.classifier(x)

        return x

    def transferWeights(self, checkpoint):
        print('Transferring weights from {}'.format(checkpoint))
        SS_net = torch.load(checkpoint, map_location='cpu')
        pretrained_keys = list(SS_net['model_state_dict'])

        model_dict = self.state_dict()
        model_keys = pretrained_keys

        for i, key in enumerate(pretrained_keys):
            pretrained_value = SS_net['model_state_dict'][key]
            model_dict[model_keys[i]] = pretrained_value

        self.load_state_dict(model_dict)

    def equalize_dimensions(self, x, upsampled_data):
        padding_dim1 = abs(x.shape[-1] - upsampled_data.shape[-1])
        padding_dim2 = abs(x.shape[-2] - upsampled_data.shape[-2])
        padding_dim3 = abs(x.shape[-3] - upsampled_data.shape[-3])

        if padding_dim1 % 2 == 0:
            padding_left = padding_dim1 // 2
            padding_right = padding_dim1 // 2
        else:
            padding_left = padding_dim1 // 2
            padding_right = padding_dim1 - padding_left

        if padding_dim2 % 2 == 0:
            padding_top = padding_dim2 // 2
            padding_bottom = padding_dim2 // 2
        else:
            padding_top = padding_dim2 // 2
            padding_bottom = padding_dim2 - padding_top

        if padding_dim3 % 2 == 0:
            padding_front = padding_dim3 // 2
            padding_back = padding_dim3 // 2
        else:
            padding_front = padding_dim3 // 2
            padding_back = padding_dim3 - padding_front

        pad_fn = nn.ConstantPad3d((padding_left, padding_right,
                                   padding_top, padding_bottom,
                                   padding_front, padding_back), 0)

        return pad_fn(x)

class SegInference(object):
    """
    Class for running inference with a segmentation network.
    :param model: the unet instance to infer with
    :param batch_size: size of training batch
    :param device: GPU device to use for inference
    """

    def __init__(self, model, batch_size=10, device='cuda:5'):
        self.model = model
        self.batch_size= batch_size
        self.device = device
        self.model.eval()

    def infer(self, vol_data):
        """
        :param vol_data: a numpy array with dimensions in the following order [depth, height, width]
        :return: volume with predictions of 21 organs
        """

        d, h, w = np.shape(vol_data)

        iterations = math.ceil(d / self.batch_size)
        prediction = np.zeros(shape=vol_data.shape)

        for j in range(iterations):
            if j < iterations-1:  # if not the last iteration
                vol_slice = vol_data[j * self.batch_size:(j + 1) * self.batch_size, :, :]
            else:
                vol_slice = vol_data[j * self.batch_size:, :, :]

            with torch.no_grad():
                vol_slice = self.ToTensor(vol_slice).type(torch.FloatTensor)
                vol_slice.requires_grad = False
                pred_slice = self.model(vol_slice.cuda(self.device))
                pred_slice = torch.argmax(pred_slice, dim=1)

            if j < iterations-1:
                prediction[j * self.batch_size:(j + 1) * self.batch_size, :, :] = pred_slice.cpu().numpy()
            else:
                prediction[j * self.batch_size:, :, :] = pred_slice.cpu().numpy()

        ##############################################################################################

        return prediction

    def ToTensor(self, image):
        """
        :param image: 3D numpy.ndarray of stacked axial slices: [# Axial Slices (batch), Height, Width]
        :return: Tensor of axial slices in the form [Batch, Channel, Height, Width]
        """
        image = np.expand_dims(image, axis=-1)
        image = image.transpose((0,3, 1, 2))
        return torch.from_numpy(image)

def resample(img, nshape=None, spacing=None, new_spacing=None, order=3, mode='constant'):
    """
        Change image resolution by resampling

        Inputs:
        - nshape (numpy.ndarray): desired shape of resampled image. Leave None if using spacing arguments
        - spacing (numpy.ndarray): current resolution
        - new_spacing (numpy.ndarray): new resolution
        - order (int: 0-5): interpolation order

        Outputs:
        - resampled image (numpy.ndarray)
        """
    if nshape is None:
        if spacing.shape[0]!=1:
            spacing = np.transpose(spacing)

        if new_spacing.shape[0]!=1:
            new_spacing = np.transpose(new_spacing)

        if np.array_equal(spacing, new_spacing):
            return img

        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / img.shape

    else:
        if img.shape == nshape:
            return img
        real_resize_factor = np.array(nshape, dtype=float) / np.array(img.shape, dtype=float)

    image = nd.interpolation.zoom(img, real_resize_factor.ravel(), order=order, mode=mode)

    return image

def make_layers(cfg, batch_norm=True, c_in=1, upsample_in=None, upsample_out=None):
    """
    :param cfg: a list giving the configuration of the layer to make; ['M', 'U', ...] for example
    :param batch_norm: boolean whether to include batch_norm after every convolution
    :param c_in: number of channels expected at the beginning of the network
    :param upsample_in: if upsampling, the number of incoming channels
    :param upsample_out: if upsampling, the number of output channels
    :return: Sequentially defined layer following configuration cfg.
    """
    layers = []
    in_channels = c_in
    for v in cfg:

        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.ConvTranspose2d(upsample_in, upsample_out, kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

def list_files(directory, extension=''):
    """
    :param directory: the location in which to list the files inside
    :param extension: the extension of files to search for. '' by default returns all files inside 'directory'
    :return: A list of files including 'extension' in their file path within the location 'directory'
    """
    results = []

    for f in os.listdir(directory):
        if extension in f:
            results.append(os.path.join(directory, f))

    return results


def remove_small_components(mask):

    ## create connected components
    labels = label(mask.astype(int), connectivity=3)
    areas = [r.area for r in regionprops(labels)]
    areas.sort(reverse=True)
    ## keep the biggest  component
    print('areas:',len(areas))
    if len(areas) >=2:
        maxArea = areas[1] - 1
        mask = remove_small_objects(labels, maxArea)

    mask = mask.astype(bool).astype(float)
    return mask

