import torch
import numpy as np
import sys
import os
import nibabel as nib
from covid_classes import *
from skimage.exposure import rescale_intensity
from dipy.align.reslice import reslice
from scipy.ndimage import binary_closing

def segment_lungs(nifty_image):
    """
    :param nifty_image: nibabe image containing the CT volume
    :return: mask: segmented lungs as nibabel image  binary mask 0 background 1 lungs
    """

    # GPU specifications
    batch_size = 16
    cuda_device = 'cuda:0'
    gpu = 0
    device = torch.device(cuda_device)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


    # Define and Load Pre-Trained Network
    current_dir = os.getcwd()
    jigsaw = os.path.join(current_dir, 'model_epoch_10.tar')
    model_path = jigsaw
    net = UNet_Seg(model_location=model_path, mode='inference')
    net = net.to(device)
    inference_instance = SegInference(model=net, batch_size=batch_size, device=device)

    with torch.no_grad():
        orig_affine = nifty_image.affine
        orig_shape = nifty_image.shape
        zooms = nifty_image.header.get_zooms()[:3]

        # Define new resolution and resample volume
        new_zooms = (1.0, 1.0, 1.0)
        vol_data = nifty_image.get_fdata()
        resampled_data, resampled_affine = reslice(vol_data, orig_affine, zooms, new_zooms)

        # Tranpose to get volume into the right shape. Dim[0] = z, Dim[1] = x, Dim[2] = y
        # Axial slices [z, :, :] should be oriented such that the spine is in the bottom center of the
        # axial slice and the liver is in the upper left of the axial slice.
        resampled_data = np.transpose(resampled_data, (2, 1, 0))
        d, h, w = resampled_data.shape
        resampled_data = resampled_data[:, ::-1, :]



        # Rescale the intensities of the volume and normalize prior to inference.
        resampled_data = rescale_intensity(resampled_data, out_range=(0.0, 1.0))
        resampled_data = (resampled_data - np.mean(resampled_data)) / (np.std(resampled_data) + sys.float_info.epsilon)

        # Run Inference
        y_pred = inference_instance.infer(resampled_data)

        # Resample prediction to original orientation and resolution.
        y_pred = y_pred[:, ::-1, :]
        y_pred = np.transpose(y_pred, (2, 1, 0))
        y_pred = resample(y_pred, nshape=orig_shape, mode='constant', order=0)



        ### postprocessing
        ## keep only the lungs

        y1 = np.zeros(np.shape(y_pred))
        # y2 = np.zeros(np.shape(y_pred))

        # index= np.where(y_pred)==8
        y1[np.where(y_pred == 8)] = 1
        y1[np.where(y_pred == 9)] = 1

        ## binary closing
        kernel= np.ones((7,7,7), dtype=int)

        y1= binary_closing(y1.astype(int), kernel)

        ## keep largest two components
        # y1= np.ones(np.shape(seg_data))
        #
        # y1[np.where(seg_data!=1)]=0
        y1 = remove_small_components(y1)

    # Save prediction as nifti image
    mask = nib.Nifti1Image(y1.astype(np.float32), orig_affine)
    return mask




################################################################################################################

# path to the nibabel image
niftyfile = '/media/HD/datasets/covid19/EXACT/train_nifty/CASE01.nii.gz'
data = nib.load(niftyfile)
mask = segment_lungs(data)
#nib.save(mask, 'test_pred_closing.nii.gz')



# # GPU specifications
# batch_size=16
# cuda_device = 'cuda:4'
# gpu = 4
# device = torch.device(cuda_device)
#
# # Define and Load Pre-Trained Network
# current_dir = os.getcwd()
# jigsaw = os.path.join(current_dir, 'model_epoch_10.tar')
# model_path = jigsaw
# net = UNet_Seg(model_location=model_path, mode='inference')
# net = net.to(device)
# print('Model', model_path)
# print('Save path', out_dir)
#
# with torch.no_grad():
#
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
#
#     inference_instance = SegInference(model=net, batch_size=batch_size, device=device)
#
#     for i, filename in enumerate(data_files):
#         # if not i % 2:
#         vol_ID = filename.split('/')[-1].split('.')[0]
#         print('\rData: {}/{}'.format(i + 1, len(data_files)), end="")
#         sys.stdout.flush()
#
#         # Load volume and get affine and original shape, resolution
#         ç
#         orig_affine = vol.affine
#         orig_shape = vol.shape
#         zooms = vol.header.get_zooms()[:3]
#
#         # Define new resolution and resample volume
#         new_zooms = (1.0, 1.0, 1.0)
#         vol_data = vol.get_fdata()
#         resampled_data, resampled_affine = reslice(vol_data, orig_affine, zooms, new_zooms)
#
#         # Tranpose to get volume into the right shape. Dim[0] = z, Dim[1] = x, Dim[2] = y
#         # Axial slices [z, :, :] should be oriented such that the spine is in the bottom center of the
#         # axial slice and the liver is in the upper left of the axial slice.
#         resampled_data = np.transpose(resampled_data, (2, 1, 0))
#         d, h, w = resampled_data.shape
#         resampled_data = resampled_data[:, ::-1, :]
#
#         # Imshow and store figure for debugging.
#         # plt.imshow(resampled_data[d // 4, :, :], cmap='gray')
#         # plt.savefig('/home/navarrof/data/covid19/EXACT_predictions/slice_orientation.png')
#         # plt.close()
#
#         # Rescale the intensities of the volume and normalize prior to inference.
#         resampled_data = rescale_intensity(resampled_data, out_range=(0.0, 1.0))
#         resampled_data = (resampled_data - np.mean(resampled_data)) / (np.std(resampled_data) + sys.float_info.epsilon)
#
#         # Run Inference
#         y_pred = inference_instance.infer(resampled_data)
#
#         # Resample prediction to original orientation and resolution.
#         y_pred = y_pred[:,::-1,:]
#         y_pred = np.transpose(y_pred, (2, 1, 0))
#         y_pred = resample(y_pred, nshape=orig_shape, mode='constant', order=0)
#
#         # Save prediction as nifti image
#         prediction = nib.Nifti1Image(y_pred.astype(np.float32), orig_affine)
#         nib.save(prediction, os.path.join(out_dir, '{}_pred.nii.gz'.format(vol_ID)))