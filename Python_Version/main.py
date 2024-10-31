import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile

from skimage import measure
from roifile import ImagejRoi
import sys
import cv2 as cv
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import label
from skimage.draw import polygon2mask

def unet_weight_map(y, wc=None, w0=10, sigma=25):
    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.
    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(y == 255)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = bwdist(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        w = w0 * np.exp(-1 * ((d1 + d2) ** 2 / (2 * sigma))) * no_labels

        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
    else:
        w = np.zeros_like(y)

    return w


def make_paths(num_classes1,results_path1):
    parent_dir = results_path1
    for counter in range(1,num_classes1+1):
        dir1 = 'class_' +  str(counter) + '_annotations'
        parent_path = os.path.join(parent_dir,dir1)
        if not os.path.exists(parent_path):
            os.mkdir(parent_path)

        dir2 = '/label_masks'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/mask_binary'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/mask_binary_without_border'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/mask_binary_without_border_erode'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/distance_maps'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/weighted_maps'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/weighted_maps_erode'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)


        dir2 = '/overlay'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/label_masks_modify'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/stacked_mask'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir2 = '/nuclei_border'
        if not os.path.exists(parent_path + dir2):
            os.mkdir(parent_path + dir2)

        dir3 = '/disk1'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk2'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)


        dir3 = '/disk3'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk4'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk5'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk6'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk7'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk8'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk9'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)

        dir3 = '/disk10'
        if not os.path.exists(parent_path + dir2 + dir3):
            os.mkdir(parent_path + dir2 + dir3)








def masks_generator(size_target, imagej_zips_path, raw_imgs_path, results_path, num_classes,debug):

    # Use a breakpoint in the code line below to debug your script.
    imagej_zips = []
    raw_imgs = []
    for file in os.listdir(imagej_zips_path):
        if file.endswith(".zip"):
            imagej_zips.append(os.path.join(imagej_zips_path, file))
    imagej_zips.sort()
    for file in os.listdir(raw_imgs_path):
        if file.endswith(".png"):
            raw_imgs.append(file)
    raw_imgs.sort()
    if debug:
        print(imagej_zips, raw_imgs)
    data = np.zeros((len(imagej_zips)+1, num_classes+1))
    make_paths(num_classes, results_path)
    object_count_tot = np.zeros((num_classes + 1, len(imagej_zips)))
    pth = os.getcwd()
    temp_path = pth + '\\tempfolder\\'
    color_code = [16711935,65535, 4294967295]
    color_opt = ['g','r','w']

    for counter in range(len(imagej_zips)):
        s = imagej_zips[counter]

        if not (os.path.exists(temp_path)):
            os.mkdir(temp_path)

        with ZipFile(s, 'r') as zip_ref:
            zip_ref.extractall(temp_path)


        ROIs = []

        for file in os.listdir(temp_path):
            if file.endswith(".roi"):
                ROIs.append(os.path.join(temp_path, file))

        object_count_tot[0, counter] = len(ROIs)

        strokeColors = np.zeros((len(ROIs)))
        for n_num in range(len(ROIs)):
            ROIName = ROIs[n_num]
            sROI = ImagejRoi.fromfile(ROIName)
            color = int.from_bytes(sROI.stroke_color, byteorder=sys.byteorder)
            strokeColors[n_num] = color

        uniqueStrokeColors = np.unique(strokeColors)

        if counter == 0:
            uniqueStrokeColors1 = uniqueStrokeColors
        else:
            if not np.array_equal(uniqueStrokeColors1,uniqueStrokeColors):
                for i in range(len(uniqueStrokeColors)):
                    temp1 = np.array(np.where((uniqueStrokeColors1 == uniqueStrokeColors[i])))
                    if temp1.size == 0:
                        uniqueStrokeColors_t = np.zeros((np.shape(uniqueStrokeColors1)[0]+1))
                        uniqueStrokeColors_t[0:-1] = uniqueStrokeColors1
                        uniqueStrokeColors_t[-1] = uniqueStrokeColors[i]
                        uniqueStrokeColors1 = uniqueStrokeColors_t

                uniqueStrokeColors = uniqueStrokeColors1

        for ii in range(len(uniqueStrokeColors)):
            nuc_class_count = 0
            mask_overlap = np.zeros((size_target, size_target), dtype=np.uint16)
            mask_overlap_modify = np.zeros((size_target, size_target),dtype=np.uint16)
            mask_overlap_borderremove = np.zeros((size_target, size_target),dtype=np.uint16)
            D_overlap = np.zeros((size_target, size_target))
            for n_num in range(len(ROIs)):
                ROIName = ROIs[n_num]
                sROI = ImagejRoi.fromfile(ROIName)
                color = int.from_bytes(sROI.stroke_color, byteorder=sys.byteorder)
                if color == uniqueStrokeColors[ii]:
                    cool = np.array([color,color,color])
                    c_ind = np.where(cool == color_code)[0]
                    nuc_class_count = nuc_class_count+1
                    # mask = np.zeros((size_target,size_target),dtype=np.uint16)
                    # mask = cv.fillConvexPoly(mask, np.array(sROI.coordinates(),dtype=np.int32),1)

                    polygon_numpy = np.array(sROI.coordinates(),dtype=np.int32)
                    polygon_numpy1 = polygon_numpy.copy()
                    polygon_numpy1[:, 0] = polygon_numpy[:, 1]
                    polygon_numpy1[:, 1] = polygon_numpy[:, 0]
                    polygon_numpy1 = np.squeeze(polygon_numpy1)

                    mask = polygon2mask((size_target, size_target), polygon_numpy1)
                    mask = np.array(mask, dtype=np.uint8)
                    if debug:
                        plt.imshow(mask)
                        plt.show()
                    mask_org = mask
                    D = bwdist(mask)
                    mask = np.double(mask) * (n_num + 1)
                    mask_overlap_modify = np.maximum(mask, mask_overlap_modify)
                    mask_overlap = mask + mask_overlap
                    D_overlap = np.maximum(D, D_overlap)
                    a = 10
                    if debug:
                        plt.imshow(D_overlap)
                        plt.show()
                    t1 = (mask_overlap_borderremove == mask_org) & (mask_overlap_borderremove == 1)
                    eshterak = np.where(t1)
                    eshterak_img = np.zeros((size_target,size_target))
                    eshterak_img[eshterak] = 1
                    mask_overlap_borderremove = np.maximum(mask_org, mask_overlap_borderremove)
                    if len(eshterak[0])!=0:
                        ksize = 3
                        gX = cv.Sobel(mask_org, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
                        gY = cv.Sobel(mask_org, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
                        gX = cv.convertScaleAbs(gX)
                        gY = cv.convertScaleAbs(gY)
                        B = cv.addWeighted(gX, 0.5, gY, 0.5, 0)
                        B[B > 0] = 1
                        B[B <= 0] = 0
                        if debug:
                            plt.imshow(B)
                            plt.show()
                        ksize = 1
                        gX = cv.Sobel(mask_org, ddepth=cv.CV_64F, dx=1, dy=0, ksize=ksize)
                        gY = cv.Sobel(mask_org, ddepth=cv.CV_64F, dx=0, dy=1, ksize=ksize)
                        gX = cv.convertScaleAbs(gX)
                        gY = cv.convertScaleAbs(gY)
                        thin_edge = cv.addWeighted(gX, 0.5, gY, 0.5, 0)
                        thin_edge[thin_edge > 0] = 1
                        thin_edge[thin_edge <= 0] = 0

                        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
                        B = np.array(B,dtype='uint16') & np.array(eshterak_img,dtype='uint16')
                        B2 = cv.dilate(B, se,iterations = 1)

                        mask_overlap_borderremove = mask_overlap_borderremove - B2
                        mask_overlap_borderremove = mask_overlap_borderremove - thin_edge
                        mask_overlap_borderremove[mask_overlap_borderremove == -1] = 0

                        a = 10
            object_count_tot[int(c_ind) + 1, counter] = nuc_class_count
            #write mask_overlap
            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'label_masks' + '/' + raw_imgs[counter][:-3]+'png'
            cv.imwrite(im_name, np.array(mask_overlap,dtype=np.uint16))

            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'label_masks_modify' + '/' + raw_imgs[counter]
            cv.imwrite(im_name, np.array(mask_overlap_modify,dtype=np.uint16))


            mask_binary = np.zeros((512, 512))
            mask_binary[mask_overlap > 0] = 255
            mask_binary = np.uint8(mask_binary)

            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'mask_binary' + '/' + raw_imgs[counter]
            cv.imwrite(im_name, mask_binary)



            for diskthick in range(3,13):
                se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (diskthick, diskthick))
                eroded_mask_binary = cv.erode(mask_binary, se, iterations = 1)
                border = mask_binary - eroded_mask_binary

                im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'nuclei_border/disk' + str(diskthick - 2) +'/' + raw_imgs[counter]
                cv.imwrite(im_name, border)


            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'mask_binary_without_border' + '/' + raw_imgs[counter]
            mask_overlap_borderremove[mask_overlap_borderremove > 0] = 255

            cv.imwrite(im_name, np.uint8(mask_overlap_borderremove))



            gt = mask_overlap_borderremove

            se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

            gt_erode = cv.erode(gt, se, iterations=1)



            weight = unet_weight_map(gt)
            weight_erode = unet_weight_map(gt_erode)
            weight = weight * 255 / np.max(weight)
            weight_erode = weight_erode * 255 / np.max(weight_erode)

            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'weighted_maps' + '/' + raw_imgs[counter]
            cv.imwrite(im_name, weight)

            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'weighted_maps_erode' + '/' + raw_imgs[counter]
            cv.imwrite(im_name, weight_erode)

            se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

            mask_binary_without_border_erode = cv.erode(mask_overlap_borderremove, se, iterations=1)



            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'mask_binary_without_border_erode' + '/' + raw_imgs[counter]
            cv.imwrite(im_name, mask_binary_without_border_erode)



            D_overlap = np.double(D_overlap)
            D_overlap1 = np.copy(D_overlap)
            D_overlap1 = D_overlap1*65535/np.max(D_overlap1)
#            D_overlap = D_overlap / np.max(D_overlap)


            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'distance_maps' + '/' + raw_imgs[counter]
            cv.imwrite(im_name, np.array(D_overlap,dtype=np.uint16))

            original = cv.imread(os.path.join(raw_imgs_path,raw_imgs[counter]))
            original = cv.resize(original, (size_target,size_target))
            img_dim = np.shape(original)
            if len(img_dim) == 2:
                original_temp = np.zeros((size_target,size_target,3))
                original_temp[:,:,0] = original
                original_temp[:,:,1] = original
                original_temp[:,:,2] = original
                original = original_temp



            original_r = np.copy(original[:,:, 0])
            original_g = np.copy(original[:,:, 1])
            original_b = np.copy(original[:,:, 2])

            original_r[mask_overlap != 0] = 255
            original_g[mask_overlap != 0] = 255
            original_b[mask_overlap != 0] = 255

            original2 = np.copy(original)
            original2[:,:, 0] = original_r
            original2[:,:, 1] = original_g
            original2[:,:, 2] = original_b

            fig= plt.figure()  # create figure & 1 axis
            fig.set_size_inches(18.5, 10.5)
            ax1 = fig.add_subplot(121)
            ax1.set_title('cropped_image')
            ax1.imshow(np.uint16(original))

            ax2 = fig.add_subplot(122)
            ax2.set_title('overlay')
            ax2.imshow(np.uint16(original))






            for i in range(1,len(ROIs)+1):

                dum = np.uint16(mask_overlap)
                dum[mask_overlap != i] = 0
                contours = measure.find_contours(dum, level=0.5)

                for contour in contours:
                    ax2.plot(contour[:, 1], contour[:, 0], color=color_opt[int(c_ind)], linewidth=1, linestyle='-')

            im_name = results_path + 'class_' + str(int(c_ind) + 1) + '_annotations/' + 'overlay' + '/' + raw_imgs[counter]

            plt.show()

            fig.savefig(im_name, dpi = 500)  # save the figure to file
            plt.close(fig)
            mask_overlap = np.zeros((size_target, size_target))
            mask_overlap_modify = np.zeros((size_target, size_target))
            mask_overlap_borderremove = np.zeros((size_target, size_target))
            D_overlap = np.zeros((size_target,size_target))

        print(raw_imgs[counter], 'has totally', np.uint8(object_count_tot[0,counter]), 'nuclei')
        data[counter,:] = [int(object_count_tot[0, counter]), int(object_count_tot[1, counter]),
                                   int(object_count_tot[2, counter]), int(object_count_tot[3, counter])]

        for file in os.listdir(temp_path):
            if file.endswith(".roi"):
                os.remove(temp_path + file)


    print('all images have total of', np.sum(object_count_tot[0,:]), 'nuclei.')
    for i in range(1,np.shape(object_count_tot)[0]):
        print('All images have ', np.sum(object_count_tot[i,:]), 'nuclei of class ', i)

    sum_value1 = np.sum(data[:, 0])
    sum_value2 = np.sum(data[:, 1])
    sum_value3 = np.sum(data[:, 2])
    sum_value4 = np.sum(data[:, 3])

    data[-1,:] = [sum_value1,sum_value2,sum_value3,sum_value4]

    raw_imgs.append('total')
    df = pd.DataFrame(data)
    df.insert(0, column='name', value = raw_imgs)
    df.to_csv(results_path + 'stats1.csv',header= ['name', 'total_nuclei', 'class1_nuclei', 'class2_nuclei', 'class3_nuclei'], index = False)
    print(df)



    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    size_target = 512
    imagej_zips_path = '.../imageJ_zips/'
    raw_imgs_path = '.../images/DAPI/'
    results_path = '.../data0/'
    num_classes = 3

    masks_generator(size_target, imagej_zips_path, raw_imgs_path, results_path, num_classes,0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
