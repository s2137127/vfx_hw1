import glob
import cv2
import numpy as np
import vfx_hw1
import os

def RGBtoGray(img_rgb):
    img_gray = 0.0742*img_rgb[:, :, 0]+0.7149*img_rgb[:, :, 1]+0.2109*img_rgb[:, :, 2]
    return img_gray

def GraytoBinary(img_gray):
    median = np.median(img_gray)
    # print(median)
    img_binary = np.array(img_gray)
    img_binary[np.where(img_binary < median)] = 0
    img_binary[np.where(img_binary > median)] = 255
    # print("binnnnnnnnnnnnn")
    return img_binary


def ExclusionBitmap(img_gray):
    median = np.median(img_gray)
    # print(img_gray.shape)
    img_ext = np.zeros(img_gray.shape)
    img_gray_array = np.array(img_gray)

    img_ext[np.where(img_gray_array < median-10)] = 255
    img_ext[np.where(img_gray_array > median+10)] = 255
    return img_ext


def BitmapShift(bitmap, x, y):
    translation = np.float32([[1, 0, x], [0, 1, y]])
    bitmap_shift = cv2.warpAffine(
        bitmap, translation, (bitmap.shape[1], bitmap.shape[0]))
    return bitmap_shift


def GetExpShift(img_gray1, img_gray2, shift_bits):
    cur_shift = [0, 0]
    shift_ret = [0, 0]

    if shift_bits > 0:
        img_small1 = cv2.resize(img_gray1, None, fx=0.5,
                                fy=0.5, interpolation=cv2.INTER_CUBIC)
        img_small2 = cv2.resize(img_gray2, None, fx=0.5,
                                fy=0.5, interpolation=cv2.INTER_CUBIC)
        cur_shift[0], cur_shift[1] = GetExpShift(
            img_small1, img_small2, shift_bits-1)
        cur_shift[0] = cur_shift[0] * 2
        cur_shift[1] = cur_shift[1] * 2
    else:
        cur_shift[0] = 0
        cur_shift[1] = 0

    bitmap_bin1 = GraytoBinary(img_gray1)
    # print(bitmap_bin1)
    bitmap_ext1 = ExclusionBitmap(img_gray1)
    bitmap_bin2 = GraytoBinary(img_gray2)
    bitmap_ext2 = ExclusionBitmap(img_gray2)

    min_err = img_gray1.shape[0] * img_gray1.shape[1]

    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = cur_shift[0] + i
            ys = cur_shift[1] + j

            shiftthres = BitmapShift(bitmap_bin2, xs, ys)
            shift_ext = BitmapShift(bitmap_ext2, xs, ys)

            diff = cv2.bitwise_xor(bitmap_bin1, shiftthres)
            diff = cv2.bitwise_and(diff, bitmap_ext1)
            diff = cv2.bitwise_and(diff, shift_ext)

            err = np.count_nonzero(diff)
            if err < min_err:
                shift_ret[0] = xs
                shift_ret[1] = ys
                min_err = err

    return shift_ret[0], shift_ret[1]


def AlignImages(img_list):
    # file_path_dict = {}
    # for file_path in glob.glob("./" + dir + "*.jpg"):
        # file_name = file_path.split("\\")[-1].split(".")[0]
        # curr_exposure = float(file_name.split("_")[2].split("-")[0]) / float(file_name.split("_")[2].split("-")[1])
        # file_path_dict.update({curr_exposure : file_path})

    # file_path_list = [file_path_dict[exposure] for exposure in sorted(file_path_dict.keys(), reverse=True)]
    # img_name_list =[os.path.join(vfx_hw1.args['img_dir'],img_name) for img_name in os.listdir(vfx_hw1.args['img_dir']) ]
    # img_name_list = ['img2/img%02d.jpg' %i for i in range(1,1) ]
    # img_list = [cv2.imread(i) for i in img_name_list]

    img_color_list = img_list
    img_gray_list = [RGBtoGray(curr_img) for curr_img in img_list]
    # for file_path in img_name_list:
    #     curr_img = cv2.imread(file_path)
    #     img_color_list.append(curr_img)
    #     img_gray_list.append(RGBtoGray(curr_img))

    if len(img_color_list) <= 1:
        return img_color_list
        
    shift_ret = [0, 0]
    img_shifted_list = []
    img_shifted_list.append(img_color_list[0])
    # cv2.imwrite("Images/8_aligned/shifted_" + file_path_list[0].split("/")[2].split("\\")[1], img_color_list[0])
    for i in range(1, len(img_color_list)):
        shift_ret[0], shift_ret[1] = GetExpShift(img_gray_list[0], img_gray_list[i], 4)
        translation_matrix = np.float32([[1, 0, shift_ret[0]], [0, 1, shift_ret[1]]])
        shifted_img = cv2.warpAffine(img_color_list[i], translation_matrix, (img_color_list[i].shape[1], img_color_list[i].shape[0]))
        img_shifted_list.append(shifted_img.astype(np.uint8))
        # cv2.imwrite(str(img_list[i]), shifted_img)
        cv2.imwrite('align/img%02d.jpg' % i,shifted_img)
    
    return img_shifted_list

# my_list = AlignImages("Images/8/")
# img_list = AlignImages(img_list)