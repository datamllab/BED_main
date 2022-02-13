import cv2
import numpy as np

# img_data = cv2.imread("../../../YOLO_V1_GPU/VOC2007/Train/JPEGImages/000012.jpg")
# img_data = cv2.resize(img_data, (224, 224), interpolation=cv2.INTER_AREA)
# img_data = img_data.astype(np.int) - 128
# img_data = img_data.transpose(2,1,0)
#
# # img_data = img_data[:, ::16, ::16]
# # img_data = img_data[:, 0:32, 0:32]
# # img_data = img_data[:, 0:64, 0:64]
# # img_data = img_data[:, 0:256, 0:256]
# # img_data = img_data[:, 0:128, 0:128]
# # img_data = img_data[:, ::8, ::8]
# # img_data = img_data[:, ::2, ::2]
# # img_data = img_data[:, ::4, ::4]
# print(img_data.shape)
# # print(img_data[0:5, 0:5, :])
# print(type(img_data))


# img_data = np.load("../../ai8x-synthesis/tests/sample_yolov1.npy")
# print(img_data)
#
# r,g,b = img_data[0,2,0], img_data[1,2,0], img_data[2,2,0]
# r_bin, g_bin, b_bin = bin(2**8+(r)), bin(2**8+(g)), bin(2**8+(b))
#
# print(r_bin, g_bin, b_bin) # index0 0x00c5c5c3, index3 0x00c5c5c3


pixel_value = "101112" # b 12, g 11, r 10 # -1 FF # FF => -1: int(FF, 16) = 255, 255 - 256 = -1

r_value = int(pixel_value[0:2], 16)
g_value = int(pixel_value[2:4], 16)
b_value = int(pixel_value[4:6], 16)

r_value_dec = r_value if r_value < 128 else r_value - 256
g_value_dec = g_value if g_value < 128 else g_value - 256
b_value_dec = b_value if b_value < 128 else b_value - 256

# r_value_dec -= 128
# g_value_dec -= 128
# b_value_dec -= 128

print(r_value_dec, g_value_dec, b_value_dec)
bgr_array = np.concatenate((b_value_dec * np.ones((1, 224,224)),
                            g_value_dec * np.ones((1, 224,224)),
                            r_value_dec * np.ones((1, 224,224))), axis=0).astype(int)

img_data = bgr_array
print(bgr_array.shape)


print(img_data.min(), img_data.max())
np.save("sample_yolov1.npy", img_data)
#
# sample_mnist = np.load("sample_faceid.npy")
# print(sample_mnist.min(), sample_mnist.max())
# print(sample_mnist.shape)
