# coding=utf-8
################################################
# 数据增广，包括
# 2018.09.02 add
################################################
import numpy as np
import os
import cv2
import copy


# 输出图片格式jpg
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)
    print('success write')


class DataAugment:
    def __init__(self, debug=False):
        self.debug = debug
        print("Data augment...")

    def basic_matrix(self, translation):
        """基础变换矩阵"""
        return np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    def adjust_transform_for_image(self, img, trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix = copy.deepcopy(trans_matrix)
        height, width, channels = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot(
            [self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform(self, img, transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                                borderValue=0, )  # cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT
        return output

    def apply(self, img, trans_matrix):
        """应用变换"""
        tmp_matrix = self.adjust_transform_for_image(img, trans_matrix)
        out_img = self.apply_transform(img, tmp_matrix)
        # if self.debug:
        #     self.show(out_img)
        return out_img

    def random_vector(self, min, max):
        """生成范围矩阵"""
        min = np.array(min)
        max = np.array(max)
        # print(min.shape, max.shape)
        assert min.shape == max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def show(self, img):
        """可视化"""
        cv2.imshow("outimg", img)
        cv2.waitKey()

    def random_transform(self, img, min_translation, max_translation):
        """平移变换"""
        factor = self.random_vector(min_translation, max_translation)
        trans_matrix = np.array([[1, 0, factor[0]], [0, 1, factor[1]], [0, 0, 1]])
        out_img = self.apply(img, trans_matrix)
        return trans_matrix, out_img

    def random_flip(self, img, factor):
        """水平或垂直翻转"""
        flip_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, flip_matrix)
        return flip_matrix, out_img

    def random_rotate(self, img, factor):
        """随机旋转"""
        angle = np.random.uniform(factor[0], factor[1])
        # print("angle:{}".format(angle))
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        out_img = self.apply(img, rotate_matrix)
        return rotate_matrix, out_img

    def random_scale(self, img, min_translation, max_translation):
        """随机缩放"""
        factor = self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, scale_matrix)
        return scale_matrix, out_img

    def random_shear(self, img, factor):
        """随机剪切，包括横向和众向剪切"""
        angle = np.random.uniform(factor[0], factor[1])
        # print("fc:{}".format(angle))
        crop_matrix = np.array([[1, factor[0], 0], [factor[1], 1, 0], [0, 0, 1]])
        out_img = self.apply(img, crop_matrix)
        return crop_matrix, out_img


if __name__ == "__main__":
    demo = DataAugment(debug=True)
    path = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/image_MouShi'
    f_list = []
    out_path = '/Users/sunjunjiao/Documents/Beam_pumping_unit_data/image_augmentation_MS/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for root, dirs, fs in os.walk(path):
        for f in fs:
            f_list.append(os.path.join(root, f))
    print(len(f_list))
    print(f_list)
    f_list.remove('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/image_MouShi/.DS_Store')
    print(len(f_list))
    print(f_list)
    for f in f_list:
        if str.find(f, '.py') != -1:
            print('.py file,skip!')
            continue
        img = cv2.imread(f)
        image_num = 0
        f = f.replace('/Users/sunjunjiao/Documents/Beam_pumping_unit_data/image_MouShi/', '')
        save_dir = os.path.join(out_path, f.replace('.jpg', ''))
        # 平移测试
        _, outimg = demo.random_transform(img, (0.1, 0.1), (0.2, 0.2))  # (-0.3,-0.3),(0.3,0.3)
        image_num += 1
        save_image(outimg, save_dir + '_', image_num)

        # 垂直变换测试
        # _, outimg = demo.random_flip(img, (1.0, -1.0))
        # image_num += 1
        # save_image(outimg, save_dir + '_', image_num)


        # 水平变换测试
        _, outimg = demo.random_flip(img, (-1.0, 1.0))
        image_num += 1
        save_image(outimg, save_dir + '_', image_num)

        # 旋转变换测试
        _, outimg = demo.random_rotate(img, (0.5, 0.8))
        image_num += 1
        save_image(outimg, save_dir + '_', image_num)

        # # 缩放变换测试
        _, outimg = demo.random_scale(img, (1.2, 1.2), (1.3, 1.3))
        image_num += 1
        save_image(outimg, save_dir + '_', image_num)

        # 随机裁剪测试
        _, outimg = demo.random_shear(img, (0.2, 0.3))
        image_num += 1
        save_image(outimg, save_dir + '_', image_num)

        # 组合变换
        t1, _ = demo.random_transform(img, (-0.3, -0.3), (0.3, 0.3))
        image_num += 1
        save_image(_, save_dir + '_', image_num)

        t2, _ = demo.random_rotate(img, (0.5, 0.8))
        image_num += 1
        save_image(_, save_dir + '_', image_num)

        t3, _ = demo.random_scale(img, (1.5, 1.5), (1.7, 1.7))
        image_num += 1
        save_image(_, save_dir + '_', image_num)

        tmp = np.linalg.multi_dot([t1, t2, t3])
        # print("tmp:{}".format(tmp))
        out = demo.apply(img, tmp)
        image_num += 1
        save_image(out, save_dir + '_', image_num)
