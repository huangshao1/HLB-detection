import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 3  # 不包含背景
    box_thresh = 0.5
    weights_path = "./model_99.pth"
    imgs_root = r"E:\AI\mask_rcnn1\data\wogan\PCA3_Healthy"
    output = "E:/AI/mask_rcnn/output/"
    label_json_path = './coco91_indices.json'
    imgs_str = []
    for root, dirs, files in os.walk(imgs_root, topdown=False):
        for name in files:
            imgs_str.append(name)


    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # from pil image to tensor, do not normalize image
    # data_transform = transforms.Compose([transforms.ToTensor()])
    # img = data_transform(original_img)
    # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    batch_size = 8
    with torch.no_grad():
        # init
        for ids in range(0, len(img_path_list)):
            img_list = []
            for img_path in img_path_list[ids:(ids + 1)]:

                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                original_img = Image.open(img_path).convert('RGB')
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                img = torch.unsqueeze(img, dim=0)
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                # t_start = time_synchronized()
                # predictions = model(img.to(device))[0]
                # t_end = time_synchronized()
                # print("inference+NMS time: {}".format(t_end - t_start))

                iteration = 100  # 示例值
                t_start = time_synchronized()
                for _ in range(iteration):
                    predictions = model(img.to(device))[0]  # 多次推理
                torch.cuda.synchronize()  # 确保所有GPU操作完成
                t_end = time_synchronized()
                elapsed_time = t_end - t_start
                speed_time = elapsed_time / iteration * 1000  # 单次时间（ms）
                fps = iteration / elapsed_time  # FPS
                print("FPS: {}".format(fps))
                print("inference+NMS time: {}".format(t_end - t_start))

                predict_boxes = predictions["boxes"].to("cpu").numpy() * 0.9
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

                # if len(predict_boxes) == 0:
                #     print("没有检测到任何目标!")
                #     return

                plot_img = draw_objs(original_img,
                                     boxes=predict_boxes,
                                     classes=predict_classes,
                                     scores=predict_scores,
                                     masks=predict_mask,
                                     category_index=category_index,
                                     line_thickness=3,
                                     font='arial.ttf',
                                     font_size=20)
                plt.imshow(plot_img)
                # plt.show()
                # 保存预测的图片结果
                plot_img.save(output + imgs_str[ids])


if __name__ == '__main__':
    main()

