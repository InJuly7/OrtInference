import cv2
import random
import numpy as np
import onnxruntime
from scipy.ndimage import zoom



def mask_nms(input_imgs, scores, iou_threshold=0.5):
    """
    Args:
        input_imgs: np.ndarray, shape (N, H, W) - 掩码集合
        scores: np.ndarray, shape (N,) - 每个掩码的得分
        iou_threshold: float - IoU阈值，高于此值的掩码会被抑制
    Returns:
        keep_indices: list - 保留的掩码索引
    """
    # 1. 按得分降序排序
    sorted_indices = np.argsort(scores)[::-1]  # 从高到低
    sorted_masks = input_imgs[sorted_indices]
    sorted_scores = scores[sorted_indices]
    keep_indices = []
    # 2. 逐个比较掩码
    while len(sorted_indices) > 0:
        # 当前得分最高的掩码
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        if len(sorted_indices) == 1:
            break
        # 计算当前掩码与其他掩码的 IoU
        current_mask = sorted_masks[0]
        other_masks = sorted_masks[1:]
        # IoU = 交集 / 并集
        intersection = np.logical_and(current_mask, other_masks).sum(axis=(1, 2))
        union = np.logical_or(current_mask, other_masks).sum(axis=(1, 2))
        iou = intersection / (union + 1e-6)  # 避免除零
        # 3. 保留 IoU 低于阈值的掩码（非高度重叠）
        low_overlap_indices = np.where(iou < iou_threshold)[0] + 1  # +1 因为 other_masks 是 [1:]
        sorted_indices = sorted_indices[low_overlap_indices]
        sorted_masks = sorted_masks[low_overlap_indices]
        sorted_scores = sorted_scores[low_overlap_indices]
    return keep_indices
# [1,25200,38],[1,32,160,160]
def mask_decode(output0, output1,ori_size=(None,None),net_size=(640,640),nms=0.5,score=0.5):
    bboxes = []
    scores = []
    output0 = output0[0] # [25200,38]
    output1 =  np.array(output1[0]) # [32,160,160]
    mask = output0[:,4] > 0.9
    output = output0[mask]
    target_rows = []
    target_masks = []
    for row in output[:]:
        cx, cy, w,h,conf,score_ = row[:6]
        cx = ori_size[0] * cx / net_size[0]
        cy =  ori_size[1] * cy / net_size[1]
        w = int(ori_size[0] * w / net_size[0])
        h = int(ori_size[1] * h / net_size[1])
        if(score_ < score):
            continue
        scores.append(conf)
        target_rows.append(row[6:])
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        bboxes.append([x,y,w,h])
        mask = np.zeros((160,160),dtype=np.float32)
        x = int((row[0] - [row[2]/2])/4)
        y = int((row[1] - [row[3]/2])/4)
        w = int(row[2]/4)
        h = int(row[3]/4)
        mask[y:y+h,x:x+w] = 1
        target_masks.append(mask)
        # cv2.imshow('mask',mask)
        # cv2.waitKey(0)
    target_scores = np.array(scores)
    target_masks = np.array(target_masks)
    target_rows = np.array(target_rows)
    output1_ = output1.reshape(32,-1)
    T = np.dot(target_rows,output1_) # (1884, 160*160)
    T = T.reshape(target_rows.shape[0],160,160)

    print(T.shape)
    keep_indices = mask_nms(target_masks, target_scores, 0.5)
    print(len(keep_indices))

    result_mask = np.zeros((160,160),dtype=np.float32)

    for idx in keep_indices:
        result_mask += target_masks[idx]

    cv2.imshow('mask', result_mask)
    cv2.waitKey(0)
def inference(img_path, onnx_file):

    # 初始化ONNX Runtime会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_file, providers=providers)
    
    # 读取并预处理图像
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    ori_img = img.copy()
    ori_h, ori_w = img.shape[:2]
    # 获取模型输入尺寸（假设输入是[1,3,640,640]）
    input_shape = session.get_inputs()[0].shape  # 例如 [1, 3, 640, 640]
    new_h, new_w = input_shape[2], input_shape[3]
    # 调整图像尺寸和格式
    img = cv2.resize(img, (new_w, new_h))  # 注意OpenCV的尺寸顺序是(width, height)

    img = np.transpose(img, (2, 0, 1))     # HWC -> CHW
    input_data = np.expand_dims(img, axis=0)  # 添加batch维度 -> [1,3,640,640]
    input_data = input_data.astype(np.float32) / 255.0  # 归一化

    # 执行推理
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # [1,25200,38]
    # [1,32,160,160]
    outputs = session.run(output_names, {input_name: input_data})
    # 在打印前添加：
    # np.set_printoptions(suppress=True, floatmode='fixed')  # 禁用科学计数法，固定小数位

    mask_decode(outputs[0], outputs[1],(ori_w, ori_h)) # only batch = 1
if __name__ == '__main__':

    onnx_file = './models/table/best.onnx'
    img_path = './assets/input/1DE646EB9BD751717427E49E1C31AF56.jpg'
    inference(img_path, onnx_file)
