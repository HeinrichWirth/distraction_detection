import numpy as np
import cv2
import os
from math import cos, sin
import onnxruntime
import numba as nb
import numpy as np
import torch
import pandas as pd
from typing import List, Literal
import itertools
import datetime
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import List, Literal
import itertools
import datetime
import json
from sklearn.metrics import confusion_matrix


def determine_roles(tensor):
    """
    Определяет роли на основе средних x-координат ключевых точек.

    Parameters:
    - tensor: PyTorch тензор размером (число точек * число людей, 3), содержащий x, y координаты и вероятности для каждой точки.

    Returns:
    - Список ролей в порядке соответствия переданным спискам точек.
    """
    if tensor.shape[0] < 2:
        return []

    num_people = tensor.shape[0] // 17

    if num_people == 1:
        return ['driver']

    else:
        num_points = tensor.shape[0] // 2
        person1_points = tensor[:num_points, :]
        person2_points = tensor[num_points:, :]

        avg_x_person1 = torch.mean(person1_points[:, 0])
        avg_x_person2 = torch.mean(person2_points[:, 0])

        if avg_x_person1 > avg_x_person2:
            return ['driver', 'assistant']
        else:
            return ['assistant', 'driver']


def determine_box_roles(boxes):
    """
    Определяет роли на основе средних x-координат баундинг боксов.

    Parameters:
    - boxes: список баундинг боксов.

    Returns:
    - Список ролей ('driver' или 'assistant') в порядке соответствия переданным баундинг боксам.
    """
    if len(boxes) == 1:
        return ['driver']

    else:
        avg_xs = [(box[0] + box[2]) / 2 for box in boxes]
        sorted_indices = sorted(range(len(avg_xs)), key=lambda k: avg_xs[k])
        roles = ['assistant' if i == sorted_indices[0] else 'driver' for i in range(len(boxes))]
        return roles
    

def filter_keypoints(data):
    """
    Фильтрует ключевые точки тензора, оставляя только интересующие нас.

    Parameters:
    - data: PyTorch тензор размером (число людей, число точек, 3).

    Returns:
    - Тензор с отфильтрованными ключевыми точками.
    """

    # Индексы ключевых точек COCO, которые нам интересны
    required_indices = [0, 3, 4, 5, 6, 7, 8, 9, 10]

    # Создаем пустой список для сохранения отфильтрованных точек
    filtered_data = []

    for person_keypoints in data:
        filtered_person_keypoints = person_keypoints[required_indices]
        filtered_data.append(filtered_person_keypoints)

    return torch.stack(filtered_data)


# Функция для заполнения недостающих данных
def fill_missing(data, size, dim=0):
    if not data:
        return torch.zeros(size)
    else:
        return torch.tensor(data).cpu()
    


def get_lb_results(
        filename: str, 
        model_output: List[int], 
        video_fps: float,
        smoothing_type: Literal['max', 'mode'],
        window: int = 5) -> dict:
    
    df = pd.DataFrame.from_dict({'model_output': model_output})
    if smoothing_type == 'max':
        df['smoothed'] = df['model_output'].rolling(window=window, min_periods=1).max().astype('int')
        df['smoothed_rev'] = df['model_output'].iloc[::-1].rolling(window=window, min_periods=1).max().astype('int').iloc[::-1]
        df['smoothed'] = df['smoothed'] * df['smoothed_rev']
    elif smoothing_type == 'mode':
        df['smoothed'] = df['model_output'].rolling(window=window, min_periods=1).mean()
        df['smoothed'] = df['smoothed'].apply(lambda x: 1 if x >= 0.5 else 0)

    smoothed_list = df['smoothed'].tolist()
    intervals = [(x[0], len(list(x[1]))) for x in itertools.groupby(smoothed_list)]

    min_frames = 3 * video_fps
    starting_times = []
    for idx, interval in enumerate(intervals):
        if interval[0] == 1 and interval[1] >= min_frames:
            starting_frame = 0
            if idx == 0:
                starting_times.append('00:00')
            else:
                for i in range(idx):
                    starting_frame += intervals[i][1]
                num_seconds = int(starting_frame / video_fps)
                starting_time = str(datetime.timedelta(seconds=num_seconds))[-5:]
                starting_times.append(starting_time)

    return {
        'filename': filename,
        'cases_count': len(starting_times),
        'timestamps': starting_times
    }


def add_entry_to_json(role, date, dist):
    try:
        with open("data.json", "r", encoding="utf-8") as file:
            loaded_data = json.load(file)
            data = loaded_data["data"]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    new_entry = {
        "role": role,
        "date": date,
        "dist": dist
    }

    data.append(new_entry)

    with open("data.json", "w", encoding="utf-8") as file:
        json.dump({"data": data}, file, ensure_ascii=False, indent=4)



class Dist_est(nn.Module):
    def __init__(self, input_size=25, hidden_size=350, num_classes=3, dropout_prob=0.2):
        super(Dist_est, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_prob)
    
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
   
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
  
        x = self.fc3(x)
        
        return x
    

idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]


def softmax(x):
    x -= np.max(x,axis=1, keepdims=True)
    a = np.exp(x)
    b = np.sum(np.exp(x), axis=1, keepdims=True)
    return a/b

def resize_and_pad(src, size, pad_color=0):
    ()
    img = src
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = \
            np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = \
            np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        pad_color = [pad_color]*3
    scaled_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=interp
    )
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    return scaled_img


@nb.njit('i8[:](f4[:,:],f4[:], f4, b1)', fastmath=True, cache=True)
def nms_cpu(boxes, confs, nms_thresh, min_mode):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def main(args):
    yolov4_head_H = 480
    yolov4_head_W = 640
    whenet_H = 224
    whenet_W = 224

    base_path = os.path.dirname(__file__)
    yolov4_model_name = 'yolov4_headdetection'
    model_name = f"{yolov4_model_name}_{yolov4_head_H}x{yolov4_head_W}.onnx"
    model_folder = f"saved_model_{whenet_H}x{whenet_W}"

    # YOLOv4-Head
    yolov4_head = onnxruntime.InferenceSession(
        os.path.join(base_path, "HeadPoseEstimation-WHENet-yolov4-onnx-openvino", model_folder, model_name),
        providers=[
            # 'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    )

    yolov4_head_input_name = yolov4_head.get_inputs()[0].name
    yolov4_head_output_names = [output.name for output in yolov4_head.get_outputs()]
    yolov4_head_output_shapes = [output.shape for output in yolov4_head.get_outputs()]
    assert yolov4_head_output_shapes[0] == [1, 18900, 1, 4] # boxes[N, num, classes, boxes]
    assert yolov4_head_output_shapes[1] == [1, 18900, 1]    # confs[N, num, classes]

    # WHENet
    whenet_input_name = None
    whenet_output_names = None
    whenet_output_shapes = None
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model_name_1 = f"whenet_1x3x224x224_prepost.onnx"
    model_folder_1 = f"saved_model_{whenet_H}x{whenet_W}"

    if args.whenet_mode == 'onnx':
        whenet = onnxruntime.InferenceSession(
            os.path.join(base_path, "HeadPoseEstimation-WHENet-yolov4-onnx-openvino", model_folder_1, model_name_1),
            providers=[
                # 'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        )
        whenet_input_name = whenet.get_inputs()[0].name
        whenet_output_names = [output.name for output in whenet.get_outputs()]

    exec_net = None
    input_name = None
    cap = args.frame
    frame = cap
    conf_thresh = 0.60
    nms_thresh = 0.50

    # Resize
    resized_frame = resize_and_pad(
        frame,
        (yolov4_head_H, yolov4_head_W)
    )
    width = resized_frame.shape[1]
    height = resized_frame.shape[0]

    rgb = resized_frame[..., ::-1]

    chw = rgb.transpose(2, 0, 1)
    # нормализация [0, 1]
    chw = np.asarray(chw / 255., dtype=np.float32)
    # hwc --> nhwc
    nchw = chw[np.newaxis, ...]

    boxes, confs = yolov4_head.run(
        output_names = yolov4_head_output_names,
        input_feed = {yolov4_head_input_name: nchw}
    )
   
    boxes = boxes[0][:, 0, :]

    confs = confs[0][:, 0]

    argwhere = confs > conf_thresh
    boxes = boxes[argwhere, :]
    confs = confs[argwhere]
    # nms
    heads = []
    keep = nms_cpu(
        boxes=boxes,
        confs=confs,
        nms_thresh=nms_thresh,
        min_mode=False
    )
    if (keep.size > 0):
        boxes = boxes[keep, :]
        confs = confs[keep]
        for k in range(boxes.shape[0]):
            heads.append(
                [
                    int(boxes[k, 0] * width),
                    int(boxes[k, 1] * height),
                    int(boxes[k, 2] * width),
                    int(boxes[k, 3] * height),
                    confs[k],
                ]
            )

    canvas = resized_frame.copy()
    # ============================================================= WHENet
    mm = []

    croped_resized_frame = None
    if len(heads) > 0:
        for head in heads:
            ll = []
            x_min = head[0]
            y_min = head[1]
            x_max = head[2]
            y_max = head[3]

            y_min = max(0, y_min - abs(y_min - y_max) / 10)
            y_max = min(resized_frame.shape[0], y_max + abs(y_min - y_max) / 10)
            x_min = max(0, x_min - abs(x_min - x_max) / 5)
            x_max = min(resized_frame.shape[1], x_max + abs(x_min - x_max) / 5)
            x_max = min(x_max, resized_frame.shape[1])
            croped_frame = resized_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            croped_resized_frame = cv2.resize(croped_frame, (whenet_W, whenet_H))
            rgb = croped_resized_frame[..., ::-1]
            chw = rgb.transpose(2, 0, 1)
            nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)

            yaw = 0.0
            pitch = 0.0
            roll = 0.0
            if args.whenet_mode == 'onnx':
                outputs = whenet.run(
                    output_names = whenet_output_names,
                    input_feed = {whenet_input_name: nchw}
                )
                yaw = outputs[0][0][0]
                roll = outputs[0][0][1]
                pitch = outputs[0][0][2]
            elif args.whenet_mode == 'openvino':
      
                rgb = ((rgb / 255.0) - mean) / std
                output = exec_net.infer(inputs={input_name: nchw})
                yaw = output['yaw_new/BiasAdd/Add']
                roll = output['roll_new/BiasAdd/Add']
                pitch = output['pitch_new/BiasAdd/Add']

            yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
            ll.append(yaw)
            ll.append(pitch)
            ll.append(roll)
            mm.append(ll)
    return mm, boxes

            
class Args:
    def __init__(self, frame):
        self.frame = frame

    whenet_mode = 'onnx'
    height_width = '480x640'


def true_positive_for_class(y_true, y_pred, target_class):
    matrix = confusion_matrix(y_true, y_pred)
    if matrix.shape[0] <= target_class or matrix.shape[1] <= target_class:
        return 0
    return matrix[target_class, target_class]


def get_date_from_image(image, reader):
    """ returns 'YYYY-MM-DD' """
    cut = image[:100, 1300:,:]
    result = reader.readtext(cut)
    one_string = '-'.join([res[1] for res in result]).replace('*', '')
    date = one_string.replace(' ', '')[:10]
    return date