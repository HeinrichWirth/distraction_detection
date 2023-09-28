from ultralytics import YOLO
import cv2
import torch
from typing import List, Literal
import pandas as pd
import easyocr
import os
from functions import determine_roles, determine_box_roles, fill_missing, get_lb_results, add_entry_to_json, Dist_est, filter_keypoints, main, Args, get_date_from_image



reader = easyocr.Reader(['en'])

model_key_points = YOLO('yolov8l-pose.pt')
model_key_cell_phone = YOLO('yolov8x.pt')

model_main = Dist_est()

# Загрузка модели
model_path = os.path.join(os.path.dirname(__file__), "dist_model.pth")
model_main.load_state_dict(torch.load(model_path))
model_main.eval()

video_path = 'path_to_video'
cap = cv2.VideoCapture(video_path)

# Проверяем, удалось ли открыть видео
if not cap.isOpened():
    print("Error.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Video FPS: {fps}")
fr = 0

ones_mech = []
twos_mech = []
threes_mech = []
empty_mech = []

ones_assest = []
twos_assest = []
threes_assest = []

while True:
    ret, frame = cap.read()
    fr+=1
    if not ret:
        break

    if fr < 10:
        data_stamp = get_date_from_image(frame, reader)

    key_points = model_key_points(frame, show=False, conf = 0.35)

    cell_phone = model_key_cell_phone(frame, show=False, conf = 0.2)
    
    cls = cell_phone[0].boxes.cls.cpu().numpy()
    xyxyn = cell_phone[0].boxes.xyxyn.cpu().numpy()
   
    # Отфильтровываем только те боксы, которые относятся к cell_phone
    cell_phone_boxes = xyxyn[cls == 67]

    args = Args(frame)

    cos_sin, boxi = main(args)

    train_driver = []
    helper = []
    if key_points[0].keypoints.data.numel() == 0:
        print('ПУСТОЙ ТЕНЗОР')
        ones_mech.append(0)
        twos_mech.append(0)
        threes_mech.append(0)
        empty_mech.append(1)
        continue

    #Роль для точек
    roles_points = determine_roles(key_points[0].keypoints.data)
    roles_box = determine_box_roles(boxi)
    roles_box_cell = determine_box_roles(cell_phone_boxes)

    #Только нужные ключевые точки
    key_points_data = key_points[0].keypoints.data
    filtered_data = filter_keypoints(key_points_data)

    # Если в кадре только один человек
    if len(roles_points) == 1:
        role = roles_points[0]  
        index_box = roles_box.index(role) if role in roles_box else None
        index_points = 0  # так как у нас только одна роль в кадре
        index_phone = roles_box_cell.index(role) if role in roles_box_cell else None
        
        cos_sin_data = fill_missing(cos_sin[index_box] if index_box is not None else None, [3])
        points_data = fill_missing(filtered_data[index_points, :, :2].tolist(), [17, 2])
        phone_data = fill_missing(cell_phone_boxes[index_phone].tolist() if (index_phone is not None and len(cell_phone_boxes) > index_phone) else None, [4])
        
        combined_tensor = torch.cat([cos_sin_data.flatten(), points_data.flatten(), phone_data.flatten()])

    else:
        driver_index_box = roles_box.index('driver') if 'driver' in roles_box else None
        assistant_index_box = roles_box.index('assistant') if 'assistant' in roles_box else None
        
        driver_index_points = 0 if 'driver' in roles_points else None
        assistant_index_points = 1 if 'assistant' in roles_points else None
        
        driver_index_phone = roles_box_cell.index('driver') if 'driver' in roles_box_cell else None
        assistant_index_phone = roles_box_cell.index('assistant') if 'assistant' in roles_box_cell else None

        # Извлечение данных
        driver_cos_sin = fill_missing(cos_sin[driver_index_box] if driver_index_box is not None else None, [3])
        assistant_cos_sin = fill_missing(cos_sin[assistant_index_box] if assistant_index_box is not None else None, [3])
        
        driver_points = fill_missing(filtered_data[driver_index_points, :, :2].tolist() if (driver_index_points is not None and len(filtered_data) > driver_index_points) else None, [9, 2])
        assistant_points = fill_missing(filtered_data[assistant_index_points, :, :2].tolist() if (assistant_index_points is not None and len(filtered_data) > assistant_index_points) else None, [9, 2])
        
        driver_phone = fill_missing(cell_phone_boxes[driver_index_phone].tolist() if (driver_index_phone is not None and len(cell_phone_boxes) > driver_index_phone) else None, [4])
        assistant_phone = fill_missing(cell_phone_boxes[assistant_index_phone].tolist() if (assistant_index_phone is not None and len(cell_phone_boxes) > assistant_index_phone) else None, [4])

        # Объединение данных в один тензор
        driver_tensor = torch.cat([driver_cos_sin.flatten(), driver_points.flatten(), driver_phone.flatten()])
        assistant_tensor = torch.cat([assistant_cos_sin.flatten(), assistant_points.flatten(), assistant_phone.flatten()])


    with torch.no_grad():
        output = model_main(driver_tensor)

    predicted_class_driver = torch.argmax(output).item()


    if predicted_class_driver == 0:
        ones_mech.append(1)
        twos_mech.append(0)
        threes_mech.append(0)
    elif predicted_class_driver == 1:
        ones_mech.append(0)
        twos_mech.append(1)
        threes_mech.append(0)
    elif predicted_class_driver == 2:
        ones_mech.append(0)
        twos_mech.append(0)
        threes_mech.append(1)


    with torch.no_grad():
        output = model_main(assistant_tensor)

    predicted_class_assest = torch.argmax(output).item()


    if predicted_class_assest == 0:
        ones_assest.append(1)
        twos_assest.append(0)
        threes_assest.append(0)
    elif predicted_class_driver == 1:
        ones_assest.append(0)
        twos_assest.append(1)
        threes_assest.append(0)
    elif predicted_class_driver == 2:
        ones_assest.append(0)
        twos_assest.append(0)
        threes_assest.append(1)

    # Освобождаем память, связанную с текущим кадром
    del frame

# Закрываем видеофайл
cap.release()


driver_time_stamp_ones = get_lb_results('test', ones_mech, 12, 'max', 5)
driver_time_stamp_twos = get_lb_results('test', twos_mech, 12, 'max', 5)
driver_time_stamp_threes = get_lb_results('test', threes_mech, 12, 'max', 5)

assest_time_stamp_ones = get_lb_results('test', ones_assest, 12, 'max', 5)
assest_time_stamp_twos = get_lb_results('test', twos_assest, 12, 'max', 5)
assest_time_stamp_threes = get_lb_results('test', threes_assest, 12, 'max', 5)


driver_stamps = [driver_time_stamp_ones, driver_time_stamp_twos, driver_time_stamp_threes]
asset_stamps = [assest_time_stamp_ones, assest_time_stamp_twos]

disturbed_mech = {1:'distracted', 2:'phone', 3:'empty_place'}
listy = []

count = 1
for i in driver_stamps:
    print(i)

    for x in range(len(i['timestamps'])):
        v = {
        "time_start": data_stamp,
        "video_time": i['timestamps'][x],
        "type": disturbed_mech[count]
        }
        listy.append(v)
    count+=1

add_entry_to_json("Машинист", data_stamp, listy)



disturbed_assetst = {1:'distracted', 2:'phone'}
listy = []

count = 1
for i in asset_stamps:
    print(i)

    for x in range(len(i['timestamps'])):
        v = {
        "time_start": data_stamp,
        "video_time": i['timestamps'][x],
        "type": disturbed_assetst[count]
        }
        listy.append(v)
    count+=1

add_entry_to_json("Помощник машиниста", data_stamp, listy)

