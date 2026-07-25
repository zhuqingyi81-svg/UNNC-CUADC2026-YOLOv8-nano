from ultralytics import YOLO
import cv2
import numpy as np  # 新增：用来处理颜色数组的工具

model = YOLO("runs/detect/train-2/weights/best.pt") 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, conf=0.3) # 基础及格线还是保留
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # 把坐标变成整数
            
            # 【你的绝招】：把 YOLO 框出来的这个区域，单独“剪裁”下来
            # 注意：OpenCV图像的切片是先Y后X
            roi = frame[y1:y2, x1:x2] 
            
            # 如果不小心剪出了空图，就跳过防报错
            if roi.size == 0: continue 

            # 把颜色转换到 HSV 模式（敲黑板：室外光线复杂，不要用RGB，用HSV最好找白色）
            # H(色调) S(饱和度/鲜艳度) V(亮度)
            # 白色的特点是：不鲜艳(S极低)，并且很亮(V很高)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 定义我们在室外认为的“白色”范围
            # 这个范围你可以根据实际当天的阳光修改：[H下限, S下限, V下限] 到 [H上限, S上限, V上限]
            lower_white = np.array([0, 0, 150])    # 亮度大于150，鲜艳度大于0
            upper_white = np.array([180, 50, 255]) # 亮度最高255，鲜艳度最高50(不鲜艳)
            
            # 制作一个“滤网”，把框里属于白色的部分变成255（纯白），不是白色的变成0（纯黑）
            mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            
            # 计算白色部分占整个框的比例
            white_pixels = cv2.countNonZero(mask) # 纯白点的数量
            total_pixels = mask.shape[0] * mask.shape[1] # 框的总像素量
            white_ratio = white_pixels / total_pixels
            
            print(f"框内白色占比: {white_ratio:.2f}")
            
            # 【核心审判】：如果你发现白色面积连 20% 都不到，说明根本不是桶！
            if white_ratio < 0.2:
                # 哪怕YOLO说是桶，颜色不对我也不认！这就是影子的克星！
                continue 

            # ========== 能活到这里的，绝对是真正的白桶！==========
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            offset_x = center_x - 320  
            offset_y = center_y - 240
            
            print(f"【发现真目标】偏差：X {offset_x}， Y {offset_y}")
            
            # 画一个绿色的框，让你在屏幕上看到真正的目标
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Drone Vision", frame)
    if cv2.waitKey(1) == ord('q'):
        break
