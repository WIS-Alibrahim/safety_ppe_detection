import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل النموذج
model = YOLO('best.pt')  # أو 'weights/best.pt'

# 2. تعريف المعدات المطلوبة للفحص
REQUIRED_GEAR = {
    0: "Helmet",   # خوذة
    4: "Goggles",  # نظارات
    2: "Vest",     # سترة
    1: "Gloves",   # قفازات
    3: "Boots"     # أحذية
}

def is_inside(item_box, person_box):
    """التحقق مما إذا كان مركز العنصر يقع داخل مربع العامل"""
    ix1, iy1, ix2, iy2 = item_box
    px1, py1, px2, py2 = person_box
    
    # حساب مركز العنصر
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    
    # فحص الارتباط المكاني
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # تشغيل الكشف
    results = model(frame, conf=0.30, imgsz=640)
    
    detected_persons = []
    detected_items = []

    # تصنيف المخرجات إلى عمال ومعدات
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            coords = tuple(map(int, box.xyxy[0]))

            if cls_id == 6:  # Person (عامل)
                detected_persons.append((coords, conf))
            elif cls_id in REQUIRED_GEAR:  # معدة سلامة
                detected_items.append((cls_id, coords, conf))

    # معالجة كل عامل على حدة وإنشاء التقرير الخاص به
    for idx, (p_box, p_conf) in enumerate(detected_persons, 1):
        px1, py1, px2, py2 = p_box

        # قاموس الفحص الخاص بهذا العامل تحديداً
        person_gear = {gear_id: False for gear_id in REQUIRED_GEAR}

        # فحص العناصر التابعة لمساحة هذا العامل
        for gear_id, i_box, i_conf in detected_items:
            if is_inside(i_box, p_box):
                person_gear[gear_id] = True

        # تحديد هل العامل ملتزم بالكامل أم مخالِف
        is_fully_compliant = all(person_gear.values())
        worker_color = (0, 255, 0) if is_fully_compliant else (0, 0, 255) # أخضر للملتزم، أحمر للمخالف

        # 1. رسم صندوق العامل
        cv2.rectangle(frame, (px1, py1), (px2, py2), worker_color, 2)

        # 2. إعداد لوحة التقرير الجانبية الملتصقة بالعامل
        panel_w, panel_h = 210, 160
        # تحديد موقع اللوحة (على يمين العامل أو يساره إذا كان قريباً من الحافة)
        panel_x = px2 + 10 if (px2 + panel_w < frame.shape[1]) else max(10, px1 - panel_w - 10)
        panel_y = max(10, py1)

        # خلفية سوداء شفافة للوحة
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # عنوان اللوحة (رقم العامل)
        cv2.putText(frame, f"Worker #{idx}", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.line(frame, (panel_x + 10, panel_y + 32), (panel_x + panel_w - 10, panel_y + 32), (100, 100, 100), 1)

        # كتابة التقرير التفصيلي لكل معدة
        y_offset = panel_y + 52
        for gear_id, gear_name in REQUIRED_GEAR.items():
            status = person_gear[gear_id]
            status_text = "[OK]" if status else "[MISSING]"
            status_color = (0, 255, 0) if status else (0, 0, 255)

            line_text = f"{gear_name}: {status_text}"
            cv2.putText(frame, line_text, (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            y_offset += 20

    # عرض عدد العمال الإجمالي في أعلى الشاشة
    cv2.putText(frame, f"Active Workers Monitored: {len(detected_persons)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Worker Safety Inspection System - Multi-Person HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()