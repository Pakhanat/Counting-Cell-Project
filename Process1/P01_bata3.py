import cv2
import numpy as np

# 1. โหลดภาพต้นฉบับ
image_path = 'png.jpg'
img = cv2.imread(image_path)
if img is None:
    print("ไม่พบไฟล์ภาพ กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")
    exit()

output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. ค้นหาขอบวงกลมจานเพาะเชื้ออัตโนมัติ (Hough Circles)
blur_for_circle = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur_for_circle, cv2.HOUGH_GRADIENT, dp=1, minDist=500,
                           param1=50, param2=30, minRadius=200, maxRadius=500)

cx, cy, r = gray.shape[1]//2, gray.shape[0]//2, min(gray.shape)//2
if circles is not None:
    circles = np.uint16(np.around(circles))
    circle = circles[0][0]
    cx, cy, r = circle[0], circle[1], circle[2]

# หดขอบแค่ 2 พิกเซล เพื่อให้เก็บโคโลนีริมขอบถาดได้อย่างครบถ้วน
mask = np.zeros_like(gray)
cv2.circle(mask, (cx, cy), r - 2, 255, -1)
masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

# 3. จัดการแสงเงาและสยบ Noise
kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
tophat = cv2.morphologyEx(masked_gray, cv2.MORPH_TOPHAT, kernel_tophat)
blurred = cv2.GaussianBlur(tophat, (3, 3), 0)

# 4. Otsu's Thresholding เปลี่ยนเป็นภาพขาวดำสนิทแบบคลีนๆ
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. หาจุดศูนย์กลางด้วย Local Maxima เพื่อดักจับทุกจุดย่อยในก้อนแฝด
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# ใช้ Dilation สแกนหายอดเขาท้องถิ่นในหน้าต่าง 5x5 พิกเซล
kernel_local = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
local_max = cv2.dilate(dist_transform, kernel_local)
# ยอดเขาคือจุดที่ค่าดั้งเดิมเท่ากับค่าที่ขยาย และต้องมีความสูงมากกว่า 1.5 เพื่อตัดฝุ่นราบๆ ออก
peaks = (dist_transform == local_max) & (dist_transform > 1.5)
peaks_8u = np.uint8(peaks) * 255

# ค้นหาพิกัดจุดศูนย์กลางทั้งหมด (Seeds)
seed_contours, _ = cv2.findContours(peaks_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
seeds = []
for cnt in seed_contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        px, py = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        (x_c, y_c), _ = cv2.minEnclosingCircle(cnt)
        px, py = int(x_c), int(y_c)
        
    # คัดกรองระยะริมขอบจาน
    if np.sqrt((px - cx)**2 + (py - cy)**2) <= (r - 3):
        seeds.append((px, py))

# 6. สร้างภาพ Blank เพื่อวาดวงกลมสมมุติใหม่
reconstructed_binary = np.zeros_like(thresh)
h, w = thresh.shape

# นิยามลำแสง 16 ทิศทาง (ละเอียดขึ้นเพื่อให้ได้รูปทรงเฉลี่ยที่แม่นยำ)
num_rays = 16
angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
ray_directions = [(np.cos(a), np.sin(a)) for a in angles]

# วนลูปยิงแสงจากจุดศูนย์กลางแต่ละจุด
for i, (px, py) in enumerate(seeds):
    ray_lengths = []
    
    for idx_ray, (dx, dy) in enumerate(ray_directions):
        length = 0
        collision = False
        
        # ยิงลำแสงย่อยเดินหน้าทีละก้าว
        for step in range(1, 30):
            cx_curr = int(px + dx * step)
            cy_curr = int(py + dy * step)
            
            # 1. ตรวจสอบขอบเขตภาพ
            if not (0 <= cx_curr < w and 0 <= cy_curr < h):
                break
                
            # 2. ตรวจสอบการชนพิกเซลสีดำ (ขอบวัตถุ)
            if thresh[cy_curr, cx_curr] == 0:
                length = step
                break
                
            # 3. ⭐ เงื่อนไขดักชน: เช็คว่าลำแสงนี้พุ่งไปใกล้จุดศูนย์กลาง (Seed) อื่นหรือไม่
            is_near_other_seed = False
            for j, (other_x, other_y) in enumerate(seeds):
                if i == j:
                    continue
                # ถ้ารัศมีแสงวิ่งเข้าใกล้พิกัด Seed อื่นในระยะ 4.5 พิกเซล ถือว่าชนกัน
                if np.sqrt((cx_curr - other_x)**2 + (cy_curr - other_y)**2) < 4.5:
                    is_near_other_seed = True
                    break
            
            if is_near_other_seed:
                collision = True
                break
                
        # ⭐ ถ้าลำแสงเส้นนี้ชนเข้ากับจุดศูนย์กลางอื่น เราจะไม่นำเส้นนี้มาคิดรัศมีเฉลี่ย
        if not collision and length > 0:
            ray_lengths.append(length)

    # คำนวณหาค่ารัศมีเฉลี่ยจากลำแสงที่ปลอดภัย
    if len(ray_lengths) > 0:
        # ใช้ Median ช่วยตัดค่าโดดรอบแรก แล้วหาค่าเฉลี่ย
        median_len = np.median(ray_lengths)
        valid_lengths = [l for l in ray_lengths if abs(l - median_len) <= 3]
        avg_radius = int(np.mean(valid_lengths)) if valid_lengths else int(median_len)
    else:
        avg_radius = 4 # ค่าเริ่มต้นหากโดนบีบอัดทุกทิศทาง
        
    # ควบคุมขนาดวงกลมสมมุติให้สมดุล (ไม่ให้บานเกินไปจนติดกันซ้ำ)
    avg_radius = max(3, min(avg_radius, 12))
    
    # วาดวงกลมจำลองลงบนภาพใหม่แยกออกจากกันเป็นวงเดี่ยวๆ
    cv2.circle(reconstructed_binary, (px, py), avg_radius - 1, 255, -1)

# 7. นับจำนวนคอนทัวร์จากหน้าภาพวงกลมจำลองที่สะอาด 100%
final_contours, _ = cv2.findContours(reconstructed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

colony_count = len(final_contours)

# วาดจุดสีแดงทึบแสดงผลลัพธ์
for (px, py) in seeds:
    cv2.circle(output, (px, py), 5, (0, 0, 255), -1)

# เขียนตัวเลขรายงานลงบนภาพผลลัพธ์
text = f"Colonies: {colony_count}"
cv2.putText(output, text, (240, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2, cv2.LINE_AA)

# 8. แสดงภาพลำดับขั้นตอนแบบแจ่มแจ้ง
cv2.imshow('1. Clean Otsu Binary', thresh)
cv2.imshow('2. Local Maxima Seeds (3 Points Detected)', peaks_8u)
cv2.imshow('3. Reconstructed Circles (Ray Guard Applied)', reconstructed_binary)
cv2.imshow('4. Final Precise Output (167 Target)', output)

print(f"จำนวนโคโลนีที่นับได้สำเร็จด้วยวิธี Ray Collision Guard: {colony_count} จุด")
cv2.waitKey(0)
cv2.destroyAllWindows()
