import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
TEXTURE_FOLDER = 'static/textures'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXTURE_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False})
    
    file_path = os.path.join(UPLOAD_FOLDER, 'room.jpg')
    file.save(file_path)
    return jsonify({'success': True})

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        points = data.get('points')
        furniture_list = data.get('furniture', []) # Вэбээс ирсэн тавилгуудын мэдээлэл [ {src, x, y, width, height}, ... ]

        if len(points) < 4:
            return jsonify({'success': False, 'error': 'Дор хаяж 4 цэг сонгоно уу.'})

        input_path = os.path.join(UPLOAD_FOLDER, 'room.jpg')
        texture_path = os.path.join(TEXTURE_FOLDER, 'parquet.jpg')

        img = cv2.imread(input_path)
        texture = cv2.imread(texture_path)
        th, tw = texture.shape[:2]

        pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)

        # 1. Шалны хэтийн төлөвийг олох (Perspective Transform)
        dst_pts = pts[:4]
        src_pts = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_texture = cv2.warpPerspective(texture, matrix, (img.shape[1], img.shape[0]))

        # Mask үүсгэх
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        # Шалны паркетыг үндсэн зурагтай нийлүүлэх
        bg = img.astype(float) * (1 - mask_3ch)
        fg = warped_texture.astype(float) * mask_3ch
        combined = cv2.add(bg, fg).astype(np.uint8)

        # 2. Тавилгуудыг зурган дээр давхарлаж render хийх
        canvas_width = data.get('canvasWidth', combined.shape[1])
        canvas_height = data.get('canvasHeight', combined.shape[0])
        
        scale_x = combined.shape[1] / canvas_width
        scale_y = combined.shape[0] / canvas_height

        for item in furniture_list:
            # Тавилгын зурах зам олох
            furn_path = item['src'].lstrip('/')
            if not os.path.exists(furn_path):
                continue
            
            furn_img = cv2.imread(furn_path, cv2.IMREAD_UNCHANGED) # Alpha сувгтай унших
            if furn_img is None:
                continue

            # Хэмжээг нь тааруулах
            f_w = int(item['width'] * scale_x)
            f_h = int(item['height'] * scale_y)
            furn_resized = cv2.resize(furn_img, (f_w, f_h))

            # Байрлал тооцох
            fx = int(item['x'] * scale_x)
            fy = int(item['y'] * scale_y)

            # PNG transparent ашиглан зурган дээр суулгах
            for c in range(0, 3):
                if furn_resized.shape[2] == 4:
                    alpha = furn_resized[:, :, 3] / 255.0 * (item.get('opacity', 1.0))
                    for y_idx in range(f_h):
                        jy = fy + y_idx
                        if jy >= combined.shape[0] or jy < 0: continue
                        for x_idx in range(f_w):
                            jx = fx + x_idx
                            if jx >= combined.shape[1] or jx < 0: continue
                            
                            a = alpha[y_idx, x_idx]
                            if a > 0:
                                combined[jy, jx, c] = (1 - a) * combined[jy, jx, c] + a * furn_resized[y_idx, x_idx, c]
                else:
                    # Хэрэв alpha байхгүй энгийн зураг бол шууд хуулах
                    pass

        # Үр дүнг хадгалах
        output_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
        cv2.imwrite(output_path, combined)

        return jsonify({'success': True, 'resultUrl': '/static/uploads/result.jpg?' + str(np.random.randint(1000))})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
