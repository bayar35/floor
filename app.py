import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл олдсонгүй'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл сонгогдоогүй байна'})
    
    filepath = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    file.save(filepath)
    return jsonify({'success': True})

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/static/furniture/<path:filename>')
def get_furniture(filename):
    return send_from_directory('static/furniture', filename)

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        points = data.get('points', [])
        furniture_list = data.get('furniture', [])
        
        img_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
        img = cv2.imread(img_path)
        if img is None:
            return jsonify({'success': False, 'error': 'Оролт болсон зураг олдсонгүй'})

        if len(points) < 4:
            return jsonify({'success': False, 'error': 'Дөрвөн булангийн цэгийг бүрэн сонгоно уу'})

        # 1. Шалны хэсгийг олох болон паркет/модон texture суурилуулах
        pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        
        # Жишээ паркет texture (Хэрэв static/furniture/wood_texture.jpg байхгүй бол өнгөөр дүүргэнэ)
        texture_path = 'static/furniture/wood_texture.jpg'
        if os.path.exists(texture_path):
            texture = cv2.imread(texture_path)
        else:
            # Паркет байхгүй бол бор өнгийн модон текстур үүсгэх эсвэл зурах
            texture = np.ones((300, 300, 3), dtype=np.uint8) * 120
            texture[:, :, 0] = 50   # B
            texture[:, :, 1] = 90   # G
            texture[:, :, 2] = 140  # R

        # Перспективийн хувиргалт хийх (Perspective Warp)
        h, w = img.shape[:2]
        t_h, t_w = texture.shape[:2]
        src_pts = np.array([[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]], dtype=np.float32)
        
        # Perspective Matrix олох
        M = cv2.getPerspectiveTransform(src_pts, pts)
        warped_texture = cv2.warpPerspective(texture, M, (w, h))

        # Маск үүсгэх
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
        
        # Зураг дээр шалыг уусгах
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        texture_fg = cv2.bitwise_and(warped_texture, warped_texture, mask=mask)
        combined = cv2.add(img_bg, texture_fg)

        # 2. Тавилгуудыг байрлал болон хэмжээний дагуу үнэн зөвөөр зурах
        disp_width = data.get('canvasWidth', w)
        disp_height = data.get('canvasHeight', h)
        
        scale_x = w / disp_width
        scale_y = h / disp_height

        for item in furniture_list:
            furn_path = item['src'].lstrip('/')
            if not os.path.exists(furn_path):
                continue
            
            furn_img = cv2.imread(furn_path, cv2.IMREAD_UNCHANGED)
            if furn_img is None:
                continue

            f_w = int(item['width'] * scale_x)
            f_h = int(item['height'] * scale_y)
            
            # Хэмжээ хэт бага эсвэл 0 байхаас сэргийлэх
            if f_w <= 0 or f_h <= 0:
                continue
                
            furn_resized = cv2.resize(furn_img, (f_w, f_h))

            fx = int(item['x'] * scale_x)
            fy = int(item['y'] * scale_y)

            # PNG transparent (Alpha) сувгаар зурган дээр давхарлаж байрлуулах
            for y_idx in range(f_h):
                jy = fy + y_idx
                if jy >= h or jy < 0: 
                    continue
                for x_idx in range(f_w):
                    jx = fx + x_idx
                    if jx >= w or jx < 0: 
                        continue
                    
                    if furn_resized.shape[2] == 4:
                        alpha = furn_resized[y_idx, x_idx, 3] / 255.0
                        if alpha > 0:
                            for c in range(3):
                                combined[jy, jx, c] = (1 - alpha) * combined[jy, jx, c] + alpha * furn_resized[y_idx, x_idx, c]
                    else:
                        for c in range(3):
                            combined[jy, jx, c] = furn_resized[y_idx, x_idx, c]

        result_path = os.path.join(RESULT_FOLDER, 'output.jpg')
        cv2.imwrite(result_path, combined)
        
        return jsonify({'success': True, 'resultUrl': '/results/output.jpg?t=' + str(np.random.randint(10000))})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
