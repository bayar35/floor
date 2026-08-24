import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'roomImage' not in request.files:
        return jsonify({'error': 'Зураг олдсонгүй'})
    
    file = request.files['roomImage']
    if file.filename == '':
        return jsonify({'error': 'Файл сонгогдоогүй байна'})
    
    points_data = request.form.get('points')
    if not points_data:
        return jsonify({'error': 'Шалны өнцгийн координат олдсонгүй'})
    
    try:
        pts_list = json.loads(points_data)
        if len(pts_list) < 3:
            return jsonify({'error': 'Дор хаяж 3 цэг шаардлагатай'})
    except Exception as e:
        return jsonify({'error': 'Координатыг задлахад алдаа гарлаа'})

    room_path = os.path.join(app.config['UPLOAD_FOLDER'], 'room.jpg')
    file.save(room_path)
    
    room_img = cv2.imread(room_path)
    if room_img is None:
        return jsonify({'error': 'Оруулсан зургийг уншихад алдаа гарлаа'})
        
    h, w, _ = room_img.shape
    
    texture_path = 'static/textures/parquet.jpg'
    texture_img = cv2.imread(texture_path)
    if texture_img is None:
        return jsonify({'error': 'static/textures/parquet.jpg текстур олдсонгүй!'})
    
    # АНХААРУУЛГА: cv2.rotate хийхгүйгээр анхны босоо (уртаашаа) чиглэлээр нь үлдээнэ!
    
    # Бүх цэгүүдийг зургийн бодит хэмжээнд шилжүүлэх
    pts_dst = np.array([[p['x'] * w, p['y'] * h] for p in pts_list], dtype=np.float32)
    
    x, y, w_box, h_box = cv2.boundingRect(pts_dst)
    
    if w_box < 10 or h_box < 10:
        return jsonify({'error': 'Сонгосон хүрээ хэт жижиг байна'})

    # Текстурийг хангалттай том хэмжээтэйгээр давтаж бэлтгэх
    h_tex, w_tex, _ = texture_img.shape
    big_texture = np.zeros((h_box * 3, w_box * 3, 3), dtype=np.uint8)
    
    for ty in range(0, h_box * 3, h_tex):
        for tx in range(0, w_box * 3, w_tex):
            ty_end = min(ty + h_tex, h_box * 3)
            tx_end = min(tx + w_tex, w_box * 3)
            big_texture[ty:ty_end, tx:tx_end] = texture_img[:ty_end - ty, :tx_end - tx]
            
    h_big, w_big, _ = big_texture.shape
    pts_src = np.float32([
        [0, 0],
        [w_big, 0],
        [w_big, h_big],
        [0, h_big]
    ])
    
    # Коридорын гүн рүү хувиргах 4 булан
    sums = pts_dst.sum(axis=1)
    diffs = np.diff(pts_dst, axis=1)
    
    tl = pts_dst[np.argmin(sums)]
    br = pts_dst[np.argmax(sums)]
    tr = pts_dst[np.argmin(diffs)]
    bl = pts_dst[np.argmax(diffs)]
    
    pts_warp_dst = np.float32([tl, tr, br, bl])
    
    matrix, _ = cv2.findHomography(pts_src[:4], pts_warp_dst)
    warped_texture = cv2.warpPerspective(big_texture, matrix, (w, h))
    
    # Нарийн маск үүсгэх
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_int = np.int32(pts_dst)
    cv2.fillPoly(mask, [pts_int], 255)
    
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    
    room_float = room_img.astype(float)
    warped_float = warped_texture.astype(float)
    
    final_float = room_float * (1 - mask_3ch) + warped_float * mask_3ch
    final_result = np.clip(final_float, 0, 255).astype(np.uint8)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(output_path, final_result)
    
    return jsonify({'success': True, 'result_url': '/static/uploads/result.jpg?'})

if __name__ == '__main__':
    app.run(debug=True)