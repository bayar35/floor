import os
import cv2
import numpy as np
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
        points = data.get('points') # [{x, y}, ...]

        if len(points) < 4:
            return jsonify({'success': False, 'error': 'Бодит харагдуулахын тулд дор хаяж 4 цэг (шарны 4 буланг) сонгоно уу.'})

        input_path = os.path.join(UPLOAD_FOLDER, 'room.jpg')
        texture_path = os.path.join(TEXTURE_FOLDER, 'parquet.jpg')

        if not os.path.exists(input_path) or not os.path.exists(texture_path):
            return jsonify({'success': False, 'error': 'Зураг эсвэл текстур олдсонгүй.'})

        img = cv2.imread(input_path)
        texture = cv2.imread(texture_path)
        th, tw = texture.shape[:2]

        pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)

        # 4 цэгийг ашиглан хэтийн төлөвт шилжүүлэх (Perspective Transform)
        # Хэрэглэгчийн сонгосон эхний 4 цэг: [доод зүүн, доод баруун, дээд баруун, дээд зүүн] дарааллаар байвал хамгийн сайн гарна
        dst_pts = pts[:4]
        
        # Текстурийн дөрвөн өнцөг
        src_pts = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])

        # Матриц олох болон паркетын зургийг гажуудуулах (warp)
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_texture = cv2.warpPerspective(texture, matrix, (img.shape[1], img.shape[0]))

        # Mask үүсгэх (олон өнцөгтийн дотор талыг сонгох)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        
        # Захоор нь зөөллөх (Blending хийхэд ирмэг нь цэвэрхэн харагдах зорилгоор)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        # Үндсэн зураг болон паркетыг холих
        bg = img.astype(float) * (1 - mask_3ch)
        fg = warped_texture.astype(float) * mask_3ch
        combined = cv2.add(bg, fg).astype(np.uint8)

        # Үр дүнг хадгалах
        output_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
        cv2.imwrite(output_path, combined)

        return jsonify({'success': True, 'resultUrl': '/static/uploads/result.jpg?' + str(np.random.randint(1000))})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
