from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
import time, os
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table, func
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)

# Load dua model
model_det = YOLO('OD-Yolov9s.pt')
model_seg = YOLO('Seg-Yolo11sAug.pt')
model_det.conf = 0.7
model_seg.conf = 0.4

paslon_labels = ['paslon1', 'paslon2', 'paslon3']
lubang_luar_label = 'lubangCoblos'
segmentation_label = 'lubangCoblos'

# Database
DATABASE_URI = 'mysql+pymysql://root:@localhost:3306/pemiludb'
engine = create_engine(DATABASE_URI)
metadata = MetaData()
metadata.reflect(bind=engine)

paslon1 = Table('paslon1', metadata, autoload_with=engine)
paslon2 = Table('paslon2', metadata, autoload_with=engine)
paslon3 = Table('paslon3', metadata, autoload_with=engine)
golput = Table('golput', metadata, autoload_with=engine)

Session = sessionmaker(bind=engine)
session = Session()

camera_active = False

# 🎥 Ambil satu frame kamera
def get_camera_frame():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, -1)
    cap.release()
    return success, frame

@app.route('/capture_tidaksah', methods=['POST'])
def capture_tidaksah():
    success, frame = get_camera_frame()
    if success:
        save_dir = 'static/suara_tidaksah'
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'golput_{timestamp}.jpg'
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        session.execute(golput.insert().values(jumlah=1))
        session.commit()

    return redirect(url_for('index', result='golput'))

# 📸 Capture dan deteksi
@app.route('/capture', methods=['POST'])
def capture_image():
    success, frame = get_camera_frame()
    if not success:
        return jsonify({'message': 'Failed to capture image'}), 500

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_det_vis = img_rgb.copy()

    det_results = model_det(img_rgb)[0]
    paslon_boxes = {}
    lubang_luar_terdeteksi = False

    for box in det_results.boxes:
        cls_id = int(box.cls)
        cls_name = model_det.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        color = (0, 255, 0) if cls_name in paslon_labels else (0, 0, 255)
        cv2.rectangle(img_det_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_det_vis, cls_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls_name in paslon_labels:
            paslon_boxes[cls_name] = (x1, y1, x2, y2)
        elif cls_name == lubang_luar_label:
            lubang_luar_terdeteksi = True

    vote_result = "golput"
    vote_status = []

    if not lubang_luar_terdeteksi:
        for paslon, (x1, y1, x2, y2) in paslon_boxes.items():
            cropped_img = frame[y1:y2, x1:x2]
            seg_result = model_seg(cropped_img)[0]

            if seg_result.masks is not None:
                for i, cls_id in enumerate(seg_result.boxes.cls):
                    cls_name = model_seg.names[int(cls_id)]
                    if cls_name == segmentation_label:
                        mask = seg_result.masks.data[i].cpu().numpy()
                        if np.sum(mask) > 10:  # ambang batas area mask
                            vote_status.append(paslon)

        if len(vote_status) == 1:
            vote_result = vote_status[0]

    # Tambahkan ke database
    if vote_result == 'paslon1':
        session.execute(paslon1.insert().values(jumlah=1))
    elif vote_result == 'paslon2':
        session.execute(paslon2.insert().values(jumlah=1))
    elif vote_result == 'paslon3':
        session.execute(paslon3.insert().values(jumlah=1))
    else:
        session.execute(golput.insert().values(jumlah=1))
    session.commit()

    # Simpan gambar
    save_dir = 'static/captures'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'capture_{timestamp}.jpg'
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)

    return redirect(url_for('index', result=vote_result))

# 🔁 Video Streaming Tanpa Deteksi
def generate_frames():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, -1)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    if camera_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera is off"

@app.route('/camera_control', methods=['POST'])
def camera_control():
    global camera_active
    if 'start' in request.form:
        camera_active = True
    elif 'stop' in request.form:
        camera_active = False
    return redirect(url_for('index'))

@app.route('/get_counts')
def get_counts():
    count1 = session.query(func.sum(paslon1.c.jumlah)).scalar() or 0
    count2 = session.query(func.sum(paslon2.c.jumlah)).scalar() or 0
    count3 = session.query(func.sum(paslon3.c.jumlah)).scalar() or 0
    count4 = session.query(func.sum(golput.c.jumlah)).scalar() or 0
    return jsonify({'paslon1': count1, 'paslon2': count2, 'paslon3': count3, 'golput': count4})

from werkzeug.utils import secure_filename

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return 'No file uploaded', 400

        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        os.makedirs('static/uploads', exist_ok=True)
        file.save(filepath)

        # Load gambar yang diunggah
        frame = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_det_vis = img_rgb.copy()

        # Inisialisasi deteksi objek
        det_results = model_det(img_rgb)[0]
        paslon_boxes = {}
        lubang_luar_terdeteksi = False

        for box in det_results.boxes:
            cls_id = int(box.cls)
            cls_name = model_det.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            if cls_name in paslon_labels:
                paslon_boxes[cls_name] = (x1, y1, x2, y2)
            elif cls_name == lubang_luar_label:
                lubang_luar_terdeteksi = True

        vote_result = "golput"
        vote_status = []

        if not lubang_luar_terdeteksi:
            for paslon, (x1, y1, x2, y2) in paslon_boxes.items():
                cropped_img = frame[y1:y2, x1:x2]
                seg_result = model_seg(cropped_img)[0]

                if seg_result.masks is not None:
                    for i, cls_id in enumerate(seg_result.boxes.cls):
                        cls_name = model_seg.names[int(cls_id)]
                        if cls_name == segmentation_label:
                            mask = seg_result.masks.data[i].cpu().numpy()
                            if np.sum(mask) > 10:
                                vote_status.append(paslon)

            if len(vote_status) == 1:
                vote_result = vote_status[0]

        # Simpan ke folder sesuai hasil
        if vote_result == 'paslon1':
            session.execute(paslon1.insert().values(jumlah=1))
        elif vote_result == 'paslon2':
            session.execute(paslon2.insert().values(jumlah=1))
        elif vote_result == 'paslon3':
            session.execute(paslon3.insert().values(jumlah=1))
        else:
            session.execute(golput.insert().values(jumlah=1))
        session.commit()

        save_folder = 'static/suara_sah' if vote_result.startswith('paslon') else 'static/suara_tidaksah'
        os.makedirs(save_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(save_folder, f'{vote_result}_{timestamp}.jpg')
        cv2.imwrite(save_path, frame)

        return render_template('upload_result.html', result=vote_result, image_path=save_path)

    return render_template('upload_form.html')


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
