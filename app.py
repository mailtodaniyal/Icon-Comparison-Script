from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import shutil
import zipfile
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
TEST_DIR = os.path.join(UPLOAD_FOLDER, 'test')
LABELLED_DIR = os.path.join(UPLOAD_FOLDER, 'labelled')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_directories():
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(LABELLED_DIR, exist_ok=True)

def save_or_unzip(file, target_dir):
    filename = secure_filename(file.filename)
    if filename.lower().endswith('.zip'):
        temp_path = os.path.join(target_dir, filename)
        file.save(temp_path)
        with zipfile.ZipFile(temp_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(temp_path)
    else:
        file.save(os.path.join(target_dir, filename))

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def compare_images(img1, img2):
    return cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]

def find_best_matches():
    results = []
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    label_files = [f for f in os.listdir(LABELLED_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for test_file in test_files:
        test_path = os.path.join(TEST_DIR, test_file)
        test_img = preprocess_image(test_path)
        best_score = -1
        best_match = ''
        for label_file in label_files:
            label_path = os.path.join(LABELLED_DIR, label_file)
            label_img = preprocess_image(label_path)
            score = compare_images(test_img, label_img)
            if score > best_score:
                best_score = score
                best_match = label_file
        results.append({'test_icon': test_file, 'best_match': best_match, 'score': round(float(best_score), 4)})
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    prepare_directories()
    test_files = request.files.getlist('test_icons')
    labelled_files = request.files.getlist('labelled_icons')
    for file in test_files:
        if file and allowed_file(file.filename):
            save_or_unzip(file, TEST_DIR)
    for file in labelled_files:
        if file and allowed_file(file.filename):
            save_or_unzip(file, LABELLED_DIR)
    results = find_best_matches()
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
