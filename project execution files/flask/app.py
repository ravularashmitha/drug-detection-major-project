from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import pytesseract
import os
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/upload', methods=['POST'])
def detect():
    if request.method == "POST":
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'Uploads')

        # Ensure the upload folder exists
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)

        # Open image and run model predictions
        im1 = Image.open(filepath)
        results = model.predict(source=im1, save=True)

        print(results)
        return display()

@app.route('/display')
def display():
    folder_path = 'runs/detect'

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        return 'Detection folder not found.'

    # Get subfolders sorted by creation time
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return 'No detection results found.'

    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)

    # Get files in the latest subfolder
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return 'No files found in the latest detection folder.'

    # Find the latest valid image file
    valid_extensions = {'jpg', 'jpeg', 'png'}
    for file in sorted(files, key=lambda x: os.path.getctime(os.path.join(directory, x)), reverse=True):
        file_extension = file.rsplit('.', 1)[-1].lower()
        if file_extension in valid_extensions:
            return send_from_directory(directory, file)

    return 'No valid image files found.'

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
