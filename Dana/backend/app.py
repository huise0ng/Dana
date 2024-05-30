from flask import Flask, request, jsonify
import torch
import pytesseract
from PIL import Image
import numpy as np

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov8', 'yolov8n', source='local', path='models/yolov8_weights.pt')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    file = request.files['image']
    image = Image.open(file.stream)
    image_np = np.array(image)
    results = model(image_np)
    boxes = results.xyxy[0].numpy()
    
    texts = []
    for box in boxes:
        x1, y1, x2, y2, _, _ = box
        cropped_img = image_np[int(y1):int(y2), int(x1):int(x2)]
        pil_img = Image.fromarray(cropped_img)
        text = pytesseract.image_to_string(pil_img)
        texts.append(text.strip())
    
    return jsonify({'texts': texts})

if __name__ == '__main__':
    app.run(debug=True)
