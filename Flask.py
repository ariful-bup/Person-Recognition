import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Load student data from Excel
excel_file_path = 'D:/Code/Code/Face_Project/Colleagues_Information.xlsx'
data = pd.read_excel(excel_file_path)

# Initialize YOLOv8 for face detection
model = YOLO('best.pt')  # Load the YOLOv8 model using Ultralytics

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        image = request.files['image'].read()
        # Process the image using YOLOv8 to detect faces
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference on the image
        results = model(image)
        
        # Extract class labels detected by YOLOv8
        detected_labels = []
        for result in results:
            for label in result.boxes.cls:
                detected_labels.append(int(label))
        
        print("Detected labels:", detected_labels)  # Debugging
        
        # Map class labels to corresponding names
        detected_names = [model.names[label] for label in detected_labels]
        print("Detected names:", detected_names)  # Debugging
        
        # Search for information in Excel file based on detected names
        recognized_students = []
        for name in detected_names:
            student_info = data[data['Name'] == name].to_dict('records')
            print(f"Student info for {name}: {student_info}")  # Debugging
            if student_info:
                recognized_students.extend(student_info)
        
        return jsonify({'recognized_students': recognized_students})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






