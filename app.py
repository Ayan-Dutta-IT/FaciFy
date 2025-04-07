import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from deepface import DeepFace
from ultralytics import YOLO
import csv
import tempfile
import gdown
import zipfile

def download_and_extract_database():
    if not os.path.exists("database"):
        print("⏳ Downloading student database...")
        url = "https://drive.google.com/uc?id=1abcDEFghiJKLmnOPQRstuVWXYZ"
        output = "database.zip"
        gdown.download(url, output, quiet=False)

        print("📦 Extracting...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")

        print("✅ Database ready!")

download_and_extract_database()


# --- Configuration ---
database_folder = r"D:\Project\project\db_images"
attendance_folder = r"D:\Project\project\attendance"
reference_images_folder = r"D:\Project\project\uploads"
yolo_model = YOLO(r"D:/Project/project/yolo_objRecognition.pt")

os.makedirs(attendance_folder, exist_ok=True)

# --- Function: Detect faces with YOLO ---
@st.cache_data(show_spinner=False)
def detect_faces_and_annotate(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    detected_faces = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        face = img[y1:y2, x1:x2]
        detected_faces.append(face)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'Face {i+1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return detected_faces, img

# --- Function: Recognize a face ---
def recognize_face(face):
    try:
        face_pil = Image.fromarray(face)
        temp_face_path = os.path.join(tempfile.gettempdir(), "temp_face.jpg")
        face_pil.save(temp_face_path)

        result_vgg = DeepFace.find(img_path=temp_face_path, db_path=database_folder,
                                   model_name='VGG-Face', enforce_detection=False)
        result_arc = DeepFace.find(img_path=temp_face_path, db_path=database_folder,
                                   model_name='ArcFace', enforce_detection=False)

        recognized_label_vgg = result_vgg[0]['identity'][0] if len(result_vgg[0]) > 0 else None
        recognized_label_arc = result_arc[0]['identity'][0] if len(result_arc[0]) > 0 else None

        final_label = recognized_label_vgg if recognized_label_vgg else recognized_label_arc

        if final_label:
            return os.path.basename(os.path.dirname(final_label))
    except:
        return None

# --- Function: Save attendance ---
def mark_attendance(recognized_names):
    csv_path = os.path.join(attendance_folder, "attendance.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Student Name"])
        for name in recognized_names:
            writer.writerow([name])
    return csv_path

# --- Streamlit UI ---
st.set_page_config(page_title="Face Attendance System", layout="centered")
st.title("📸 Face Attendance System (Multi-Angle View)")

# --- Sample Download Section ---
st.subheader("📥 Download Sample Image Templates")

sample_cols = st.columns(3)
sample_files = {
    "Front View": "front.jpg",
    "Left View": "left.jpg",
    "Right View": "right.jpg"
}

for idx, (label, filename) in enumerate(sample_files.items()):
    file_path = os.path.join(reference_images_folder, filename)
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    sample_cols[idx].download_button(
        label=f"Download {label}",
        data=file_bytes,
        file_name=filename,
        mime="image/jpeg"
    )

st.markdown("---")

# --- Upload Section ---
st.subheader("📤 Upload Captured Images (Front, Left, Right)")

col1, col2, col3 = st.columns(3)
with col1:
    front_image = st.file_uploader("Front View", type=["jpg", "png", "jpeg"], key="front")
with col2:
    left_image = st.file_uploader("Left View", type=["jpg", "png", "jpeg"], key="left")
with col3:
    right_image = st.file_uploader("Right View", type=["jpg", "png", "jpeg"], key="right")

uploaded_images = [("Front", front_image), ("Left", left_image), ("Right", right_image)]

# If all images are uploaded
if all(img for _, img in uploaded_images):

    if 'recognized_names' not in st.session_state:
        temp_paths = []

        st.markdown("### 🖼️ Uploaded Image Previews")
        cols = st.columns(3)
        for idx, (label, uploaded) in enumerate(uploaded_images):
            temp_img_path = os.path.join(tempfile.gettempdir(), f"{label.lower()}_{uploaded.name}")
            with open(temp_img_path, "wb") as f:
                f.write(uploaded.read())
            temp_paths.append(temp_img_path)
            with cols[idx]:
                st.image(temp_img_path, caption=f"{label} View", use_container_width=True)

        with st.spinner("Detecting faces using YOLO from all views..."):
            all_faces = []
            for img_path in temp_paths:
                faces, annotated = detect_faces_and_annotate(img_path)
                all_faces.extend(faces)
                st.image(annotated, channels="BGR", caption=f"Detected Faces ({os.path.basename(img_path)})", use_container_width=True)

        if len(all_faces) == 0:
            st.warning("No faces detected in any view.")
        else:
            with st.spinner("Recognizing faces and marking attendance..."):
                recognized_names = set()
                for face in all_faces:
                    label = recognize_face(face)
                    if label:
                        recognized_names.add(label)

            if recognized_names:
                st.session_state['recognized_names'] = list(recognized_names)
                csv_path = mark_attendance(recognized_names)
                st.session_state['csv_path'] = csv_path
                st.success(f"✅ Attendance marked for {len(recognized_names)} students.")
                st.download_button("📄 Download Attendance CSV", data=open(csv_path, "rb"), file_name="attendance.csv", mime="text/csv")
            else:
                st.error("❌ No known faces recognized.")
    else:
        st.info("✅ Attendance already marked. Refresh the page to reprocess.")
