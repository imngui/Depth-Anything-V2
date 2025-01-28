import cv2
import time
import torch
import numpy as np
import mediapipe as mp
import matplotlib
import threading

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Depth Estimation
cmap = matplotlib.colormaps.get_cmap('Spectral_r')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model.to(DEVICE).eval()

# Face Features
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

BUFFER_SIZE = 50  # Define the buffer size around the face in pixels

current_depth_frame = None
depth_img = None
depth_changed = False
running = True
def background_depth():
    while running:
        print(running)
        if current_depth_frame is not None:
            depth_img = model.infer_image(current_depth_frame)
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            # depth = depth.astype(np.uint8)
            # depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            # depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return

def normalized_landmarks_to_array(landmarks):
    """Convert mediapipe landmarks to a numpy array"""
    return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

def get_face_bounding_box(landmarks, frame_width, frame_height, buffer=0):
    """Calculate the bounding box for the face based on landmarks, with an optional buffer."""
    x_min = max(int(np.min(landmarks[:, 0]) * frame_width) - buffer, 0)
    x_max = min(int(np.max(landmarks[:, 0]) * frame_width) + buffer, frame_width)
    y_min = max(int(np.min(landmarks[:, 1]) * frame_height) - buffer, 0)
    y_max = min(int(np.max(landmarks[:, 1]) * frame_height) + buffer, frame_height)

    return x_min, x_max, y_min, y_max


def main():
    depth_thread = threading.Thread(target=background_depth)
    depth_thread.daemon = True
    depth_thread.start()

    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                )
            landmarks = normalized_landmarks_to_array(face_landmarks.landmark)
            left_iris = face_landmarks.landmark[474:478]
            right_iris = face_landmarks.landmark[469:473]

            left_iris_depth = left_iris[0].z
            right_iris_depth = right_iris[0].z

            fbb_x_min, fbb_x_max, fbb_y_min, fbb_y_max = get_face_bounding_box(landmarks, frame_width, frame_height, buffer=BUFFER_SIZE)
            # current_depth_frame = frame[fbb_y_min:fbb_y_max, fbb_x_min:fbb_x_max].copy()

        img = frame
        
        new_frame_time = time.time()

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)
        cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.putText(img, f"left iris depth: {left_iris_depth}", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, f"right iris depth: {right_iris_depth}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

        cv2.imshow("Eye Tracker", img)

        if cv2.waitKey(5) & 0xFF == 27:
            running = False
            break

    depth_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()