import cv2
import time
import torch
import numpy as np
import mediapipe as mp
import threading

from depth_anything_v2.dpt import DepthAnythingV2

class DepthFaceTracker:
    def __init__(self, model, device, buffer_size=50):
        self.model = model.to(device).eval()
        self.device = device
        self.buffer_size = buffer_size
        self.current_depth_frame = None
        self.depth_img = None
        self.running = True
        self.lock = threading.Lock()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def background_depth(self):
        while self.running:
            with self.lock:
                if self.current_depth_frame is not None:
                    self.depth_img = self.model.infer_image(self.current_depth_frame)
                    # print(self.depth_img)
        return

    @staticmethod
    def normalized_landmarks_to_array(landmarks):
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])

    def get_face_bounding_box(self, landmarks, frame_width, frame_height):
        x_min = max(int(np.min(landmarks[:, 0]) * frame_width) - self.buffer_size, 0)
        x_max = min(int(np.max(landmarks[:, 0]) * frame_width) + self.buffer_size, frame_width)
        y_min = max(int(np.min(landmarks[:, 1]) * frame_height) - self.buffer_size, 0)
        y_max = min(int(np.max(landmarks[:, 1]) * frame_height) + self.buffer_size, frame_height)
        return x_min, x_max, y_min, y_max

    def run(self):
        # Setup background depth computation
        depth_thread = threading.Thread(target=self.background_depth)
        depth_thread.daemon = True
        depth_thread.start()

        # Start camera feed
        if torch.backends.mps.is_available():
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(0)

        # Initialize face mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

        # Process camera feed
        prev_frame_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_height, frame_width = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face mesh
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get face bounding box for depth estimation
                    # landmarks = self.normalized_landmarks_to_array(face_landmarks.landmark)
                    # x_min, x_max, y_min, y_max = self.get_face_bounding_box(landmarks, frame.shape[1], frame.shape[0])
                    # self.current_depth_frame = frame[y_min:y_max, x_min:x_max].copy()
                    self.current_depth_frame = frame

                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                    )

                # Extract iris positions
                left_iris = face_landmarks.landmark[474:478]
                right_iris = face_landmarks.landmark[469:473]

                left_iris_depth = left_iris[0].z
                right_iris_depth = right_iris[0].z
                
                if self.depth_img is not None:
                    depth = self.depth_img.copy()
                    
                    lid = depth[int(frame_width*left_iris[0].x), int(frame_height*left_iris[0].y)]
                    rid = depth[int(frame_width*right_iris[0].x), int(frame_height*right_iris[0].y)]
                    print(f"lid: {left_iris_depth} -- {lid} ")
                    print(f"rid: {right_iris_depth} -- {rid} ")

            # FPS
            fps = 1 / (time.time() - prev_frame_time)
            prev_frame_time = time.time()
            cv2.putText(frame, f"FPS: {int(fps)}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

            # Iris pos:
            # cv2.putText(frame, f"left iris depth: {left_iris_depth}", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)
            # cv2.putText(frame, f"right iris depth: {right_iris_depth}", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0), 2)

            # Display image
            cv2.imshow("Depth Face Tracker", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                self.running = False
                break

        depth_thread.join()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'mps' if torch.backends.mps.is_avialable() else DEVICE
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
    model.load_state_dict(torch.load('checkpoints/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))

    tracker = DepthFaceTracker(model, DEVICE)
    tracker.run()
