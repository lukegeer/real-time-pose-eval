import cv2
import numpy as np
import pandas as pd
import mediapipe as mp



class MediaPipePose:
    def __init__(self, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 static_image_mode=False,
                 model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe pose landmark indices
        self.landmark_names = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
    
    def detect_landmarks(self, frame, confidence_threshold=0.3):
        """
        Return both image-space (pose_landmarks) and world-space (pose_world_landmarks)
        as (33, 4) arrays [x, y, z, visibility]. Image coords are normalized to [0,1].
        Landmarks below the confidence threshold are set to NaN with visibility 0 to avoid
        downstream top-left jumps when detections drop.
        """
        results = self.pose.process(frame)

        kp_img = np.full((33, 4), np.nan, dtype=float)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                vis = lm.visibility
                if vis >= confidence_threshold:
                    kp_img[i] = (lm.x, lm.y, lm.z, vis)
                else:
                    kp_img[i] = (np.nan, np.nan, np.nan, 0.0)
        else:
            kp_img[:, 3] = 0.0

        return kp_img


    aist17_to_mediapipe = [
        0,   # nose -> NOSE
        2,   # left_eye -> LEFT_EYE
        5,   # right_eye -> RIGHT_EYE
        7,   # left_ear -> LEFT_EAR
        8,   # right_ear -> RIGHT_EAR
        11,  # left_shoulder -> LEFT_SHOULDER
        12,  # right_shoulder -> RIGHT_SHOULDER
        13,  # left_elbow -> LEFT_ELBOW
        14,  # right_elbow -> RIGHT_ELBOW
        15,  # left_wrist -> LEFT_WRIST
        16,  # right_wrist -> RIGHT_WRIST
        23,  # left_hip -> LEFT_HIP
        24,  # right_hip -> RIGHT_HIP
        25,  # left_knee -> LEFT_KNEE
        26,  # right_knee -> RIGHT_KNEE
        27,  # left_ankle -> LEFT_ANKLE
        28   # right_ankle -> RIGHT_ANKLE
    ]

    def convert_to_aist17(self, kp33):

        kp17 = np.zeros((17, 4), dtype=kp33.dtype)
        for i, mp_idx in enumerate(self.aist17_to_mediapipe):
            x, y, z, conf = kp33[mp_idx]
            kp17[i] = [x, y, z, conf]
        return kp17
        

    
