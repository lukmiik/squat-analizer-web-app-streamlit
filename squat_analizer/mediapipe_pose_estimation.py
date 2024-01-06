import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


class BaseMediaPipePoseEstimation:
    LEFT_SIDE_STRING: str = "left side"
    RIGHT_SIDE_STRING: str = "right side"
    side: str | None = None
    mp_draw = mp.solutions.drawing_utils  # type: ignore
    mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
    DEFAULT_STYLE = mp_drawing_styles.get_default_pose_landmarks_style()
    mp_pose = mp.solutions.pose  # type: ignore

    def __init__(self) -> None:
        self.pose = self.mp_pose.Pose()

    def get_pose_estimation_landmarks(
        self, frame: np.ndarray
    ) -> NormalizedLandmarkList:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb).pose_landmarks

    def draw_all_pose_landmarks(
        self, frame: np.ndarray, pose_landmarks: NormalizedLandmarkList
    ) -> None:
        self.mp_draw.draw_landmarks(
            frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )

    def draw_specific_landmarks(
        self,
        frame: np.ndarray,
        pose_landmarks: NormalizedLandmarkList,
        connections: list,
    ) -> None:
        self.mp_draw.draw_landmarks(frame, pose_landmarks, connections)

    def get_side_based_on_legs(self, pose_landmarks: NormalizedLandmarkList) -> str:
        landmarks = pose_landmarks.landmark  # type: ignore
        left_hip_visibility = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].visibility  # type: ignore
        left_knee_visibility = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].visibility  # type: ignore
        right_hip_visibility = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].visibility  # type: ignore
        right_knee_visibility = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].visibility  # type: ignore
        if (
            left_hip_visibility + left_knee_visibility
            >= right_hip_visibility + right_knee_visibility
        ):
            return self.LEFT_SIDE_STRING
        else:
            return self.RIGHT_SIDE_STRING

    def set_side_based_on_legs(self, pose_landmarks: NormalizedLandmarkList) -> None:
        landmarks = pose_landmarks.landmark  # type: ignore
        left_hip_visibility = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].visibility  # type: ignore
        left_knee_visibility = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].visibility  # type: ignore
        right_hip_visibility = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].visibility  # type: ignore
        right_knee_visibility = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].visibility  # type: ignore
        if (
            left_hip_visibility + left_knee_visibility
            >= right_hip_visibility + right_knee_visibility
        ):
            self.side = self.LEFT_SIDE_STRING
        else:
            self.side = self.RIGHT_SIDE_STRING
