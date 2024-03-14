from collections import deque
import av
from av.container.output import OutputContainer

import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from scipy.spatial.distance import cdist
from streamlit.delta_generator import DeltaGenerator as stDeltaGenerator

from squat_analizer.byte_track_object_tracking import BaseByteTrackObjectTracking
from squat_analizer.mediapipe_pose_estimation import BaseMediaPipePoseEstimation
from squat_analizer.object_detection import BaseVideoObjectDetection


class SquatAnalizer(BaseVideoObjectDetection, BaseMediaPipePoseEstimation):
    DEFAULT_YOLOV8_MODEL: str = 'data/human_detection/yolov8s.pt'
    BARBELL_YOLOV8_MODEL: str = (
        'data/barbell_detection/runs/detect/train/weights/best.pt'
    )
    HUMAN_CLASS_ID: int = 0
    DESIRED_VIDEO_HEIGHT: int = 640
    DEFAULT_FPS: int = 30
    MAX_HUMAN_TO_BARBELL_DISTANCE_MULTIPLIER: float = 1.5
    ECCENTRIC_CHANGE_Y_MARGIN_PERCENTAGE: float = 0.03
    CONCETRIC_CHANGE_Y_MARGIN_PERCENTAGE: float = 0.005
    N_OF_FRAMES_TO_CHANGE_TRACKED_HUMAN: int = 10
    N_OF_FRAMES_TO_CHANGE_TRACKED_BARBELL: int = 10
    N_OF_FRAMES_TO_CHANGE_TRACKED_HUMAN_HIGHEST_CONF: int = 5
    N_OF_FRAMES_FEET_MOVING_LIMIT: int = 3
    FEET_MOVEMENT_DISTANCE_DIVIDER: int = 8
    HIP_KNEE_ANGLE_CURRENT_REP_LOWER_LIMIT: int = -20
    HIP_KNEE_ANGLE_CURRENT_REP_UPPER_LIMIT: int = 30
    HIP_KNEE_ANGLE_SQUAT_DEPTH_LOWER_LIMIT: int = -20
    HIP_KNEE_ANGLE_SQUAT_DEPTH_UPPER_LIMIT: int = 5
    N_OF_FRAMES_ASSIGN_TO_ECCENTRIC_FROM_PAUSE: int = 10
    STARTING_HEIGHT_MULTIPLIER: float = 3
    PAUSE_STATE: str = 'pause_state'
    ECCENTRIC_PHASE_STATE: str = 'eccentric_phase_state'
    CONCENTRIC_PHASE_STATE: str = 'concentric_phase_state'
    ECCENTRIC_PHASE_BARBELL_PATH_COLOR: tuple[int, int, int] = (255, 255, 255)
    CONCENTRIC_PHASE_BARBELL_PATH_COLOR: tuple[int, int, int] = (0, 0, 255)
    SQUAT_DEPTH_TRUE_COLOR: tuple[int, int, int] = (0, 255, 0)
    SQUAT_DEPTH_FALSE_COLOR: tuple[int, int, int] = (0, 0, 255)
    video_writer: cv2.VideoWriter | None = None
    human_byte_tracker: BaseByteTrackObjectTracking
    barbell_byte_tracker: BaseByteTrackObjectTracking
    starting_height: float
    eccentric_change_y_margin: float
    concentric_change_y_margin: float
    hip_knee_angle: int
    previous_barbell_assigned_to_main_human_tracker_id: int
    main_human_bbox: np.ndarray
    main_barbell_center_y: int
    barbell_assigned_to_main_human_bbox: np.ndarray

    def __init__(
        self,
        video_path: str,
        fps: int | None = None,
        save_video: bool = False,
        save_video_path: str | None = None,
        models: list[str] = [DEFAULT_YOLOV8_MODEL, BARBELL_YOLOV8_MODEL],
    ) -> None:
        BaseVideoObjectDetection.__init__(self, video_path, models)
        BaseMediaPipePoseEstimation.__init__(self)
        if fps is None:
            self.fps = int(self.get_fps()) or self.DEFAULT_FPS
        if save_video and save_video_path is not None:
            aspect_ratio = self.video.get(cv2.CAP_PROP_FRAME_WIDTH) / self.video.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            desired_width = int((self.DESIRED_VIDEO_HEIGHT * aspect_ratio))
            self.video_writer = cv2.VideoWriter(
                save_video_path,
                cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
                self.fps,
                (desired_width, self.DESIRED_VIDEO_HEIGHT),
            )
        self.human_byte_tracker = BaseByteTrackObjectTracking(self.fps)
        self.barbell_byte_tracker = BaseByteTrackObjectTracking(self.fps)
        self.human_model = self.models[0]
        self.barbell_model = self.models[1]
        self.stop_printing_yolo_logs()

    def init_squat_stats(self) -> None:
        self.current_state: str = self.PAUSE_STATE
        self.current_rep: bool = False
        self.n_of_reps: int = 0
        self.depth: bool = False
        self.barbell_path_eccentric_phase_points: list[tuple[int, int]] = []
        self.barbell_path_concentric_phase_points: list[tuple[int, int]] = []
        self.previous_feet_pos: tuple[str, float, float] | None = None
        self.n_of_frames_feet_moving: int = 0
        self.lowest_main_barbell_center_y: float | None = None
        self.last_depth_data: tuple[int, int, int, int, tuple[int, int, int]] | None = (
            None
        )
        self.main_human_roi: np.ndarray | None = None
        self.finish_rep_flag: bool = False
        self.squat_data: dict[int, list] = {}
        self.n_of_frames: int = 0
        self.barbell_bboxes_from_pause_to_eccentric: deque[np.ndarray] = deque(
            maxlen=self.N_OF_FRAMES_ASSIGN_TO_ECCENTRIC_FROM_PAUSE
        )

    def init_tracker_variables(self) -> None:
        self.main_human_tracker_id: int | None = None
        self.n_of_frames_same_tracker_id: int = 0
        self.n_of_frames_highest_conf_human_barbell_tracker_id: int = 0
        self.previous_tracker_ids: dict[int, int] = {}
        self.n_of_frames_previous_barbell_assigned_to_main_human_tracker_id = 0

    def init_humans_barbells_variables(self) -> None:
        self.humans_bboxes: list[np.ndarray] = []
        self.humans_conf_scores: list[np.float32] = []
        self.humans_points: list[tuple[int, int]] = []
        self.barbell_bboxes: list[np.ndarray] = []
        self.barbells_conf_scores: list[np.float32] = []
        self.barbells_points: list[tuple[int, int]] = []

    def detect_humans(self, frame: np.ndarray) -> list[np.ndarray]:
        return self.get_results(self.human_model, frame, self.HUMAN_CLASS_ID)

    def detect_barbells(self, frame: np.ndarray) -> list[np.ndarray]:
        return self.get_results(self.barbell_model, frame)

    def set_humans_data(self, humans: list[np.ndarray]) -> None:
        for human in humans:
            human_bbox, human_conf_score = self.get_bbox(human), self.get_conf_score(
                human
            )
            human_bbox_coord_for_analysis = self.get_human_bbox_coord_for_analysis(
                human_bbox
            )
            self.humans_bboxes.append(human_bbox)
            self.humans_conf_scores.append(human_conf_score)
            self.humans_points.append(human_bbox_coord_for_analysis)

    def set_barbells_data(self, barbells: list[np.ndarray]) -> None:
        for barbell in barbells:
            barbell_bbox, barbell_conf_score = self.get_bbox(
                barbell
            ), self.get_conf_score(barbell)
            barbell_bbox_center = self.get_coord_center_bbox(barbell_bbox)
            self.barbell_bboxes.append(barbell_bbox)
            self.barbells_conf_scores.append(barbell_conf_score)
            self.barbells_points.append(barbell_bbox_center)

    def get_human_bbox_coord_for_analysis(self, bbox: np.ndarray) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox[0]
        height = y2 - y1
        new_y = y1 + height // 10
        return (x1 + x2) // 2, new_y

    def assign_humans_to_barbells(
        self,
        points1: list[tuple[int, int]],
        points2: list[tuple[int, int]],
        human_bboxes: list[np.ndarray],
    ) -> list[tuple[int, int]]:
        human_bboxes_widths = self.get_bboxes_widths(human_bboxes)
        indices_min_distances = []
        np_points1, np_points2 = np.array(points1), np.array(points2)
        distances = cdist(np_points1, np_points2)
        while not np.min(distances) >= np.inf:
            min_distance = np.min(distances)
            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            if (
                min_distance
                < human_bboxes_widths[min_index[0]]
                * self.MAX_HUMAN_TO_BARBELL_DISTANCE_MULTIPLIER
            ):
                distances[:, min_index[1]] = np.inf
                indices_min_distances.append(min_index)
            distances[min_index[0], :] = np.inf
        return indices_min_distances

    def get_bboxes_widths(self, bboxes: list[np.ndarray]) -> list[float]:
        return [bbox[0][2] - bbox[0][0] for bbox in bboxes]

    def get_bboxes_humans_with_barbells(
        self,
        humans_bboxes: list[np.ndarray],
        humans_barbells_pairs: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        return [humans_bboxes[pair[0]] for pair in humans_barbells_pairs]

    def get_bboxes_barbells_of_humans(
        self,
        barbells_bboxes: list[np.ndarray],
        humans_barbells_pairs: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        return [barbells_bboxes[pair[1]] for pair in humans_barbells_pairs]

    def get_conf_scores_humans_with_barbells(
        self,
        conf_scores: list[np.float32],
        humans_barbells_pairs: list[tuple[int, int]],
    ) -> list[np.float32]:
        return [conf_scores[pair[0]] for pair in humans_barbells_pairs]

    def get_conf_scores_barbells_of_humans(
        self,
        conf_scores: list[np.float32],
        humans_barbells_pairs: list[tuple[int, int]],
    ) -> list[np.float32]:
        return [conf_scores[pair[1]] for pair in humans_barbells_pairs]

    def get_human_barbell_highest_conf_tracker_id(
        self,
        human_bboxes_idxs: list[int],
        humans_conf_scores: list[np.float32],
        barbells_conf_scores: list[np.float32],
        human_tracker_ids: list[int],
    ) -> int:
        human_barbell_highest_conf_idx = self.get_human_barbell_highest_conf_idx(
            human_bboxes_idxs,
            humans_conf_scores,
            barbells_conf_scores,
        )
        human_barbell_highest_conf_pair_human_idx = human_bboxes_idxs.index(
            human_barbell_highest_conf_idx
        )
        return human_tracker_ids[human_barbell_highest_conf_pair_human_idx]

    def get_human_barbell_highest_conf_idx(
        self,
        human_bboxes_idxs: list[int],
        humans_conf_scores: list[np.float32],
        barbells_conf_scores: list[np.float32],
    ) -> int:
        highest_combined_conf_score = -1
        for idx in human_bboxes_idxs:
            human_conf_score = humans_conf_scores[idx]
            barbell_conf_score = barbells_conf_scores[idx]
            combined_conf_score = human_conf_score + barbell_conf_score  # type: ignore
            if combined_conf_score > highest_combined_conf_score:
                highest_combined_conf_score = combined_conf_score
                highest_conf_idx = idx
        return highest_conf_idx  # type: ignore

    def check_main_human_tracker_id(self, human_tracker_ids: list[int]) -> None:
        '''Checks if main_human_tracker_id should change. It should change if current main_human_tracker_id is no longer visible on the screen for N_OF_FRAMES_TO_CHANGE_TRACKED_HUMAN frames'''
        if self.main_human_tracker_id in human_tracker_ids:
            self.previous_tracker_ids = {}
            self.n_of_frames_same_tracker_id = 0
        elif any(
            tracker_id in human_tracker_ids
            for tracker_id in self.previous_tracker_ids.keys()
        ):
            self.n_of_frames_same_tracker_id += 1
            previous_tracker_ids_keys = list(self.previous_tracker_ids.keys())
            for tracker_id in human_tracker_ids:
                if tracker_id in previous_tracker_ids_keys:
                    self.previous_tracker_ids[tracker_id] += 1
                else:
                    self.previous_tracker_ids[tracker_id] = 1
            previous_tracker_ids_to_delete = [
                tracker_id
                for tracker_id in previous_tracker_ids_keys
                if tracker_id not in human_tracker_ids
            ]
            for tracker_id_to_delete in previous_tracker_ids_to_delete:
                del self.previous_tracker_ids[tracker_id_to_delete]
            if new_main_human_tracker_id := self.get_new_tracker_id(
                self.previous_tracker_ids
            ):
                self.main_human_tracker_id = new_main_human_tracker_id
                self.reset_stats_new_main_human_tracker_id()
        else:
            for id in human_tracker_ids:
                self.previous_tracker_ids[id] = 1
            self.n_of_frames_same_tracker_id = 0

    def check_main_human_tracker_id_based_on_highest_conf(
        self,
        human_barbell_highest_conf_human_tracker_id: int,
    ) -> None:
        if human_barbell_highest_conf_human_tracker_id != self.main_human_tracker_id:
            self.n_of_frames_highest_conf_human_barbell_tracker_id += 1
            if (
                self.n_of_frames_highest_conf_human_barbell_tracker_id
                >= self.N_OF_FRAMES_TO_CHANGE_TRACKED_HUMAN_HIGHEST_CONF
            ):
                self.main_human_tracker_id = human_barbell_highest_conf_human_tracker_id
                self.reset_stats_new_main_human_tracker_id()
        else:
            self.n_of_frames_highest_conf_human_barbell_tracker_id = 0

    def reset_stats_new_main_human_tracker_id(self) -> None:
        self.n_of_frames_highest_conf_human_barbell_tracker_id = 0
        self.n_of_frames_same_tracker_id = 0
        self.previous_tracker_ids = {}
        self.current_rep = False
        self.depth = False
        self.barbell_path_concentric_phase_points = []
        self.barbell_path_eccentric_phase_points = []
        self.lowest_main_barbell_center_y = None
        self.current_state = self.PAUSE_STATE
        self.n_of_reps = 0

    def get_new_tracker_id(self, tracker_ids: dict) -> int | None:
        for tracker_id, n_of_frames in tracker_ids.items():
            if n_of_frames >= self.N_OF_FRAMES_TO_CHANGE_TRACKED_HUMAN:
                return tracker_id
        return None

    def get_main_barbell_tracker_id(
        self,
        main_human_tracker_id: int,
        human_tracker_ids: list[int],
        barbell_tracker_ids: list[int],
    ) -> int:
        main_human_tracker_idx = self.get_main_tracker_idx(
            main_human_tracker_id, human_tracker_ids
        )
        return barbell_tracker_ids[main_human_tracker_idx]

    def check_main_barbell_tracker_id(
        self, human_tracker_ids: list[int], barbell_tracker_ids: list[int]
    ) -> None:
        current_barbell_assigned_to_main_human = self.get_main_barbell_tracker_id(
            self.main_human_tracker_id, human_tracker_ids, barbell_tracker_ids  # type: ignore
        )
        if self.main_barbell_tracker_id == current_barbell_assigned_to_main_human:
            self.previous_barbell_assigned_to_main_human_tracker_id = (
                current_barbell_assigned_to_main_human
            )
            self.n_of_frames_previous_barbell_assigned_to_main_human_tracker_id = 0
        elif (
            self.previous_barbell_assigned_to_main_human_tracker_id
            == current_barbell_assigned_to_main_human
        ):
            self.n_of_frames_previous_barbell_assigned_to_main_human_tracker_id += 1
            if self.n_of_frames_previous_barbell_assigned_to_main_human_tracker_id >= (
                self.N_OF_FRAMES_TO_CHANGE_TRACKED_BARBELL
            ):
                self.main_barbell_tracker_id = current_barbell_assigned_to_main_human
                self.n_of_frames_previous_barbell_assigned_to_main_human_tracker_id = 0
        else:
            self.previous_barbell_assigned_to_main_human_tracker_id = (
                current_barbell_assigned_to_main_human
            )
            self.n_of_frames_previous_barbell_assigned_to_main_human_tracker_id = 1

    def get_main_tracker_idx(self, main_tracker_id: int, tracker_ids: list[int]) -> int:
        return tracker_ids.index(main_tracker_id)

    def get_main_bbox(
        self,
        main_tracker_id: int,
        tracker_ids: list[int],
        bboxes_idxs: list[int],
        bboxes: list[np.ndarray],
    ) -> np.ndarray:
        main_tracker_idx = self.get_main_tracker_idx(main_tracker_id, tracker_ids)
        main_detection_idx = bboxes_idxs[main_tracker_idx]
        return bboxes[main_detection_idx]

    def get_human_roi(
        self,
        frame: np.ndarray,
        human_bbox: np.ndarray,
    ) -> np.ndarray:
        main_human_roi_x1 = int(human_bbox[0][0])
        main_human_roi_y1 = int(human_bbox[0][1])
        main_human_roi_x2 = int(human_bbox[0][2])
        main_human_roi_y2 = int(human_bbox[0][3])
        return frame[
            main_human_roi_y1:main_human_roi_y2,
            main_human_roi_x1:main_human_roi_x2,
        ]

    def have_feet_moved(
        self,
        human_roi_frame: np.ndarray,
        human_roi_x1: float,
        pose_landmarks: NormalizedLandmarkList,
    ) -> bool:
        landmarks = pose_landmarks.landmark  # type: ignore
        if self.side == self.LEFT_SIDE_STRING:
            foot_heel_x = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].x
            foot_index_x = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
        elif self.side == self.RIGHT_SIDE_STRING:
            foot_heel_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL].x
            foot_index_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
        else:
            return False
        human_roi_frame_width = human_roi_frame.shape[1]
        foot_heel_x = foot_heel_x * human_roi_frame_width + human_roi_x1
        foot_index_x = foot_index_x * human_roi_frame_width + human_roi_x1
        if self.previous_feet_pos is None or self.side != self.previous_feet_pos[0]:
            self.previous_feet_pos = (self.side, foot_heel_x, foot_index_x)  # type: ignore
            self.n_of_frames_feet_moving = 0
            return False
        distance_limit = human_roi_frame_width / self.FEET_MOVEMENT_DISTANCE_DIVIDER
        if (
            abs(foot_heel_x - self.previous_feet_pos[1]) > distance_limit
            and abs(foot_index_x - self.previous_feet_pos[2]) > distance_limit
        ):
            self.n_of_frames_feet_moving += 1
            if self.n_of_frames_feet_moving >= self.N_OF_FRAMES_FEET_MOVING_LIMIT:
                self.n_of_frames_feet_moving = 0
                self.previous_feet_pos = (self.side, foot_heel_x, foot_index_x)  # type: ignore
                return True
            else:
                return False
        self.n_of_frames_feet_moving = 0
        self.previous_feet_pos = (self.side, foot_heel_x, foot_index_x)  # type: ignore
        return False

    def get_feet_y(
        self,
        human_roi_frame: np.ndarray,
        human_roi_y1: float,
        pose_landmarks: NormalizedLandmarkList,
    ) -> float:
        landmarks = pose_landmarks.landmark  # type: ignore
        if self.side is None:
            self.side = self.get_side_based_on_legs(pose_landmarks)
        if self.side == self.LEFT_SIDE_STRING:
            feet_y = (
                landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
                + landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].y
            ) / 2
        else:
            feet_y = (
                landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
                + landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL].y  # type: ignore
            ) / 2
        human_roi_frame_height = human_roi_frame.shape[0]
        feet_y = feet_y * human_roi_frame_height + human_roi_y1
        return round(feet_y, 2)

    def get_hip_knee_angle(
        self, human_roi_frame: np.ndarray, pose_landmarks: NormalizedLandmarkList
    ) -> int:
        landmarks = pose_landmarks.landmark  # type: ignore
        if self.side is None:
            self.side = self.get_side_based_on_legs(pose_landmarks)
        if self.side == self.LEFT_SIDE_STRING:
            x_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * -1
            y_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
            x_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x * -1
            y_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y
        else:
            x_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x
            y_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y
            x_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x
            y_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y
        human_roi_frame_width = human_roi_frame.shape[1]
        human_roi_frame_height = human_roi_frame.shape[0]
        x_knee = x_knee * human_roi_frame_width
        y_knee = y_knee * human_roi_frame_height
        x_hip = x_hip * human_roi_frame_width
        y_hip = y_hip * human_roi_frame_height
        delta_x = x_knee - x_hip
        delta_y = y_knee - y_hip
        return int(np.degrees(np.arctan(delta_y / delta_x)))

    def draw_squat_depth_info(
        self,
        human_roi_frame: np.ndarray,
        pose_landmarks: NormalizedLandmarkList | None,
    ) -> None:
        if pose_landmarks is None and self.last_depth_data is not None:
            hip_x, hip_y, knee_x, knee_y, color = self.last_depth_data
            self.draw_circle(human_roi_frame, hip_x, hip_y, color=color)
            self.draw_circle(human_roi_frame, knee_x, knee_y, color=color)
            self.draw_line(human_roi_frame, hip_x, hip_y, knee_x, knee_y, color)
        else:
            landmarks = pose_landmarks.landmark  # type: ignore
            if self.side == self.LEFT_SIDE_STRING:
                hip_x = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x
                hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
                knee_x = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x
                knee_y = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y
            elif self.side == self.RIGHT_SIDE_STRING:
                hip_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x
                hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y
                knee_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x
                knee_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y
            else:
                return
            human_roi_frame_width, human_roi_frame_height = (
                human_roi_frame.shape[1],
                human_roi_frame.shape[0],
            )
            hip_x = int(hip_x * human_roi_frame_width)
            hip_y = int(hip_y * human_roi_frame_height)
            knee_x = int(knee_x * human_roi_frame_width)
            knee_y = int(knee_y * human_roi_frame_height)
            color = (
                self.SQUAT_DEPTH_TRUE_COLOR
                if self.depth
                else self.SQUAT_DEPTH_FALSE_COLOR
            )
            self.last_depth_data = (hip_x, hip_y, knee_x, knee_y, color)
            self.draw_circle(human_roi_frame, hip_x, hip_y, color=color)
            self.draw_circle(human_roi_frame, knee_x, knee_y, color=color)
            self.draw_line(human_roi_frame, hip_x, hip_y, knee_x, knee_y, color)

    def add_barbell_path_point(self, barbell_bbox: np.ndarray) -> None:
        barbell_center_x, barbell_center_y = self.get_coord_center_bbox(barbell_bbox)
        new_barbell_path_point = (int(barbell_center_x), int(barbell_center_y))
        if self.current_state == self.ECCENTRIC_PHASE_STATE:
            self.barbell_path_eccentric_phase_points.append(new_barbell_path_point)
        elif self.current_state == self.CONCENTRIC_PHASE_STATE:
            if not self.barbell_path_concentric_phase_points:
                self.barbell_path_concentric_phase_points.append(
                    self.barbell_path_eccentric_phase_points[-1]
                )
            self.barbell_path_concentric_phase_points.append(new_barbell_path_point)

    def draw_barbell_path(self, frame: np.ndarray) -> None:
        cv2.polylines(
            frame,
            [np.array(self.barbell_path_eccentric_phase_points)],
            isClosed=False,
            color=self.ECCENTRIC_PHASE_BARBELL_PATH_COLOR,
            thickness=2,
        )
        cv2.polylines(
            frame,
            [np.array(self.barbell_path_concentric_phase_points)],
            isClosed=False,
            color=self.CONCENTRIC_PHASE_BARBELL_PATH_COLOR,
            thickness=2,
        )

    def detect_and_update_squat_data(
        self, pose_landmarks: NormalizedLandmarkList
    ) -> None:
        if self.current_state == self.PAUSE_STATE:
            self.current_rep = False
            self.depth = False
            if self.lowest_main_barbell_center_y is None:
                if pose_landmarks:
                    self.set_y_margins_and_starting_height(pose_landmarks)
                self.barbell_bboxes_from_pause_to_eccentric.append(
                    self.barbell_assigned_to_main_human_bbox
                )
            elif self.main_barbell_center_y - self.lowest_main_barbell_center_y > self.eccentric_change_y_margin:  # type: ignore
                self.change_state_to_eccentric_phase()
            else:
                self.lowest_main_barbell_center_y = min(
                    self.lowest_main_barbell_center_y, self.main_barbell_center_y  # type: ignore
                )
                self.barbell_bboxes_from_pause_to_eccentric.append(
                    self.barbell_assigned_to_main_human_bbox
                )
        elif self.current_state == self.ECCENTRIC_PHASE_STATE:
            if (
                self.lowest_main_barbell_center_y - self.main_barbell_center_y  # type: ignore
                > self.concentric_change_y_margin
            ):
                self.check_barbell_going_up_during_eccentric_phase(pose_landmarks)
            elif pose_landmarks:
                self.check_hip_knee_angle(pose_landmarks)
            self.add_barbell_path_point(self.barbell_assigned_to_main_human_bbox)
            self.lowest_main_barbell_center_y = max(
                self.lowest_main_barbell_center_y,  # type: ignore
                self.main_barbell_center_y,
            )
        elif self.current_state == self.CONCENTRIC_PHASE_STATE:
            self.add_barbell_path_point(self.barbell_assigned_to_main_human_bbox)
            if self.finish_rep_flag and not self.lowest_main_barbell_center_y - self.main_barbell_center_y > self.concentric_change_y_margin:  # type: ignore
                self.finish_rep()
            elif not self.finish_rep_flag and self.main_barbell_center_y < self.starting_height:  # type: ignore
                self.finish_rep_flag = True
                self.n_of_reps += 1
            self.lowest_main_barbell_center_y = self.main_barbell_center_y

    def set_y_margins_and_starting_height(
        self, pose_landmarks: NormalizedLandmarkList
    ) -> None:
        self.feet_y = self.get_feet_y(
            self.main_human_roi,  # type: ignore
            self.main_human_bbox[0][0],
            pose_landmarks,
        )
        self.lowest_main_barbell_center_y = self.main_barbell_center_y
        self.eccentric_change_y_margin = (
            self.feet_y - self.main_barbell_center_y
        ) * self.ECCENTRIC_CHANGE_Y_MARGIN_PERCENTAGE
        self.concentric_change_y_margin = (
            self.feet_y - self.main_barbell_center_y
        ) * self.CONCETRIC_CHANGE_Y_MARGIN_PERCENTAGE
        self.starting_height = (
            self.main_barbell_center_y
            + self.STARTING_HEIGHT_MULTIPLIER * self.eccentric_change_y_margin
        )

    def change_state_to_eccentric_phase(self) -> None:
        self.eccentric_phase_start_frame = (
            self.n_of_frames - self.N_OF_FRAMES_ASSIGN_TO_ECCENTRIC_FROM_PAUSE
        )
        self.current_state = self.ECCENTRIC_PHASE_STATE
        self.barbell_path_concentric_phase_points = []
        self.barbell_path_eccentric_phase_points = []
        for barbell_bbox in self.barbell_bboxes_from_pause_to_eccentric:
            self.add_barbell_path_point(barbell_bbox)
        self.add_barbell_path_point(self.barbell_assigned_to_main_human_bbox)
        self.barbell_bboxes_from_pause_to_eccentric.clear()
        self.lowest_main_barbell_center_y = self.main_barbell_center_y

    def check_barbell_going_up_during_eccentric_phase(
        self, pose_landmarks: NormalizedLandmarkList
    ) -> None:
        if not self.current_rep:
            self.cancel_rep_barbell_going_up_during_eccentric_phase()
        else:
            self.change_state_to_concentric_phase(pose_landmarks)

    def cancel_rep_barbell_going_up_during_eccentric_phase(self) -> None:
        self.barbell_path_concentric_phase_points = []
        self.barbell_path_eccentric_phase_points = []
        self.current_state = self.PAUSE_STATE

    def change_state_to_concentric_phase(
        self, pose_landmarks: NormalizedLandmarkList
    ) -> None:
        if pose_landmarks:
            self.hip_knee_angle = self.get_hip_knee_angle(self.main_human_roi, pose_landmarks)  # type: ignore
            self.current_rep = True
            if (
                self.HIP_KNEE_ANGLE_SQUAT_DEPTH_LOWER_LIMIT
                <= self.hip_knee_angle
                <= self.HIP_KNEE_ANGLE_SQUAT_DEPTH_UPPER_LIMIT
            ):
                self.depth = True
        self.concentric_phase_start_frame = self.eccentric_phase_finish_frame = (
            self.n_of_frames
        )
        self.current_state = self.CONCENTRIC_PHASE_STATE

    def check_hip_knee_angle(self, pose_landmarks: NormalizedLandmarkList) -> None:
        self.hip_knee_angle = self.get_hip_knee_angle(self.main_human_roi, pose_landmarks)  # type: ignore
        if (
            not self.current_rep
            and self.HIP_KNEE_ANGLE_CURRENT_REP_LOWER_LIMIT
            <= self.hip_knee_angle
            <= self.HIP_KNEE_ANGLE_CURRENT_REP_UPPER_LIMIT
        ):
            self.current_rep = True
        if (
            self.HIP_KNEE_ANGLE_SQUAT_DEPTH_LOWER_LIMIT
            <= self.hip_knee_angle
            <= self.HIP_KNEE_ANGLE_SQUAT_DEPTH_UPPER_LIMIT
        ):
            self.depth = True

    def finish_rep(self) -> None:
        self.finish_rep_flag = False
        self.concentric_phase_finish_frame = self.n_of_frames
        self.barbell_path_points = []
        self.current_state = self.PAUSE_STATE
        self.add_rep_data_to_squat_data()

    def add_rep_data_to_squat_data(self) -> None:
        current_rep_dict = {
            'rep': self.n_of_reps,
            'depth': self.depth,
            'eccentric_time': round(
                (self.eccentric_phase_finish_frame - self.eccentric_phase_start_frame)
                / self.fps,
                2,
            ),
            'concentric_time': round(
                (self.concentric_phase_finish_frame - self.concentric_phase_start_frame)
                / self.fps,
                2,
            ),
        }
        if self.main_human_tracker_id not in self.squat_data:
            self.squat_data[self.main_human_tracker_id] = [current_rep_dict]  # type: ignore
        else:
            self.squat_data[self.main_human_tracker_id].append(current_rep_dict)

    def downscale_video(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] > self.DESIRED_VIDEO_HEIGHT:
            return self.downscale_frame_by_height(frame, self.DESIRED_VIDEO_HEIGHT)
        return frame

    def add_average_to_data(self, data: list[dict]) -> list[dict]:
        avg_eccentric_time = sum(d['eccentric_time'] for d in data) / len(data)
        avg_concentric_time = sum(d['concentric_time'] for d in data) / len(data)
        avg_dict = {
            'rep': 'avg',
            'depth': '',
            'eccentric_time': avg_eccentric_time,
            'concentric_time': avg_concentric_time,
        }
        data.append(avg_dict)
        return data

    def get_main_squat_data(self) -> list[dict]:
        if self.squat_data:
            main_squat_data_key = max(
                self.squat_data, key=lambda k: len(self.squat_data[k])
            )
            main_sqaut_data = self.squat_data[main_squat_data_key]
            main_squat_data = self.add_average_to_data(main_sqaut_data)
            return main_squat_data
        return []

    def analize_squat_streamlit(self, stream: OutputContainer, progress_bar: stDeltaGenerator, progress_text: str, output: OutputContainer) -> list[dict]:
        total_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.init_squat_stats()
        self.init_tracker_variables()
        while True:
            ret, frame = self.video.read()
            if ret:
                frame = self.downscale_video(frame)
                self.n_of_frames += 1
                humans = self.detect_humans(frame)
                if humans:
                    barbells = self.detect_barbells(frame)
                    if barbells:
                        self.init_humans_barbells_variables()
                        self.set_humans_data(humans)
                        self.set_barbells_data(barbells)
                        humans_barbells_pairs = self.assign_humans_to_barbells(
                            self.humans_points,
                            self.barbells_points,
                            self.humans_bboxes,
                        )
                        if humans_barbells_pairs:
                            humans_with_barbells_bboxes = (
                                self.get_bboxes_humans_with_barbells(
                                    self.humans_bboxes, humans_barbells_pairs
                                )
                            )
                            barbells_of_humans_bboxes = (
                                self.get_bboxes_barbells_of_humans(
                                    self.barbell_bboxes, humans_barbells_pairs
                                )
                            )
                            humans_with_barbells_conf_scores = (
                                self.get_conf_scores_humans_with_barbells(
                                    self.humans_conf_scores, humans_barbells_pairs
                                )
                            )
                            barbells_of_humans_conf_scores = (
                                self.get_conf_scores_barbells_of_humans(
                                    self.barbells_conf_scores,
                                    humans_barbells_pairs,
                                )
                            )
                            humans_detections_converted_to_byte_tracker_format = (
                                self.human_byte_tracker.convert_detections_data(
                                    humans_with_barbells_bboxes,
                                    humans_with_barbells_conf_scores,
                                )
                            )
                            human_byte_tracker_tracks = (
                                self.human_byte_tracker.update(
                                    humans_detections_converted_to_byte_tracker_format,
                                    (frame.shape[0], frame.shape[1]),
                                    (frame.shape[0], frame.shape[1]),
                                )
                            )
                            (
                                human_tracker_ids,
                                human_bboxes_idxs,
                            ) = self.human_byte_tracker.assign_ids_to_detections(
                                humans_with_barbells_bboxes,
                                human_byte_tracker_tracks,
                            )
                            if human_tracker_ids is not None:
                                human_barbell_highest_conf_human_tracker_id = self.get_human_barbell_highest_conf_tracker_id(
                                    human_bboxes_idxs,  # type: ignore
                                    humans_with_barbells_conf_scores,  # type: ignore
                                    barbells_of_humans_conf_scores,
                                    human_tracker_ids,
                                )
                                self.check_main_human_tracker_id(
                                    human_tracker_ids
                                )
                                self.check_main_human_tracker_id_based_on_highest_conf(
                                    human_barbell_highest_conf_human_tracker_id
                                )
                                if (
                                    self.main_human_tracker_id is not None
                                    and self.main_human_tracker_id
                                    in human_tracker_ids
                                ):
                                    self.barbell_assigned_to_main_human_bbox = (
                                        self.get_main_bbox(
                                            self.main_human_tracker_id,
                                            human_tracker_ids,
                                            human_bboxes_idxs,  # type: ignore
                                            barbells_of_humans_bboxes,
                                        )
                                    )
                                    self.main_barbell_center_y = self.get_coord_center_bbox(
                                        self.barbell_assigned_to_main_human_bbox
                                    )[
                                        1
                                    ]
                                    self.main_human_bbox = (
                                        self.get_main_bbox(
                                            self.main_human_tracker_id,
                                            human_tracker_ids,
                                            human_bboxes_idxs,  # type: ignore
                                            humans_with_barbells_bboxes,
                                        )
                                    )
                                    self.main_human_roi = (
                                        self.get_human_roi(
                                            frame, self.main_human_bbox
                                        )
                                    )
                                    pose_landmarks = (
                                        self.get_pose_estimation_landmarks(
                                            self.main_human_roi
                                        )
                                    )
                                    if pose_landmarks:
                                        self.set_side_based_on_legs(
                                            pose_landmarks
                                        )
                                        if self.have_feet_moved(
                                            self.main_human_roi,
                                            self.main_human_bbox[0][0],
                                            pose_landmarks,
                                        ):
                                            self.lowest_main_barbell_center_y = (
                                                None
                                            )
                                            self.current_state = (
                                                self.PAUSE_STATE
                                            )
                                    self.detect_and_update_squat_data(
                                        pose_landmarks
                                    )
                if (
                    self.current_state == self.ECCENTRIC_PHASE_STATE
                    or self.current_state == self.CONCENTRIC_PHASE_STATE
                ) and pose_landmarks:  # type: ignore
                    self.draw_squat_depth_info(
                        self.main_human_roi, # type: ignore
                        pose_landmarks,  # type: ignore
                    )
                self.draw_barbell_path(frame)
                website_video_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                packet = stream.encode(website_video_frame) # type: ignore
                output.mux(packet) # type: ignore
                progress_bar.progress(
                    self.n_of_frames / total_frames,
                    text=progress_text,
                )
            else:
                main_squat_data = self.get_main_squat_data()
                self.video.release()
                if self.video_writer is not None:
                    self.video_writer.release()
                return main_squat_data
