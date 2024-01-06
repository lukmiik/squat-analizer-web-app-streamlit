import logging

import cv2
import numpy as np
from ultralytics import YOLO


class BaseVideoObjectDetection:
    CLOSE_WINDOW_KEY: int = 27  # 27 is the ESC key
    CONFIDENCE_THRESHOLD: float = 0.6
    models: list[YOLO] = []

    def __init__(self, video_path: str, models: list[str]) -> None:
        self.load_video(video_path)
        self.load_models(models)

    def load_video(self, path: str) -> None:
        self.video = cv2.VideoCapture(path)

    def load_models(self, models: list[str]) -> None:
        for model_path in models:
            model = YOLO(model_path)  # type: ignore
            self.models.append(model)

    def stop_printing_yolo_logs(self) -> None:
        ultralytics_logger = logging.getLogger('ultralytics')
        ultralytics_logger.setLevel(logging.CRITICAL)

    def downscale_frame_by_height(
        self, frame: np.ndarray, desired_height: int
    ) -> np.ndarray:
        height = frame.shape[0]
        if height > desired_height:
            width = frame.shape[1]
            aspect_ratio = width / height
            desired_width = int((desired_height * aspect_ratio))
            frame = cv2.resize(frame, (desired_width, desired_height))
        return frame

    def get_fps(self) -> float:
        return self.video.get(cv2.CAP_PROP_FPS)

    def get_results(
        self,
        model: YOLO,
        frame: np.ndarray,
        classes: int | list[int] | None = None,
        conf: float = CONFIDENCE_THRESHOLD,
    ) -> list[np.ndarray]:
        return model(frame, classes=classes, conf=conf)[0]

    def get_detection_data(self, detection: np.ndarray) -> np.ndarray:
        # x1, y1, x2, y2, score, class_id
        return detection.boxes.data.cpu().numpy()[0]  # type: ignore

    def get_bbox(self, detection: np.ndarray) -> np.ndarray:
        return detection.boxes.xyxy.cpu().numpy()  # type: ignore

    def get_class_id(self, detection: np.ndarray) -> np.float32:
        return detection.boxes.cls.cpu().numpy()[0].astype(int)  # type: ignore

    def get_conf_score(self, detection: np.ndarray) -> np.float32:
        return detection.boxes.conf.cpu().numpy()  # type: ignore

    def get_coord_center_bbox(self, bbox: np.ndarray) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox[0]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def draw_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        bbox = [int(x) for x in bbox[0]]  # type:  ignore
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def draw_bbox_label(
        self,
        frame: np.ndarray,
        label: str,
        bbox: np.ndarray,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.7,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        line_type: int = cv2.LINE_AA,
    ) -> None:
        bbox = [int(x) for x in bbox[0]]  # type:  ignore
        x1, y1, x2, y2 = bbox
        cv2.putText(
            frame, label, (x1, y1 - 5), font, font_scale, color, thickness, line_type
        )

    def draw_top_left_label(
        self,
        frame: np.ndarray,
        label: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        line_type: int = cv2.LINE_AA,
    ) -> None:
        # DEBUG
        cv2.putText(
            frame, label, (0, 30), font, font_scale, color, thickness, line_type
        )

    def draw_bottom_left_circle(
        self,
        frame: np.ndarray,
        radius: int = 30,
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = -1,
    ) -> None:
        # DEBUG
        cv2.circle(
            frame, (radius + 5, frame.shape[0] - radius - 5), radius, color, thickness
        )

    def draw_circle(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        radius: int = 2,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        cv2.circle(frame, (x, y), radius, color, thickness)

    def draw_line(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    def display_frame(
        self,
        frame: np.ndarray,
        title: str = "object detection",
    ) -> None:
        cv2.imshow(title, frame)

    def close_window(self) -> bool:
        if cv2.waitKey(1) == self.CLOSE_WINDOW_KEY:
            return True
        return False
