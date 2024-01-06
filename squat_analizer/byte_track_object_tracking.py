import numpy as np
from onemetric.cv.utils.iou import box_iou_batch
from yolox.tracker.byte_tracker import BYTETracker, STrack


class ByteTrackArgs:
    track_thresh = 0.6
    track_buffer = 100
    match_thresh = 0.8
    aspect_ratio_thresh = 10.0
    min_box_area = 1.0
    mot20 = False


class BaseByteTrackObjectTracking(BYTETracker):
    def __init__(self, frame_rate: int = 30) -> None:
        super().__init__(ByteTrackArgs, frame_rate)

    def convert_detections_data(
        self, bboxes: list[np.ndarray], conf_scores: list[np.float32]
    ) -> np.ndarray:
        flat_conf_scores = np.hstack(conf_scores).flatten()
        reshaped_bbox = self._convert_bboxes_list_data(bboxes)
        return np.hstack((reshaped_bbox, flat_conf_scores[:, np.newaxis]))

    def _convert_tracking_data(self, tracks: list[STrack]) -> np.ndarray:
        return np.array([track.tlbr for track in tracks], dtype=float)

    def assign_ids_to_detections(
        self, bboxes: list[np.ndarray], tracks: list[STrack]
    ) -> tuple[list[int], list[int]] | tuple[None, None]:
        reshaped_bbox = self._convert_bboxes_list_data(bboxes)
        if not np.any(reshaped_bbox) or len(tracks) == 0:
            return None, None
        track_np = self._convert_tracking_data(tracks)
        iou = box_iou_batch(track_np, reshaped_bbox)
        tracks_detections_idxs = np.argmax(iou, axis=1)
        tracker_ids = []
        detections_idxs = []
        for tracker_idx, detection_idx in enumerate(tracks_detections_idxs):
            if iou[tracker_idx, detection_idx] > 0.5:
                tracker_ids.append(tracks[tracker_idx].track_id)
                detections_idxs.append(detection_idx)
        if tracker_ids:
            return tracker_ids, detections_idxs
        return None, None

    def _convert_bboxes_list_data(self, bboxes: list[np.ndarray]) -> np.ndarray:
        flat_bbox = np.hstack(bboxes).flatten()
        reshaped_bbox = flat_bbox.reshape(-1, 4)
        return reshaped_bbox
