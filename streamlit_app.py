import io
import os
import random
import string
from typing import TYPE_CHECKING

import av
import cv2
import pandas as pd
import streamlit as st

from squat_analizer.squat_analizer import SquatAnalizer

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

st.title('Squat analizer')

video_data = st.file_uploader("Upload file", ['mp4', 'mov', 'avi'])


def write_bytesio_to_file(filename: str, bytesio: 'UploadedFile') -> None:
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


if video_data:
    progress_text = "Analyzing video..."
    my_bar = st.progress(0, text=progress_text)
    while os.path.exists(
        file_to_analize := (
            ''.join(
                random.choice(string.ascii_letters + string.digits) for _ in range(20)
            )
            + ".mp4"
        )
    ):
        pass
    write_bytesio_to_file(file_to_analize, video_data)
    squat_analizer = SquatAnalizer(file_to_analize)
    squat_analizer.stop_printing_yolo_logs()
    output_memory_file = io.BytesIO()
    output = av.open(output_memory_file, 'w', format="mp4")
    stream = output.add_stream('h264', squat_analizer.fps)  # type: ignore
    stream.width = squat_analizer.video.get(cv2.CAP_PROP_FRAME_WIDTH)
    stream.height = squat_analizer.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17'}
    total_frames = squat_analizer.video.get(cv2.CAP_PROP_FRAME_COUNT)
    squat_analizer.init_squat_stats()
    squat_analizer.init_tracker_variables()
    while True:
        ret, frame = squat_analizer.video.read()
        if ret:
            frame = squat_analizer.downscale_video(frame)
            squat_analizer.n_of_frames += 1
            humans = squat_analizer.detect_humans(frame)
            if humans:
                barbells = squat_analizer.detect_barbells(frame)
                if barbells:
                    squat_analizer.init_humans_barbells_variables()
                    squat_analizer.set_humans_data(humans)
                    squat_analizer.set_barbells_data(barbells)
                    humans_barbells_pairs = squat_analizer.assign_humans_to_barbells(
                        squat_analizer.humans_points,
                        squat_analizer.barbells_points,
                        squat_analizer.humans_bboxes,
                    )
                    if humans_barbells_pairs:
                        humans_with_barbells_bboxes = (
                            squat_analizer.get_bboxes_humans_with_barbells(
                                squat_analizer.humans_bboxes, humans_barbells_pairs
                            )
                        )
                        barbells_of_humans_bboxes = (
                            squat_analizer.get_bboxes_barbells_of_humans(
                                squat_analizer.barbell_bboxes, humans_barbells_pairs
                            )
                        )
                        humans_with_barbells_conf_scores = (
                            squat_analizer.get_conf_scores_humans_with_barbells(
                                squat_analizer.humans_conf_scores, humans_barbells_pairs
                            )
                        )
                        barbells_of_humans_conf_scores = (
                            squat_analizer.get_conf_scores_barbells_of_humans(
                                squat_analizer.barbells_conf_scores,
                                humans_barbells_pairs,
                            )
                        )
                        humans_detections_converted_to_byte_tracker_format = (
                            squat_analizer.human_byte_tracker.convert_detections_data(
                                humans_with_barbells_bboxes,
                                humans_with_barbells_conf_scores,
                            )
                        )
                        human_byte_tracker_tracks = (
                            squat_analizer.human_byte_tracker.update(
                                humans_detections_converted_to_byte_tracker_format,
                                (frame.shape[0], frame.shape[1]),
                                (frame.shape[0], frame.shape[1]),
                            )
                        )
                        (
                            human_tracker_ids,
                            human_bboxes_idxs,
                        ) = squat_analizer.human_byte_tracker.assign_ids_to_detections(
                            humans_with_barbells_bboxes,
                            human_byte_tracker_tracks,
                        )
                        if human_tracker_ids is not None:
                            human_barbell_highest_conf_human_tracker_id = squat_analizer.get_human_barbell_highest_conf_tracker_id(
                                human_bboxes_idxs,  # type: ignore
                                humans_with_barbells_conf_scores,
                                barbells_of_humans_conf_scores,
                                human_tracker_ids,
                            )
                            squat_analizer.check_main_human_tracker_id(
                                human_tracker_ids
                            )
                            squat_analizer.check_main_human_tracker_id_based_on_highest_conf(
                                human_barbell_highest_conf_human_tracker_id
                            )
                            if (
                                squat_analizer.main_human_tracker_id is not None
                                and squat_analizer.main_human_tracker_id
                                in human_tracker_ids
                            ):
                                squat_analizer.barbell_assigned_to_main_human_bbox = (
                                    squat_analizer.get_main_bbox(
                                        squat_analizer.main_human_tracker_id,
                                        human_tracker_ids,
                                        human_bboxes_idxs,  # type: ignore
                                        barbells_of_humans_bboxes,
                                    )
                                )
                                squat_analizer.main_barbell_center_y = squat_analizer.get_coord_center_bbox(
                                    squat_analizer.barbell_assigned_to_main_human_bbox
                                )[
                                    1
                                ]
                                squat_analizer.main_human_bbox = (
                                    squat_analizer.get_main_bbox(
                                        squat_analizer.main_human_tracker_id,
                                        human_tracker_ids,
                                        human_bboxes_idxs,  # type: ignore
                                        humans_with_barbells_bboxes,
                                    )
                                )
                                squat_analizer.main_human_roi = (
                                    squat_analizer.get_human_roi(
                                        frame, squat_analizer.main_human_bbox
                                    )
                                )
                                pose_landmarks = (
                                    squat_analizer.get_pose_estimation_landmarks(
                                        squat_analizer.main_human_roi
                                    )
                                )
                                if pose_landmarks:
                                    squat_analizer.set_side_based_on_legs(
                                        pose_landmarks
                                    )
                                    if squat_analizer.have_feet_moved(
                                        squat_analizer.main_human_roi,
                                        squat_analizer.main_human_bbox[0][0],
                                        pose_landmarks,
                                    ):
                                        squat_analizer.lowest_main_barbell_center_y = (
                                            None
                                        )
                                        squat_analizer.current_state = (
                                            squat_analizer.PAUSE_STATE
                                        )
                                squat_analizer.detect_and_update_squat_data(
                                    pose_landmarks
                                )
            if (
                squat_analizer.current_state == squat_analizer.ECCENTRIC_PHASE_STATE
                or squat_analizer.current_state == squat_analizer.CONCENTRIC_PHASE_STATE
            ) and pose_landmarks:  # type: ignore
                squat_analizer.draw_squat_depth_info(
                    squat_analizer.main_human_roi,
                    pose_landmarks,  # type: ignore
                )
            squat_analizer.draw_barbell_path(frame)
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packet = stream.encode(frame)
            output.mux(packet)  # type: ignore
            my_bar.progress(
                squat_analizer.n_of_frames / total_frames,
                text=progress_text,
            )
        else:
            main_squat_data = squat_analizer.get_main_squat_data()
            squat_analizer.video.release()
            if squat_analizer.video_writer is not None:
                squat_analizer.video_writer.release()
            break
    my_bar.empty()
    packet = stream.encode(None)
    output.mux(packet)  # type: ignore
    output.close()  # type: ignore
    output_memory_file.seek(0)
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.video(output_memory_file)
    df = pd.DataFrame(main_squat_data)
    st.table(df.style.format({"eccentric_time": "{:.2f}", "concentric_time": "{:.2f}"}))
    os.remove(file_to_analize)
