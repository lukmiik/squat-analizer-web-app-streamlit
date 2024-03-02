import io
import logging
import os
import random
import string
import threading
import time
from time import perf_counter
from typing import TYPE_CHECKING

import av
import cv2
import pandas as pd
import streamlit as st

from squat_analizer.squat_analizer import SquatAnalizer

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

CURRENTLY_ANALIZED_VIDEOS = 'currently_analized_videos'
FILE_DELETE_DELAY = 3600
LOG_DIRECTORY = 'logs'

os.makedirs(CURRENTLY_ANALIZED_VIDEOS, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIRECTORY, 'info.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

st.set_page_config(
    layout="wide", page_icon=":weight_lifter:", page_title="Squat analizer"
)
_, col_main, _ = st.columns([1, 2, 1])
col_main.title('Squat analizer :weight_lifter:')

video_data = col_main.file_uploader("Upload file", ['mp4', 'mov', 'avi'])


def delete_file_after_delay(filename: str, delay: int = FILE_DELETE_DELAY) -> None:
    def delete_file() -> None:
        time.sleep(delay)
        if os.path.exists(filename):
            os.remove(filename)

    threading.Thread(target=delete_file).start()


def write_bytesio_to_file(filename: str, bytesio: 'UploadedFile') -> None:
    delete_file_after_delay(filename)
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


if video_data:
    start_time = perf_counter()
    progress_text = "Analyzing video..."
    my_bar = col_main.progress(0, text=progress_text)
    while os.path.exists(
        file_to_analize := (
            ''.join(
                random.choice(string.ascii_letters + string.digits) for _ in range(20)
            )
            + ".mp4"
        )
    ):
        pass
    file_to_analize_path = os.path.join(CURRENTLY_ANALIZED_VIDEOS, file_to_analize)
    write_bytesio_to_file(file_to_analize_path, video_data)
    size_in_bytes = video_data.getbuffer().nbytes
    size_in_mb = size_in_bytes / (1024 * 1024)
    squat_analizer = SquatAnalizer(file_to_analize_path)
    squat_analizer.stop_printing_yolo_logs()
    output_memory_file = io.BytesIO()
    output = av.open(output_memory_file, 'w', format="mp4")
    stream = output.add_stream('h264', squat_analizer.fps)  # type: ignore
    stream.width = squat_analizer.video.get(cv2.CAP_PROP_FRAME_WIDTH)
    stream.height = squat_analizer.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17'}
    total_frames = squat_analizer.video.get(cv2.CAP_PROP_FRAME_COUNT)
    length_in_seconds = total_frames / squat_analizer.fps
    uploaded_file_name = video_data.name
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
                                humans_with_barbells_conf_scores,  # type: ignore
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
                    squat_analizer.main_human_roi, # type: ignore
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
    _, col1, col2, _ = st.columns([1, 0.8, 1.2, 1])
    col1.video(output_memory_file)
    if main_squat_data:
        df = pd.DataFrame(main_squat_data)
        df['depth'] = df['depth'].apply(
            lambda x: '✅' if x is True else ('❌' if x is False else '')
        )
        col2.dataframe(
            df.style.format({"eccentric_time": "{:.2f}", "concentric_time": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        col2.info("No squats detected")
    end_time = perf_counter()
    time_elapsed = end_time - start_time
    logging.info(
        f"Video name: '{uploaded_file_name}', Video length: {length_in_seconds:.2f} seconds, File size: {size_in_mb:.2f} mb, Time elapsed main function: {time_elapsed:.2f} seconds, Hardware {squat_analizer.models[0].device}"
    )
