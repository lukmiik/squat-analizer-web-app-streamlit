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
    progress_bar = col_main.progress(0, text=progress_text)
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
    analized_squat_data = squat_analizer.analize_squat_streamlit(stream, progress_bar, progress_text, output)
    progress_bar.empty()
    packet = stream.encode(None)
    output.mux(packet)  # type: ignore
    output.close()  # type: ignore
    output_memory_file.seek(0)
    _, col1, col2, _ = st.columns([1, 0.8, 1.2, 1])
    col1.video(output_memory_file)
    if analized_squat_data:
        df = pd.DataFrame(analized_squat_data)
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
