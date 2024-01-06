import io

import av
import cv2
import streamlit as st

video_data = st.file_uploader("Upload file", ['mp4', 'mov', 'avi'])

file_to_save = './file_to_save.mp4'


def write_bytesio_to_file(filename, bytesio) -> None:
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


if video_data:
    write_bytesio_to_file(file_to_save, video_data)
    width, height = 368, 480

    output_memory_file = io.BytesIO()

    output = av.open(output_memory_file, 'w', format="mp4")
    stream = output.add_stream('h264', str(30))  # type: ignore
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17'}

    video = cv2.VideoCapture(file_to_save)

    while True:
        ret, frame = video.read()
        if ret:
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packet = stream.encode(frame)
            output.mux(packet)  # type: ignore
        else:
            break
    packet = stream.encode(None)
    output.mux(packet)  # type: ignore
    output.close()  # type: ignore

    output_memory_file.seek(0)
    st.video(output_memory_file)
