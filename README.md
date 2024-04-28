# Squat analizer web app
Web app analizes barbell squat based on the video uploaded by a user. It shows the amount of reps, depth, eccentric phase time, concentric phase time for each rep and average eccentric and concentric time. Moreover it displays video uploaded by a user with depth correctness information drawn for each rep and a bar path with different colors for eccentric and concentric phase. App uses OpenCV for video processing, YOLOv8 for object detection, ByteTrack for object tracking and MediaPipe for pose estimation.

https://github.com/lukmiik/squat-analizer-web-app-streamlit/assets/72356234/d2ea8d21-7973-4de4-9bb5-68b68162b022

***Install requirements***
```bash
pip install -r requirements.txt
```

***Run application***
```bash
streamlit run streamlit_app.py
```
