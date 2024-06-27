import sys
import os
import cv2                     #컴퓨터 비전 작업을 위한 OpenCV 라이브러리 -> 비디오 프레임을 읽고 처리하는 기능이 포함
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QPushButton, QSlider, QVBoxLayout, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal #GUI(그래픽 사용자 인터페이스)와 그 구성 요소를 생성하기 위해 사용
from pytube import YouTube                   # YouTube 비디오를 다운로드하기 위해 사용
from pytube.exceptions import VideoUnavailable
from google.protobuf.message import DecodeError
import mediapipe as mp       #카메라나 비디오 프레임에서 포즈 추정을 위해 사용
import time
import pyttsx3               #텍스트를 음성으로 변환하는 기능을 제공

class VoiceFeedbackThread(QThread):
    def __init__(self):
        super().__init__()
        self.feedback = ""
        self.engine = pyttsx3.init()

    def set_feedback(self, feedback):
        self.feedback = feedback

    def run(self):
        while True:
            if self.feedback:
                self.engine.say(self.feedback)
                self.engine.runAndWait()
                self.feedback = "" 
            time.sleep(1)

#이 클래스는 웹캠 및 YouTube 비디오를 표시하고 다양한 컨트롤 및 기능을 제공하기 위한 GUI 위젯을 설정
class VideoDisplayWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.video_links = {
            "운동 비디오1 (상체, 팔 운동)": "https://youtu.be/I6N1v1Jqazo?si=U-msF87sfLrw5e9k",
            "운동 비디오2 (복근) ": "https://youtu.be/QSZ0mUGO_CA?si=JMazUBoXYFNj61FX",
            "운동 비디오3 (기초 체력 중심)": "https://youtu.be/8zTsNYJkoXQ?si=NcKN738GvNYhpnrv",
            "운동 비디오4 (스트레칭)": "https://youtu.be/pc_hXPTLirA?si=C94aIDJl3f-mkGAs",
        }
        self.youtube_video_path = None
        self.target_size = (640, 480)
        self.initUI()

    #UI를 초기화하며, 창 제목과 크기를 설정
    def initUI(self):
        self.setWindowTitle("체력 향상 프로그램")
        self.setGeometry(100, 100, 1300, 600)

        self.layout = QGridLayout()    #위젯을 행과 열의 그리드로 구성, 배치

        self.webcam_label = QLabel(self)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.webcam_label, 0, 0)

        self.youtube_label = QLabel(self)
        self.youtube_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.youtube_label, 0, 1)

        self.result_label = QLabel("운동 자세를 평가 중입니다...", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label, 1, 0, 1, 2)

        self.start_webcam_button = QPushButton("웹캠 시작", self)
        self.start_webcam_button.clicked.connect(self.start_webcam)
        self.layout.addWidget(self.start_webcam_button, 2, 0)

        self.start_youtube_button = QPushButton("유튜브 비디오 시작", self)
        self.start_youtube_button.clicked.connect(self.start_youtube)
        self.layout.addWidget(self.start_youtube_button, 2, 1)

        self.video_combo_box = QComboBox(self)
        for title in self.video_links.keys():
            self.video_combo_box.addItem(title)
        self.video_combo_box.currentIndexChanged.connect(self.on_video_selection_changed)
        self.layout.addWidget(self.video_combo_box, 3, 0, 1, 2)

        self.youtube_slider = QSlider(Qt.Horizontal, self)
        self.youtube_slider.setRange(0, 100)
        self.youtube_slider.sliderMoved.connect(self.set_youtube_position)
        self.layout.addWidget(self.youtube_slider, 4, 0, 1, 2)

        self.setLayout(self.layout)

        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self.update_webcam_frame)

        self.youtube_timer = QTimer(self)
        self.youtube_timer.timeout.connect(self.update_youtube_frame)

        self.cap = None
        self.youtube_cap = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        self.reference_landmarks = None
        self.previous_landmarks = []

        self.voice_thread = VoiceFeedbackThread()
        self.voice_thread.start()

    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다.")
                return
            self.webcam_timer.start(30)
            self.start_webcam_button.setText("웹캠 중지")
        else:
            self.cap.release()
            self.cap = None
            self.webcam_timer.stop()
            self.webcam_label.clear()
            self.start_webcam_button.setText("웹캠 시작")

    def start_youtube(self):
        if self.youtube_cap is None and self.youtube_video_path:
            self.youtube_cap = cv2.VideoCapture(self.youtube_video_path)
            if not self.youtube_cap.isOpened():
                print("유튜브 비디오를 열 수 없습니다.")
                return
            self.youtube_timer.start(30)
            self.start_youtube_button.setText("유튜브 비디오 중지")
        elif self.youtube_cap is None and not self.youtube_video_path:
            print("먼저 유튜브 비디오를 다운로드하세요.")
        else:
            self.youtube_cap.release()
            self.youtube_cap = None
            self.youtube_timer.stop()
            self.youtube_label.clear()
            self.start_youtube_button.setText("유튜브 비디오 시작")

    def on_video_selection_changed(self, index):
        video_title = self.video_combo_box.currentText()
        if video_title in self.video_links:
            youtube_url = self.video_links[video_title]
            self.download_youtube_video(youtube_url)

    def download_youtube_video(self, youtube_url):
        retry_count = 3
        for attempt in range(retry_count):
            try:
                yt = YouTube(youtube_url)
                video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                if video:
                    self.youtube_video_path = os.path.abspath(video.download(filename='downloaded_video.mp4'))
                    print(f"다운로드된 비디오: {self.youtube_video_path}")
                    break
                else:
                    print("프로그레시브 MP4 스트림을 사용할 수 없습니다.")
            except (VideoUnavailable, DecodeError, Exception) as e:
                print(f"유튜브 비디오 다운로드 오류: {e}")
                if attempt < retry_count - 1:
                    print("재시도 중...")
                    time.sleep(2)
                else:
                    print("유튜브 비디오 다운로드를 포기합니다.")

    def update_webcam_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    self.previous_landmarks.append(results.pose_landmarks)
                    if len(self.previous_landmarks) > 5:
                        self.previous_landmarks.pop(0)
                    if self.reference_landmarks:
                        is_correct = self.is_correct_pose(results.pose_landmarks, self.reference_landmarks)
                        self.update_result_label(is_correct)
                self.display_frame(frame, self.webcam_label)

    def update_youtube_frame(self):
        if self.youtube_cap is not None:
            ret, frame = self.youtube_cap.read()
            if ret:
                frame = cv2.resize(frame, self.target_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    self.reference_landmarks = results.pose_landmarks
                    self.update_result_label(True)
                self.display_frame(frame, self.youtube_label)
                self.update_slider_position()

    def display_frame(self, frame, label):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qImg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)

    #사용자의 포즈와 기준 포즈를 비교하여 올바른 자세인지 여부를 판단
    def is_correct_pose(self, user_landmarks, reference_landmarks, threshold_distance=0.05, threshold_angle_deg=5):
        if user_landmarks and reference_landmarks:
            user_coords = [(lm.x, lm.y) for lm in user_landmarks.landmark]
            ref_coords = [(lm.x, lm.y) for lm in reference_landmarks.landmark]

            for (ux, uy), (rx, ry) in zip(user_coords, ref_coords):
                distance = np.sqrt((ux - rx) ** 2 + (uy - ry) ** 2)
                if distance > threshold_distance:
                    return False

            user_angles = self.calculate_joint_angles(user_coords)
            ref_angles = self.calculate_joint_angles(ref_coords)
            angle_diffs = np.abs(np.array(user_angles) - np.array(ref_angles))
            if np.any(angle_diffs > threshold_angle_deg):
                return False

            return True
        return False

    #관절 각도 계산    ** 부족한 부분
    def calculate_joint_angles(self, landmarks):
        angles = []
        joints = [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)]
        for a, b, c in joints:
            vec1 = np.array([landmarks[a][0] - landmarks[b][0], landmarks[a][1] - landmarks[b][1]])
            vec2 = np.array([landmarks[c][0] - landmarks[b][0], landmarks[c][1] - landmarks[b][1]])
            angle_rad = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
        return angles

    #운동 자세 평가 결과에 따라 GUI 상의 라벨(result_label)을 업데이트하고, 
    #음성 피드백 스레드(voice_thread)에도 결과를 전달
    def update_result_label(self, is_correct):
        if is_correct:
            self.result_label.setText("운동 자세가 올바릅니다.")
            self.result_label.setStyleSheet("color: green; font-weight: bold;")
            self.voice_thread.set_feedback("운동 자세가 올바릅니다.계속 열심히 해주세요!")
        else:
            self.result_label.setText("운동 자세가 올바르지 않습니다.")
            self.result_label.setStyleSheet("color: red; font-weight: bold;")
            self.voice_thread.set_feedback("운동 자세가 올바르지 않습니다. 자세를 고쳐주세요!")

    def set_youtube_position(self, position):
        if self.youtube_cap is not None:
            total_frames = int(self.youtube_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            new_frame_number = int((position / 100) * total_frames)
            self.youtube_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_number)

    def update_slider_position(self):
        if self.youtube_cap is not None:
            total_frames = int(self.youtube_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = int(self.youtube_cap.get(cv2.CAP_PROP_POS_FRAMES))
            position = int((current_frame / total_frames) * 100)
            self.youtube_slider.setValue(position)

    def closeEvent(self, event):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.youtube_cap is not None and self.youtube_cap.isOpened():
            self.youtube_cap.release()
        self.webcam_timer.stop()
        self.youtube_timer.stop()
        self.voice_thread.terminate()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoDisplayWidget()
    ex.show()
    sys.exit(app.exec_())
