import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import mediapipe as mp
import time
import random
import string
from gtts import gTTS
import base64
import os

# ================== CONFIG UI STREAMLIT ==================
st.set_page_config(page_title="SignAI MathVision", layout="wide")
st.title("SignAI MathVision - SLB Madiun")
st.write("Media Pembelajaran Matematika Isyarat Berbasis AI")

# ================== STATE MANAGEMENT (ANTI-RESET) ==================
# Streamlit selalu me-rerun skrip dari atas. Kita pakai session_state agar skor & soal terkunci.
if 'skor' not in st.session_state:
    st.session_state.skor = 0
if 'current_soal' not in st.session_state:
    st.session_state.current_soal = 0
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""
if 'feedback_time' not in st.session_state:
    st.session_state.feedback_time = 0
if 'has_spoken_soal' not in st.session_state:
    st.session_state.has_spoken_soal = False

# Bank soal asli kamu
if 'soal_list' not in st.session_state:
    soals = [
        "0+0", "5-5", "10-10", "20-20", "30-30", "40-40", "50-50",
        "5+0", "1+1", "2+2", "3+3", "4+4", "5+2", "5+4", "9-5", "7-3", "10-4",
        "10+0", "5+5", "15-5", "20-10", "10+4", "10+7", "18-4", "19-2",
        "20+0", "10+10", "15+5", "30-10", "25+2", "29-4", "24+4", "38-12",
        "30+0", "20+10", "25+5", "40-10", "15+15", "50-20", "35+2", "45-12",
        "40+0", "30+10", "20+20", "50-10", "35+5", "45+5", "22+22", "55-15"
    ]
    random.shuffle(soals)
    st.session_state.soal_list = soals

soal_list = st.session_state.soal_list
current_soal = st.session_state.current_soal

# ================== OUTPUT SUARA VERSI WEB (ANTI-LAG) ==================
def speak_web(text):
    """Menggunakan HTML5 Audio tag agar suara berbunyi di browser Guru (bukan di server)"""
    try:
        tts = gTTS(text=text, lang='id', slow=False)
        filename = f"temp_voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            audio_html = f"""
                <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        os.remove(filename)
    except:
        pass

# Bersuara untuk soal pertama/baru
if not st.session_state.has_spoken_soal:
    soal_teks = soal_list[current_soal].replace("+", " tambah ").replace("-", " kurang ")
    if "0" in soal_teks:
        soal_teks = soal_teks.replace("0", "nol")
    speak_web(f"Berapakah {soal_teks}?")
    st.session_state.has_spoken_soal = True

# ================== UI DASHBOARD UTAMA ==================
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown(f"## 📝 Soal: <span style='color:#00C8FF'>{soal_list[current_soal]}</span>", unsafe_allow_html=True)
    st.metric(label="🏆 Skor Kamu", value=st.session_state.skor)
    
    if st.session_state.feedback != "":
        if "BENAR" in st.session_state.feedback:
            st.success(st.session_state.feedback)
        else:
            st.error(st.session_state.feedback)

# ================== LOGIKA ASL INTERPRETATION ==================
def konversi_ke_angka_asl(hand, handedness):
    label = handedness.classification[0].label
    thumb_up = 0
    if label == "Right":  
        if hand.landmark[4].x < hand.landmark[2].x: thumb_up = 1
    else:  
        if hand.landmark[4].x > hand.landmark[2].x: thumb_up = 1

    index_up = 1 if hand.landmark[8].y < hand.landmark[5].y else 0
    middle_up = 1 if hand.landmark[12].y < hand.landmark[9].y else 0
    ring_up = 1 if hand.landmark[16].y < hand.landmark[13].y else 0
    pinky_up = 1 if hand.landmark[20].y < hand.landmark[17].y else 0

    if thumb_up == 0 and index_up == 0 and middle_up == 0 and ring_up == 0 and pinky_up == 0:
        return 0
    if thumb_up == 0 and index_up == 1 and middle_up == 0 and ring_up == 0 and pinky_up == 0:
        return 1
    elif thumb_up == 0 and index_up == 1 and middle_up == 1 and ring_up == 0 and pinky_up == 0:
        return 2
    elif thumb_up == 0 and index_up == 1 and middle_up == 1 and ring_up == 1 and pinky_up == 0:
        return 3
    elif thumb_up == 0 and index_up == 1 and middle_up == 1 and ring_up == 1 and pinky_up == 1:
        return 4
    elif thumb_up == 1 and index_up == 1 and middle_up == 1 and ring_up == 1 and pinky_up == 1:
        return 5
    elif thumb_up == 1 and index_up == 0 and middle_up == 0 and ring_up == 0 and pinky_up == 0:
        return 6
    elif thumb_up == 1 and index_up == 1 and middle_up == 0 and ring_up == 0 and pinky_up == 0:
        return 7
    elif thumb_up == 1 and index_up == 1 and middle_up == 1 and ring_up == 0 and pinky_up == 0:
        return 8
    elif thumb_up == 1 and index_up == 1 and middle_up == 1 and ring_up == 1 and pinky_up == 0:
        return 9
    else:
        return thumb_up + index_up + middle_up + ring_up + pinky_up

# ================== CORE ENGINE (WEB-BASED) ==================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_detected_fingers = -1
        self.stable_start_time = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)

        total_nilai_isyarat = 0
        tangan_muncul = False

        if result.multi_hand_landmarks and result.multi_handedness:
            tangan_muncul = True
            daftar_tangan = []
            
            for i in range(len(result.multi_hand_landmarks)):
                hand_landmarks = result.multi_hand_landmarks[i]
                handedness = result.multi_handedness[i]
                
                self.mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                nilai_angka = konversi_ke_angka_asl(hand_landmarks, handedness)
                wrist_x = hand_landmarks.landmark[0].x
                daftar_tangan.append((nilai_angka, wrist_x))
            
            if len(daftar_tangan) == 1:
                single_nilai = daftar_tangan[0][0]
                single_x = daftar_tangan[0][1]
                if single_x < 0.5:
                    total_nilai_isyarat = single_nilai * 10
                else:
                    total_nilai_isyarat = single_nilai
            elif len(daftar_tangan) == 2:
                daftar_tangan.sort(key=lambda x: x[1])
                total_nilai_isyarat = (daftar_tangan[0][0] * 10) + daftar_tangan[1][0]

        # Logika Evaluasi Jawaban Terkunci
        current_time = time.time()
        if tangan_muncul:
            if total_nilai_isyarat != self.last_detected_fingers:
                self.last_detected_fingers = total_nilai_isyarat
                self.stable_start_time = current_time
            elif self.stable_start_time and (current_time - self.stable_start_time > 1.5):
                soal = soal_list[current_soal]
                if "+" in soal:
                    a, b = map(int, soal.split("+"))
                    jawaban_benar = a + b
                else:
                    a, b = map(int, soal.split("-"))
                    jawaban_benar = a - b

                if total_nilai_isyarat == jawaban_benar:
                    st.session_state.skor += 1
                    st.session_state.feedback = "BENAR 👍"
                    st.session_state.current_soal = (st.session_state.current_soal + 1) % len(soal_list)
                    st.session_state.has_spoken_soal = False
                else:
                    st.session_state.feedback = "SALAH ❌, Ayo coba lagi!"
                
                self.stable_start_time = None
        else:
            self.stable_start_time = None

        # Gambar UI overlay di video web
        cv2.putText(img, f"Isyarat Terbaca: {total_nilai_isyarat}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        return img

with col1:
    webrtc_streamer(
        key="SignAI-MathVision",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
