import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import random
import time
import base64
from gtts import gTTS

# ================== PEMICU IMPOR AMAN UNTUK LINUX SERVER ==================
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# ================== CONFIG UI STREAMLIT ==================
st.set_page_config(page_title="SignAI MathVision", layout="wide")
st.title("🧮 SignAI MathVision - SLB Madiun")
st.write("Media Pembelajaran Matematika Isyarat Berbasis AI")

# ================== STATE MANAGEMENT (ANTI-RESET) ==================
if 'skor' not in st.session_state:
    st.session_state.skor = 0
if 'current_soal' not in st.session_state:
    st.session_state.current_soal = 0
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""
if 'last_spoken_soal' not in st.session_state:
    st.session_state.last_spoken_soal = None

# Bank Soal PKM-PM
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

soal_sekarang = st.session_state.soal_list[st.session_state.current_soal]

# ================== LOGIKA AUDIO (gTTS AUTO-PLAY) ==================
def putar_suara(text):
    try:
        tts = gTTS(text=text, lang='id', slow=False)
        filename = "temp_voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            audio_html = f'<audio autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(audio_html, unsafe_url_allowed=True)
    except Exception as e:
        pass

# Trigger suara soal baru jika belum pernah diucapkan
if st.session_state.last_spoken_soal != soal_sekarang:
    soal_text = f"Berapakah {soal_sekarang.replace('+', ' ditambah ').replace('-', ' dikurangi ')}"
    putar_suara(soal_text)
    st.session_state.last_spoken_soal = soal_sekarang

# ================== LAYOUT DASHBOARD ==================
col_info, col_cam = st.columns([1, 2])

with col_info:
    st.subheader("📋 Informasi Soal")
    st.metric(label="SOAL MATEMATIKA", value=f"{soal_sekarang} = ?")
    st.metric(label="TOTAL SKOR", value=st.session_state.skor)
    
    if st.session_state.feedback:
        if "BENAR" in st.session_state.feedback:
            st.success(st.session_state.feedback)
        else:
            st.error(st.session_state.feedback)
            
    if st.button("Skip Soal Ini ➡️"):
        st.session_state.current_soal = (st.session_state.current_soal + 1) % len(st.session_state.soal_list)
        st.session_state.feedback = ""
        st.rerun()

# ================== MEDIA PIPE DETECTOR LOGIC ==================
hands = mp_hands.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    max_num_hands=2
)

def hitung_jari_asl(hand, handedness):
    label = handedness.classification[0].label
    thumb_up = 1 if ((label == "Right" and hand.landmark[4].x < hand.landmark[2].x) or (label == "Left" and hand.landmark[4].x > hand.landmark[2].x)) else 0
    index_up = 1 if hand.landmark[8].y < hand.landmark[5].y else 0
    middle_up = 1 if hand.landmark[12].y < hand.landmark[9].y else 0
    ring_up = 1 if hand.landmark[16].y < hand.landmark[13].y else 0
    pinky_up = 1 if hand.landmark[20].y < hand.landmark[17].y else 0
    
    if thumb_up == 0 and index_up == 0 and middle_up == 0 and ring_up == 0 and pinky_up == 0:
        return 0
    return thumb_up + index_up + middle_up + ring_up + pinky_up

class GameVisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_val = -1
        self.stable_time = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        total_nilai_isyarat = 0
        tangan_muncul = False
        
        if result.multi_hand_landmarks and result.multi_handedness:
            tangan_muncul = True
            daftar_tangan = []
            
            for i in range(len(result.multi_hand_landmarks)):
                hand_landmarks = result.multi_hand_landmarks[i]
                handedness = result.multi_handedness[i]
                
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                nilai_angka = hitung_jari_asl(hand_landmarks, handedness)
                wrist_x = hand_landmarks.landmark[0].x
                daftar_tangan.append((nilai_angka, wrist_x))
            
            if len(daftar_tangan) == 1:
                single_nilai = daftar_tangan[0][0]
                single_x = daftar_tangan[0][1]
                total_nilai_isyarat = single_nilai * 10 if single_x < 0.5 else single_nilai
            elif len(daftar_tangan) == 2:
                daftar_tangan.sort(key=lambda x: x[1])
                total_nilai_isyarat = (daftar_tangan[0][0] * 10) + daftar_tangan[1][0]

        if tangan_muncul:
            cv2.putText(img, f"Isyarat: {total_nilai_isyarat}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            if total_nilai_isyarat != self.last_val:
                self.last_val = total_nilai_isyarat
                self.stable_time = time.time()
            elif self.stable_time and (time.time() - self.stable_time > 1.5):
                soal = st.session_state.soal_list[st.session_state.current_soal]
                a, b = map(int, soal.split("+")) if "+" in soal else map(int, soal.split("-"))
                jawaban_benar = a + b if "+" in soal else a - b
                
                if total_nilai_isyarat == jawaban_benar:
                    st.session_state.skor += 1
                    st.session_state.feedback = f"BENAR! 🎉 Jawabannya adalah {jawaban_benar}."
                    st.session_state.current_soal = (st.session_state.current_soal + 1) % len(st.session_state.soal_list)
                else:
                    st.session_state.feedback = f"SALAH ❌ Terbaca {total_nilai_isyarat}, coba lagi!"
                
                self.stable_time = None
        else:
            self.stable_time = None

        return frame.from_ndarray(img, format="bgr24")

# ================== RENDER KAMERA WEB (MULTI-STUN SERVERS) ==================
with col_cam:
    st.subheader("🎥 Deteksi Kamera Web")
    webrtc_streamer(
        key="SignAI-MathVision-PIMNAS-FINAL-PRO",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=GameVisionProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun.services.mozilla.com"]}
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
