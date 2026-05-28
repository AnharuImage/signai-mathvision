import streamlit as st
import cv2
import mediapipe as mp
import time
import threading
import os
import random
import string
import numpy as np
from gtts import gTTS
import base64
import io

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="SignAI MathVision",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS STYLING ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #00c8ff, #7b2ff7, #00c8ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 0.2rem;
    }
    @keyframes shine {
        0% { background-position: 0%; }
        100% { background-position: 200%; }
    }
    .subtitle {
        text-align: center;
        color: #7a8fa6;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Soal Card */
    .soal-card {
        background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(123,47,247,0.08));
        border: 1px solid rgba(0,200,255,0.3);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(0,200,255,0.1);
    }
    .soal-label {
        font-size: 0.85rem;
        color: #7a8fa6;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .soal-text {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        color: #00c8ff;
        text-shadow: 0 0 15px rgba(0,200,255,0.5);
    }

    /* Stats row */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .stat-card {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #7a8fa6;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Feedback */
    .feedback-benar {
        background: linear-gradient(135deg, rgba(120,220,0,0.15), rgba(0,200,100,0.1));
        border: 2px solid #78dc00;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        color: #78dc00;
        text-shadow: 0 0 10px rgba(120,220,0,0.5);
        animation: pulse 0.5s ease-in-out;
    }
    .feedback-salah {
        background: linear-gradient(135deg, rgba(255,50,50,0.15), rgba(200,0,50,0.1));
        border: 2px solid #ff3232;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        color: #ff3232;
        text-shadow: 0 0 10px rgba(255,50,50,0.5);
        animation: shake 0.4s ease-in-out;
    }
    @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
    @keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-5px)} 75%{transform:translateX(5px)} }

    /* Scanning bar */
    .scanning-bar-wrap {
        background: rgba(255,255,255,0.05);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .scanning-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00c8ff, #7b2ff7);
        transition: width 0.1s linear;
    }

    /* Isyarat detected */
    .isyarat-badge {
        background: rgba(255,200,0,0.1);
        border: 1px solid rgba(255,200,0,0.4);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        color: #ffc800;
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }

    /* Panduan */
    .panduan-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }
    .panduan-title {
        font-size: 0.8rem;
        color: #7a8fa6;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .panduan-item {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #a0b0c0;
    }

    /* Camera frame */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid rgba(0,200,255,0.2);
        box-shadow: 0 0 25px rgba(0,200,255,0.08);
    }

    .stButton > button {
        background: linear-gradient(135deg, #00c8ff22, #7b2ff722);
        border: 1px solid rgba(0,200,255,0.4);
        color: #00c8ff;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00c8ff44, #7b2ff744);
        border-color: #00c8ff;
        box-shadow: 0 0 12px rgba(0,200,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ================== MEDIAPIPE SETUP ==================
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

# ================== SESSION STATE ==================
def init_state():
    soal_list = [
        "0+0","5-5","10-10","20-20","30-30","40-40","50-50",
        "5+0","1+1","2+2","3+3","4+4","5+2","5+4","9-5","7-3","10-4",
        "10+0","5+5","15-5","20-10","10+4","10+7","18-4","19-2",
        "20+0","10+10","15+5","30-10","25+2","29-4","24+4","38-12",
        "30+0","20+10","25+5","40-10","15+15","50-20","35+2","45-12",
        "40+0","30+10","20+20","50-10","35+5","45+5","22+22","55-15"
    ]
    random.shuffle(soal_list)
    return soal_list

if "soal_list"          not in st.session_state: st.session_state.soal_list          = init_state()
if "current_soal"       not in st.session_state: st.session_state.current_soal       = 0
if "skor"               not in st.session_state: st.session_state.skor               = 0
if "total_dijawab"      not in st.session_state: st.session_state.total_dijawab      = 0
if "feedback"           not in st.session_state: st.session_state.feedback           = ""
if "feedback_time"      not in st.session_state: st.session_state.feedback_time      = 0
if "stable_start"       not in st.session_state: st.session_state.stable_start       = None
if "last_fingers"       not in st.session_state: st.session_state.last_fingers       = -1
if "camera_on"          not in st.session_state: st.session_state.camera_on          = False
if "audio_b64"          not in st.session_state: st.session_state.audio_b64          = ""
if "just_answered"      not in st.session_state: st.session_state.just_answered      = False

# ================== FUNGSI SUARA (gTTS → base64 → autoplay) ==================
def get_audio_b64(text: str) -> str:
    try:
        buf = io.BytesIO()
        tts = gTTS(text=text, lang='id', slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except:
        return ""

def play_audio_html(b64: str):
    if b64:
        st.markdown(
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

# ================== KONVERSI ISYARAT ASL ==================
def konversi_ke_angka_asl(hand, handedness):
    label = handedness.classification[0].label
    thumb_up = 0
    if label == "Right":
        if hand.landmark[4].x < hand.landmark[2].x: thumb_up = 1
    else:
        if hand.landmark[4].x > hand.landmark[2].x: thumb_up = 1

    index_up  = 1 if hand.landmark[8].y  < hand.landmark[5].y  else 0
    middle_up = 1 if hand.landmark[12].y < hand.landmark[9].y  else 0
    ring_up   = 1 if hand.landmark[16].y < hand.landmark[13].y else 0
    pinky_up  = 1 if hand.landmark[20].y < hand.landmark[17].y else 0

    if thumb_up==0 and index_up==0 and middle_up==0 and ring_up==0 and pinky_up==0: return 0
    if thumb_up==0 and index_up==1 and middle_up==0 and ring_up==0 and pinky_up==0: return 1
    if thumb_up==0 and index_up==1 and middle_up==1 and ring_up==0 and pinky_up==0: return 2
    if thumb_up==0 and index_up==1 and middle_up==1 and ring_up==1 and pinky_up==0: return 3
    if thumb_up==0 and index_up==1 and middle_up==1 and ring_up==1 and pinky_up==1: return 4
    if thumb_up==1 and index_up==1 and middle_up==1 and ring_up==1 and pinky_up==1: return 5
    if thumb_up==1 and index_up==0 and middle_up==0 and ring_up==0 and pinky_up==0: return 6
    if thumb_up==1 and index_up==1 and middle_up==0 and ring_up==0 and pinky_up==0: return 7
    if thumb_up==1 and index_up==1 and middle_up==1 and ring_up==0 and pinky_up==0: return 8
    if thumb_up==1 and index_up==1 and middle_up==1 and ring_up==1 and pinky_up==0: return 9
    return thumb_up + index_up + middle_up + ring_up + pinky_up

# ================== PROSES FRAME ==================
def proses_frame(frame, hands_model):
    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = hands_model.process(rgb)

    total_isyarat = 0
    tangan_muncul = False

    if result.multi_hand_landmarks and result.multi_handedness:
        tangan_muncul = True
        daftar_tangan = []
        for i, hl in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,200,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(123,47,247), thickness=2)
            )
            nilai = konversi_ke_angka_asl(hl, result.multi_handedness[i])
            daftar_tangan.append((nilai, hl.landmark[0].x))

        if len(daftar_tangan) == 1:
            v, x = daftar_tangan[0]
            total_isyarat = v * 10 if x < 0.5 else v
        elif len(daftar_tangan) == 2:
            daftar_tangan.sort(key=lambda t: t[1])
            total_isyarat = daftar_tangan[0][0] * 10 + daftar_tangan[1][0]

    return frame, total_isyarat, tangan_muncul

# ================== LAYOUT UI ==================
st.markdown('<div class="main-title">🤟 SignAI MathVision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">SLB Madiun Mitra · Hand Sign Math Quiz</div>', unsafe_allow_html=True)

col_cam, col_panel = st.columns([3, 2], gap="large")

# -------- PANEL KANAN --------
with col_panel:
    soal_skr = st.session_state.soal_list[st.session_state.current_soal]

    # Kartu soal
    st.markdown(f"""
    <div class="soal-card">
        <div class="soal-label">✦ Soal Sekarang ✦</div>
        <div class="soal-text">{soal_skr} = ?</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    akurasi = int(st.session_state.skor / st.session_state.total_dijawab * 100) if st.session_state.total_dijawab else 0
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value" style="color:#78dc00">{st.session_state.skor}</div>
            <div class="stat-label">Benar</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#ff3232">{st.session_state.total_dijawab - st.session_state.skor}</div>
            <div class="stat-label">Salah</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#00c8ff">{akurasi}%</div>
            <div class="stat-label">Akurasi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feedback placeholder
    feedback_ph = st.empty()
    if st.session_state.feedback:
        if "BENAR" in st.session_state.feedback:
            feedback_ph.markdown(f'<div class="feedback-benar">✅ {st.session_state.feedback}</div>', unsafe_allow_html=True)
        else:
            feedback_ph.markdown(f'<div class="feedback-salah">❌ {st.session_state.feedback}</div>', unsafe_allow_html=True)

    # Isyarat badge placeholder
    isyarat_ph  = st.empty()
    progress_ph = st.empty()

    # Panduan
    st.markdown("""
    <div class="panduan-card">
        <div class="panduan-title">📖 Cara Bermain</div>
        <div class="panduan-item">🤚 <span>Satu tangan di <b>kiri layar</b> = puluhan (×10)</span></div>
        <div class="panduan-item">🤚 <span>Satu tangan di <b>kanan layar</b> = satuan</span></div>
        <div class="panduan-item">🙌 <span>Dua tangan = kiri(×10) + kanan</span></div>
        <div class="panduan-item">✊ <span>Kepalan = angka 0</span></div>
        <div class="panduan-item">⏱️ <span>Tahan 1.5 detik untuk mengunci jawaban</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Tombol
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Mulai Kamera" if not st.session_state.camera_on else "⏹ Stop Kamera"):
            st.session_state.camera_on = not st.session_state.camera_on
            st.rerun()
    with c2:
        if st.button("🔄 Reset Skor"):
            st.session_state.skor          = 0
            st.session_state.total_dijawab = 0
            st.session_state.feedback      = ""
            st.session_state.soal_list     = init_state()
            st.session_state.current_soal  = 0
            st.rerun()

# -------- PANEL KIRI (KAMERA) --------
with col_cam:
    cam_placeholder = st.empty()

    if not st.session_state.camera_on:
        cam_placeholder.markdown("""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 2px dashed rgba(0,200,255,0.2);
            border-radius: 16px;
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #3a5a6a;
            font-size: 1rem;
            text-align: center;
            gap: 1rem;
        ">
            <span style="font-size:3rem">📷</span>
            <span>Tekan <b style="color:#00c8ff">Mulai Kamera</b> untuk memulai</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Audio soal pertama kali
        if "audio_soal_played" not in st.session_state or st.session_state.get("audio_soal_idx") != st.session_state.current_soal:
            soal_teks = soal_skr.replace("+", " tambah ").replace("-", " kurang ").replace("0","nol")
            b64 = get_audio_b64(f"Berapakah {soal_teks}?")
            play_audio_html(b64)
            st.session_state.audio_soal_idx = st.session_state.current_soal

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with mp_hands.Hands(
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
            max_num_hands=2
        ) as hands_model:

            FRAME_LIMIT = 300  # max frame sebelum auto-stop
            frame_count = 0

            while st.session_state.camera_on and frame_count < FRAME_LIMIT:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Kamera tidak terdeteksi.")
                    break

                frame, total_isyarat, tangan_muncul = proses_frame(frame, hands_model)
                current_time = time.time()

                # Logika deteksi stabil
                if tangan_muncul:
                    if total_isyarat != st.session_state.last_fingers:
                        st.session_state.last_fingers  = total_isyarat
                        st.session_state.stable_start  = current_time
                        st.session_state.just_answered = False

                    elif st.session_state.stable_start and not st.session_state.just_answered:
                        elapsed  = current_time - st.session_state.stable_start
                        progress = min(int(elapsed / 1.5 * 100), 100)

                        with col_panel:
                            isyarat_ph.markdown(f'<div class="isyarat-badge">Isyarat Terbaca: {total_isyarat}</div>', unsafe_allow_html=True)
                            progress_ph.markdown(f"""
                            <div class="scanning-bar-wrap">
                                <div class="scanning-bar-fill" style="width:{progress}%"></div>
                            </div>
                            """, unsafe_allow_html=True)

                        if elapsed > 1.5:
                            soal = st.session_state.soal_list[st.session_state.current_soal]
                            if "+" in soal:
                                a, b = map(int, soal.split("+"))
                                jawaban_benar = a + b
                            else:
                                a, b = map(int, soal.split("-"))
                                jawaban_benar = a - b

                            st.session_state.total_dijawab += 1
                            st.session_state.just_answered  = True
                            st.session_state.stable_start   = None

                            if total_isyarat == jawaban_benar:
                                st.session_state.feedback     = "BENAR! 👍"
                                st.session_state.skor        += 1
                                st.session_state.current_soal = (st.session_state.current_soal + 1) % len(st.session_state.soal_list)
                                audio_b64 = get_audio_b64("Benar")
                                st.session_state.audio_soal_idx = -1  # trigger soal baru
                            else:
                                st.session_state.feedback = f"SALAH! Jawaban: {jawaban_benar}"
                                audio_b64 = get_audio_b64("Salah, ayo coba lagi")

                            with col_panel:
                                if "BENAR" in st.session_state.feedback:
                                    feedback_ph.markdown(f'<div class="feedback-benar">✅ {st.session_state.feedback}</div>', unsafe_allow_html=True)
                                else:
                                    feedback_ph.markdown(f'<div class="feedback-salah">❌ {st.session_state.feedback}</div>', unsafe_allow_html=True)
                            play_audio_html(audio_b64)
                            time.sleep(1.5)
                            st.rerun()
                else:
                    st.session_state.stable_start = None
                    with col_panel:
                        isyarat_ph.empty()
                        progress_ph.empty()

                # Tampilkan frame kamera
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                frame_count += 1

        cap.release()
