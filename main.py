"""
Smart Hand-Controlled Volume Adjuster
Author: Mehdi Anvari
Date: 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import math
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

VOLUME_LOCK_DURATION = 3
is_volume_locked = False
lock_start_time = 0
base_ratio = None
last_vol_percent = 50 

def calculate_hand_ratio(landmarks, image_shape):
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    hand_length = math.hypot(
        (wrist.x - middle_mcp.x) * image_shape[1],
        (wrist.y - middle_mcp.y) * image_shape[0]
    )
    
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    finger_distance = math.hypot(
        (thumb_tip.x - index_tip.x) * image_shape[1],
        (thumb_tip.y - index_tip.y) * image_shape[0]
    )
    
    return finger_distance / hand_length if hand_length > 0 else 0

def is_hand_open(landmarks):
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    for tip in tips:
        pip = tip - 2
        tip_y = landmarks.landmark[tip].y
        pip_y = landmarks.landmark[pip].y
        if tip_y > pip_y + 0.05:
            return False
    return True

def update_volume(current_ratio):
    global base_ratio, last_vol_percent
    
    if base_ratio is None:
        base_ratio = current_ratio
        return last_vol_percent
    
    ratio_change = current_ratio / base_ratio
    vol = np.interp(ratio_change, [0.5, 2.0], [min_vol, max_vol])
    volume.SetMasterVolumeLevel(vol, None)
    
    last_vol_percent = np.interp(vol, [min_vol, max_vol], [0, 100])
    return last_vol_percent

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    
    current_vol_percent = last_vol_percent
    hand_detected = False
    
    if results.multi_hand_landmarks:
        hand_detected = True
        hand_landmarks = results.multi_hand_landmarks[0]
        
        hand_open = is_hand_open(hand_landmarks)
        current_ratio = calculate_hand_ratio(hand_landmarks, image.shape)
        
        if hand_open:
            if not is_volume_locked:
                is_volume_locked = True
                lock_start_time = time.time()
                base_ratio = None
        else:
            if is_volume_locked:
                if time.time() - lock_start_time > VOLUME_LOCK_DURATION:
                    is_volume_locked = False
            else:
                current_vol_percent = update_volume(current_ratio)

    if hand_detected:
        display_vol = last_vol_percent if is_volume_locked else current_vol_percent
        
        cv2.rectangle(image, (50, 150), (85, 350), (0, 255, 0), 2)
        vol_height = int(200 * (display_vol/100))
        cv2.rectangle(image, (50, 350 - vol_height), (85, 350), (0, 255, 0), -1)
        
        cv2.putText(image, f"{int(display_vol)}%", (40, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if is_volume_locked:
            remaining = int(VOLUME_LOCK_DURATION - (time.time() - lock_start_time))
            cv2.putText(image, f"LOCKED ({remaining}s)", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Smart Volume Control', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
