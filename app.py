import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import tempfile
# Helper functions
def extract_timestamp_from_log(line):
   match = re.search(r'time: (\d{8}_\d{6})', line)
   if match:
       return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
   return None
def filter_logs_by_time(logs, start_time, end_time):
   return [log for log in logs if (ts := extract_timestamp_from_log(log)) and start_time <= ts <= end_time]
def train_xgboost_classifier(log_lines, labels):
   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(log_lines)
   model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
   model.fit(X, labels)
   return model, vectorizer
def predict_anomalies(log_lines, model, vectorizer):
   X = vectorizer.transform(log_lines)
   preds = model.predict(X)
   return [(line, i) for i, line in enumerate(log_lines) if preds[i] == 1]
# Streamlit UI
st.title("📡 Telecom Log Analyzer (Anomaly Detection + Natural Language Queries)")
uploaded_files = st.file_uploader("Upload your log files (.txt)", type=["txt"], accept_multiple_files=True)
if uploaded_files:
   all_logs = []
   for uploaded_file in uploaded_files:
       with tempfile.NamedTemporaryFile(delete=False) as tmp:
           tmp.write(uploaded_file.read())
           tmp_path = tmp.name
       with open(tmp_path, 'r', errors='ignore') as f:
           all_logs += [line.strip() for line in f.readlines()]
       os.remove(tmp_path)
   tabs = st.tabs(["📊 Detect Anomalies", "🔍 Time-based Log Queries"])
   with tabs[0]:
       st.subheader("Step 1: Train Model from Labeled Examples")
       training_data = [
           "Keepalive OK from IP 192.168.1.1",
           "Link status for RP 4 on plane B is DOWN",
           "Timeout occurred from 192.168.1.10",
           "Reset received from RP 2",
           "Config loaded successfully",
           "Ping to 192.168.1.2 successful"
       ]
       labels = [0, 1, 1, 1, 0, 0]
       model, vectorizer = train_xgboost_classifier(training_data, labels)
       if st.button("🚨 Detect Anomalies"):
           anomalies = predict_anomalies(all_logs, model, vectorizer)
           if anomalies:
               st.write("### Detected Anomalies:")
               for line, idx in anomalies:
                   ts_line = all_logs[idx - 3] if idx >= 3 else "Timestamp not found"
                   st.text(f"{ts_line}\n→ {line}\n")
           else:
               st.success("✅ No anomalies found.")
   with tabs[1]:
       st.subheader("Ask about logs in a specific time range")
       col1, col2 = st.columns(2)
       with col1:
           start = st.text_input("Start time (YYYYMMDD_HHMMSS)", "20240510_034000")
       with col2:
           end = st.text_input("End time (YYYYMMDD_HHMMSS)", "20240510_040000")
       if st.button("🔍 Show Logs in Time Range"):
           try:
               start_dt = datetime.strptime(start, "%Y%m%d_%H%M%S")
               end_dt = datetime.strptime(end, "%Y%m%d_%H%M%S")
               filtered = filter_logs_by_time(all_logs, start_dt, end_dt)
               st.write(f"### Found {len(filtered)} log lines in given range:")
               for line in filtered:
                   st.text(line)
           except Exception as e:
               st.error(f"Invalid format. Use YYYYMMDD_HHMMSS. Error: {e}")