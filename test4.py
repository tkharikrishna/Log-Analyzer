import os
import re
import streamlit as st
import pandas as pd
from collections import Counter
st.set_page_config(layout="wide")
st.title("📡 AI-Powered Log Anomaly Detector (Event Summary)")
# Step 1: Read log lines from "logs/" folder
log_folder = "logs"
log_lines = []
for file_name in os.listdir(log_folder):
   if file_name.endswith(".txt"):
       with open(os.path.join(log_folder, file_name), "r", encoding="utf-8", errors="ignore") as file:
           lines = [line.strip() for line in file.readlines()]
           log_lines.extend(lines)
if not log_lines:
   st.warning("No logs found in the 'logs/' folder.")
   st.stop()
# Step 2: Define keywords
keywords = ["timeout", "reset", "down", "error", "alarm", "unreachable", "fail", "reject"]
ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
rp_pattern = r"RP\s*\d+"
# Step 3: Detect anomalies
anomalies = []
ip_counts = Counter()
rp_counts = Counter()
for i, line in enumerate(log_lines):
   if any(k in line.lower() for k in keywords):
       timestamp = ""
       if i >= 3:
           for offset in range(1, 4):
               if "AP time:" in log_lines[i - offset]:
                   timestamp_match = re.search(r"AP time:\s*(\d{8}_\d{6})", log_lines[i - offset])
                   if timestamp_match:
                       timestamp = timestamp_match.group(1)
                       break
       ip_matches = re.findall(ip_pattern, line)
       rp_matches = re.findall(rp_pattern, line, re.IGNORECASE)
       for ip in ip_matches:
           ip_counts[ip] += 1
       for rp in rp_matches:
           rp_counts[rp.upper()] += 1
       anomalies.append({
           "timestamp": timestamp,
           "log": line
       })
# Step 4: User-triggered display
if st.button("🔍 Show Anomalies"):
   st.subheader("📄 Detected Anomalies")
   st.write(f"Total anomalies found: {len(anomalies)}")
   st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
   st.subheader("📈 Summary Insights")
   if ip_counts:
       top_ip = ip_counts.most_common(1)[0]
       st.write(f"🔹 IP with most anomalies: `{top_ip[0]}` occurred `{top_ip[1]}` times")
   if rp_counts:
       top_rp = rp_counts.most_common(1)[0]
       st.write(f"🔹 RP/Link with most fluctuations: `{top_rp[0]}` occurred `{top_rp[1]}` times")
   if not ip_counts and not rp_counts:
       st.info("No IPs or RPs matched anomaly patterns.")