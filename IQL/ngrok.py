# filename: launch_streamlit.py

import os
import time
from pyngrok import ngrok

# 1. streamlit 앱 경로
streamlit_script = "gui_streamlit.py"

# 2. ngrok 연결 (포트 8501)
public_url = ngrok.connect(8501)

print(f"✅ 외부 접속 주소: {public_url}")

# 3. streamlit 실행 (백그라운드 아님)
os.system(f"streamlit run {streamlit_script} --server.port 8501")
