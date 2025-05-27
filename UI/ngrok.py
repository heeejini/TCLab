# filename: launch_streamlit.py

import os
import time
from pyngrok import ngrok

streamlit_script = "gui_streamlit.py"

public_url = ngrok.connect(8501)

print(f"✅ 외부 접속 주소: {public_url}")

os.system(f"streamlit run {streamlit_script} --server.port 8501")
