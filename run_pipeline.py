import subprocess

print("🚀 [1/2] Running real-time tracker...")
subprocess.run(["python", "realtime_activity_tracker.py"])

print("\n📊 [2/2] Launching Streamlit dashboard...")
subprocess.run(["python", "-m", "streamlit", "run", "streamlit_dashboard.py"])
