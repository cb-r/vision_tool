import torch
from datetime import datetime
import platform
import psutil
import streamlit as st


try:
    import GPUtil
    gpus_available = True
except ImportError:
    gpus_available = False

def timestamped_message(message):
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return f"{current_time} : {message}"

# Function to load the entire model from a file
def load_entire_model(model_name):
    return torch.load(f'{model_name}.pth', map_location=torch.device('cpu'))


# Function to display system information
def system_info():
    with st.expander("Click here to see System Information"):
        st.subheader('CPU Information')
        st.text(f"Physical cores: {psutil.cpu_count(logical=False)}")
        st.text(f"Total cores: {psutil.cpu_count(logical=True)}")
        st.text(f"Max Frequency: {psutil.cpu_freq().max:.2f}Mhz")
        st.text(f"Min Frequency: {psutil.cpu_freq().min:.2f}Mhz")
        st.text(f"Current Frequency: {psutil.cpu_freq().current:.2f}Mhz")
        st.text(f"Total CPU Usage: {psutil.cpu_percent()}%")

        st.subheader('RAM Information')
        ram = psutil.virtual_memory()
        st.text(f"Total: {ram.total / (1024 ** 3):.2f} GB")
        st.text(f"Available: {ram.available / (1024 ** 3):.2f} GB")
        st.text(f"Used: {ram.used / (1024 ** 3):.2f} GB")
        st.text(f"Percentage: {ram.percent}%")

        if gpus_available:
            gpus = GPUtil.getGPUs()
            if gpus:
                st.subheader('GPU Information')
                for gpu in gpus:
                    st.text(f"ID: {gpu.id}, Name: {gpu.name}")
                    st.text(f"Load: {gpu.load*100}%")
                    st.text(f"Free Memory: {gpu.memoryFree}MB")
                    st.text(f"Used Memory: {gpu.memoryUsed}MB")
                    st.text(f"Total Memory: {gpu.memoryTotal}MB")
                    st.text(f"Temperature: {gpu.temperature} °C")
        else:
            st.error("GPUtil module required for GPU info. Install with `pip install gputil` if you have a GPU.")
        st.subheader('Operating System Information')
        st.text(f"System: {platform.system()}")
        st.text(f"Node Name: {platform.node()}")
        st.text(f"Release: {platform.release()}")
        st.text(f"Version: {platform.version()}")
        st.text(f"Machine: {platform.machine()}")
        st.text(f"Processor: {platform.processor()}")
