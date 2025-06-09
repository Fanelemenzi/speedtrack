import streamlit as st
import cv2
import tempfile
import os
import ultralytics
from ultralytics import solutions
import numpy as np
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Speed Estimation Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Vehicle Speed Estimation Dashboard")
st.markdown("Upload a video to analyze vehicle speeds using YOLO11")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_options = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
selected_model = st.sidebar.selectbox("Select YOLO Model", model_options, index=0)

# Speed estimation parameters
st.sidebar.subheader("Speed Estimation Parameters")
max_speed = st.sidebar.number_input("Max Speed (km/h)", value=120, min_value=10, max_value=300, step=10)
max_hist = st.sidebar.number_input("Min Frames for Tracking", value=5, min_value=1, max_value=20, step=1)
meter_per_pixel = st.sidebar.number_input("Meter per Pixel", value=0.05, min_value=0.01, max_value=1.0, step=0.01, format="%.3f")

# Display options
st.sidebar.subheader("Display Options")
show_output = st.sidebar.checkbox("Show Real-time Output", value=True)
line_width = st.sidebar.slider("Bounding Box Line Width", min_value=1, max_value=10, value=2)

# Class selection
st.sidebar.subheader("Object Classes")
class_options = {
    "All Classes": None,
    "Vehicles Only": [0, 1, 2, 3, 5, 7],  # person, bicycle, car, motorcycle, bus, truck
    "Cars Only": [2],
    "Cars and Trucks": [2, 7],
    "Custom": "custom"
}
selected_classes = st.sidebar.selectbox("Select Classes to Track", list(class_options.keys()))

if selected_classes == "Custom":
    custom_classes = st.sidebar.text_input("Enter class IDs (comma-separated)", "0,2")
    try:
        class_ids = [int(x.strip()) for x in custom_classes.split(",")]
    except:
        class_ids = [0, 2]
        st.sidebar.error("Invalid class IDs, using default [0, 2]")
else:
    class_ids = class_options[selected_classes]

# File upload
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    # Display video info
    cap = cv2.VideoCapture(temp_video_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        # Display video information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", f"{w}x{h}")
        with col2:
            st.metric("FPS", fps)
        with col3:
            st.metric("Duration", f"{duration:.1f}s")
        with col4:
            st.metric("Frames", frame_count)
        
        cap.release()
        
        # Process button
        if st.button("🚀 Start Speed Estimation", type="primary"):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create columns for display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                video_placeholder = st.empty()
            with col2:
                stats_placeholder = st.empty()
            
            try:
                # Initialize video capture
                cap = cv2.VideoCapture(temp_video_path)
                
                # Create output video path
                output_path = tempfile.mktemp(suffix='.mp4')
                
                # Video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                
                # Initialize speed estimation object
                speedestimator = solutions.SpeedEstimator(
                    show=False,  # We'll handle display ourselves
                    model=selected_model,
                    fps=fps,
                    max_speed=max_speed,
                    max_hist=max_hist,
                    meter_per_pixel=meter_per_pixel,
                    classes=class_ids,
                    line_width=line_width,
                )
                
                frame_num = 0
                detected_speeds = []
                
                # Process video frame by frame
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Run speed estimation
                    results = speedestimator(frame)
                    processed_frame = results.plot_im
                    
                    # Write frame to output video
                    video_writer.write(processed_frame)
                    
                    # Update progress
                    progress = (frame_num + 1) / frame_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_num + 1}/{frame_count}")
                    
                    # Display frame every 10 frames to avoid overwhelming
                    if frame_num % 10 == 0 and show_output:
                        # Convert BGR to RGB for display
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(display_frame, channels="RGB", use_column_width=True)
                    
                    # Extract speed information if available
                    if hasattr(results, 'speed_data') and results.speed_data:
                        for speed in results.speed_data:
                            if speed > 0:  # Only add positive speeds
                                detected_speeds.append(speed)
                    
                    # Update statistics
                    if detected_speeds:
                        avg_speed = np.mean(detected_speeds)
                        max_detected = max(detected_speeds)
                        min_detected = min(detected_speeds)
                        
                        stats_placeholder.markdown(f"""
                        ### 📊 Speed Statistics
                        - **Average Speed**: {avg_speed:.1f} km/h
                        - **Max Speed**: {max_detected:.1f} km/h
                        - **Min Speed**: {min_detected:.1f} km/h
                        - **Detections**: {len(detected_speeds)}
                        """)
                    
                    frame_num += 1
                
                # Clean up
                cap.release()
                video_writer.release()
                
                # Success message
                st.success("✅ Video processing completed!")
                
                # Provide download link for processed video
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="speed_estimation_output.mp4",
                        mime="video/mp4"
                    )
                    
                    # Clean up temporary files
                    os.unlink(output_path)
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.info("Make sure you have the required dependencies installed: `pip install ultralytics opencv-python`")
        
        # Clean up temporary input file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
    
    else:
        st.error("❌ Could not read the uploaded video file. Please check the file format.")

else:
    # Display instructions when no file is uploaded
    st.info("👆 Please upload a video file to begin speed estimation analysis.")
    
    # Show example configuration
    with st.expander("ℹ️ Configuration Guide"):
        st.markdown("""
        ### Model Selection
        - **yolo11n.pt**: Fastest, least accurate
        - **yolo11s.pt**: Balanced speed and accuracy
        - **yolo11m.pt**: Medium accuracy and speed
        - **yolo11l.pt**: High accuracy, slower
        - **yolo11x.pt**: Highest accuracy, slowest
        
        ### Speed Parameters
        - **Max Speed**: Cap speeds to avoid outliers
        - **Min Frames for Tracking**: Minimum frames before calculating speed
        - **Meter per Pixel**: Calibration factor (depends on camera setup)
        
        ### Class IDs (COCO Dataset)
        - 0: Person
        - 1: Bicycle  
        - 2: Car
        - 3: Motorcycle
        - 5: Bus
        - 7: Truck
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and YOLO11 • Upload a video to get started!")