import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')  # Replace with the path to your model

class_colors = {
    "Merah": (0, 255, 0),  # Green
    "Hijau": (0, 0, 255),  # Red
}

# Path to your video
video_path = 'video.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video was loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_path = 'output_video.mp4'  # Path to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no more frames are available

    # Perform object detection on the current frame
    results = model.predict(source=frame, conf=0.2, stream=True)

    # Lists to store bounding boxes for red and green objects
    red_boxes = []
    green_boxes = []

    # Process detection results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Get class indices
        names = result.names  # Get class names

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            tag = f"{names[int(cls)]}"
            label = f"{tag} {score:.2f}"

            # Use class-specific color or default color
            color = class_colors.get(names[int(cls)], (0, 255, 255))  # Yellow if no match

            # Add boxes to respective lists
            if tag == "Merah":
                red_boxes.append((x1, y1, x2, y2))
            elif tag == "Hijau":
                green_boxes.append((x1, y1, x2, y2))

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def calculate_area(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    # Find the largest box for red and green
    if red_boxes:
        largest_red_box = max(red_boxes, key=calculate_area)
        xr, yr, x2r, y2r = largest_red_box
        cen_xr = (xr + x2r) // 2
        cen_yr = (yr + y2r) // 2
    else:
        cen_xr, cen_yr = None, None

    if green_boxes:
        largest_green_box = max(green_boxes, key=calculate_area)
        xg, yg, x2g, y2g = largest_green_box
        cen_xg = (xg + x2g) // 2
        cen_yg = (yg + y2g) // 2
    else:
        cen_xg, cen_yg = None, None

    # Draw the line between the largest red and green boxes if both exist
    if cen_xr is not None and cen_xg is not None:
        cv2.line(frame, (cen_xr, cen_yr), (cen_xg, cen_yg), (255, 0, 0), 3)
        mid_x = (cen_xr + cen_xg) // 2
        mid_y = (cen_yr + cen_yg) // 2
        cv2.circle(frame, (mid_x, mid_y), 6, (255, 255, 255), -1)

    # Draw the frame's midlines for reference
    frame_height, frame_width = frame.shape[:2]
    midline_left1 = frame_width // 2 - 160
    midline_right1 = frame_width // 2 + 160
    midline_left2 = frame_width // 2 - 40
    midline_right2 = frame_width // 2 + 40
    cv2.line(frame, (midline_left1, 0), (midline_left1, frame_height), (255, 255, 255), 2)
    cv2.line(frame, (midline_right1, 0), (midline_right1, frame_height), (255, 255, 255), 2)
    cv2.line(frame, (midline_left2, 0), (midline_left2, frame_height), (255, 255, 255), 2)
    cv2.line(frame, (midline_right2, 0), (midline_right2, frame_height), (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to: {output_path}")