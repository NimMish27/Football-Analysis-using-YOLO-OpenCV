from ultralytics import YOLO
model = YOLO('models/best.pt')

results = model.predict(r'C:\Users\nimis\Desktop\Football Analysis\input_videos\08fd33_4.mp4',save=True)
print(results[0])
for box in results[0].boxes:
    print(box)
    