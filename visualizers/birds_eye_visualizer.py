import cv2
import numpy as np
from collections import defaultdict, deque

class BirdsEyeVisualizer:
    def __init__(self, field_length=105, field_width=68, image_size=(720, 480), trail_length=15, field_image_path='assests/football_field_with_goals.jpg'):
        self.field_length = field_length
        self.field_width = field_width
        self.image_size = image_size
        self.trail_length = trail_length
        self.player_trails = defaultdict(lambda: deque(maxlen=trail_length))

        # Load and resize soccer field background
        field_img = cv2.imread(field_image_path)
        self.background = cv2.resize(field_img, self.image_size)

    def draw_frame(self, player_detections, ball_detections):
        # Start with soccer field background
        image = self.background.copy()

        for player_id, track in player_detections.items():
            if 'transformed_center' not in track:
                continue
            x, y = track['transformed_center']
            px = int(x / self.field_length * self.image_size[0])
            py = int(y / self.field_width * self.image_size[1])

            self.player_trails[player_id].append((px, py))

            trail_points = list(self.player_trails[player_id])
            for i in range(1, len(trail_points)):
                alpha = int(255 * (i / len(trail_points)))
                color = track.get('team_color', (255, 255, 255))
                faded_color = tuple(int(c * alpha / 255) for c in color)
                cv2.line(image, trail_points[i - 1], trail_points[i], faded_color, 2)

            has_ball = track.get('has_ball', False)
            if has_ball:
                cv2.circle(image, (px, py), 10, (0, 255, 255), -1)
            else:
                color = track.get('team_color', (255, 255, 255))
                cv2.circle(image, (px, py), 8, color, -1)

        # Draw ball
        if 'transformed_center' in ball_detections[1]:
            bx, by = ball_detections[1]['transformed_center']
            bx = int(bx / self.field_length * self.image_size[0])
            by = int(by / self.field_width * self.image_size[1])
            cv2.circle(image, (bx, by), 6, (255, 255, 255), -1)

        return image
