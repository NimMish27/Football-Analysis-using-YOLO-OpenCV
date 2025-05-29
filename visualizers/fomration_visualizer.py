import cv2
import numpy as np

class FormationVisualizer:
    def __init__(self, field_image_path, field_size=(105, 68)):
        """
        Args:
            field_image_path (str): Path to the image of the football field.
            field_size (tuple): Real-world dimensions of a football field in meters (length, width).
        """
        self.field_image_path = field_image_path
        self.field_size = field_size
        self.field_image = cv2.imread(field_image_path)
        if self.field_image is None:
            raise FileNotFoundError(f"Field image not found at {field_image_path}")
        self.image_height, self.image_width = self.field_image.shape[:2]

        # Team colors (BGR)
        self.team_colors = {
            1: (0, 0, 255),     # Red
            2: (0, 255, 255),   # Yellow
        }
        self.default_color = (255, 255, 255)  # White

    def _transform_to_field_coordinates(self, point, frame_shape):
        """
        Transform coordinates from video frame to field image space.
        """
        frame_height, frame_width = frame_shape[:2]
        fx = self.image_width / frame_width
        fy = self.image_height / frame_height
        x, y = point
        return int(x * fx), int(y * fy)

    def draw_formation(self, player_detections, frame_shape):
        """
        Draw players on the football field image.

        Args:
            player_detections (dict): player_id -> {'team': int, 'center': (x, y)}
            frame_shape (tuple): Shape of the original frame

        Returns:
            np.ndarray: Field image with players drawn
        """
        field_vis = self.field_image.copy()

        for player_id, player in player_detections.items():
            team = player.get('team')
            center = player.get('center')

            if center is None or team is None:
                continue

            field_x, field_y = self._transform_to_field_coordinates(center, frame_shape)
            color = self.team_colors.get(team, self.default_color)

            # Draw circle and ID
            cv2.circle(field_vis, (field_x, field_y), 8, color, -1)
            cv2.putText(field_vis, str(player_id), (field_x + 10, field_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return field_vis
