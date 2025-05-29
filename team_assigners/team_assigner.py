from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)

        if image_2d.shape[0] == 0:
            raise ValueError("Empty image passed to clustering model.")

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        # Clip bbox to frame dimensions to avoid out-of-bounds slicing
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            print(f"Invalid bbox: {bbox}")
            return np.array([0, 0, 0])  # fallback color

        image = frame[y1:y2, x1:x2]

        # Get top half of image
        top_half_height = max(1, int(image.shape[0] / 2))
        top_half_image = image[:top_half_height, :]

        if top_half_image.size == 0:
            print(f"Empty top_half_image for bbox: {bbox}")
            return np.array([0, 0, 0])  # fallback color

        try:
            kmeans = self.get_clustering_model(top_half_image)
        except ValueError as e:
            print(f"KMeans failed for bbox {bbox}: {e}")
            return np.array([0, 0, 0])  # fallback color

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        player_colors = np.array(player_colors)

        # Remove all-zero colors (fallbacks)
        player_colors = player_colors[~np.all(player_colors == 0, axis=1)]

        if len(player_colors) < 2:
            print("Not enough valid players to assign team colors.")
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        if np.all(player_color == 0):
            return -1  # fallback if player color couldn't be determined

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        # Special rule for ID 91
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
