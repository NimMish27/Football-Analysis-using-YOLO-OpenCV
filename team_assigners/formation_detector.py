import numpy as np
import cv2
from sklearn.cluster import KMeans

class FormationDetector:
    def __init__(self, team_assigner):
        self.team_assigner = team_assigner

    def get_team_player_positions(self, frame, player_detections):
        team_players = {1: [], 2: []}
        for player_id, player in player_detections.items():
            bbox = player["bbox"]
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)
            team_id = self.team_assigner.get_player_team(frame, bbox, player_id)
            team_players[team_id].append((x_center, y_center))
        return team_players

    def detect_formation(self, frame, player_detections):
        team_players = self.get_team_player_positions(frame, player_detections)
        formations = {}

        for team_id, players in team_players.items():
            if len(players) < 3:
                formations[team_id] = "Unknown"
                continue

            y_coords = np.array([y for (_, y) in players]).reshape(-1, 1)
            n_clusters = min(3, len(players))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(y_coords)
            cluster_labels = kmeans.labels_

            # Count players in each horizontal layer (defense to attack)
            line_counts = [0] * n_clusters
            for label in cluster_labels:
                line_counts[label] += 1

            # Sort lines from top to bottom
            centers = kmeans.cluster_centers_.flatten()
            sorted_lines = [x for _, x in sorted(zip(centers, line_counts))]

            formation = "-".join(str(x) for x in sorted_lines)
            formations[team_id] = formation

        return formations

    def draw_formation(self, frame, player_detections):
        team_players = self.get_team_player_positions(frame, player_detections)
        formations = self.detect_formation(frame, player_detections)
        colors = {1: (0, 0, 255), 2: (0, 255, 255)}  # Team 1 red, Team 2 yellow

        # Draw players only (no lines)
        for team_id, players in team_players.items():
            if len(players) < 3:
                continue

            for x, y in players:
                cv2.circle(frame, (int(x), int(y)), 6, colors[team_id], -1)

        # Draw text overlay (formation info) at bottom-left corner
        height = frame.shape[0]
        y_offset = height - 60  # Start from bottom

        for team_id, formation in formations.items():
            text = f"Team {team_id} Formation: {formation}"
            color = colors[team_id]
            cv2.putText(frame, text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            y_offset += 40  # Stack upward if both teams

        return frame
