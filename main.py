from utils.video_utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import pandas as pd
from team_assigners.team_assigner import TeamAssigner
from team_assigners.formation_detector import FormationDetector
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from visualizers.fomration_visualizer import FormationVisualizer
import os

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Add positions to tracks
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformation
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimation
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    player_movement_data = []
    ball_movement_data = []

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track['bbox'], player_id
            )
            track['team'] = team
            track['team_color'] = team_assigner.team_colors[team]

            # Add center point
            x1, y1, x2, y2 = track['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            track['center'] = (cx, cy)

            # Save player movement for heatmap
            player_movement_data.append({
                'frame': frame_num,
                'player_id': player_id,
                'team': team,
                'x': cx,
                'y': cy
            })

        # Save ball position
        ball_track = tracks['ball'][frame_num][1]
        ball_bbox = ball_track['bbox']
        bx = int((ball_bbox[0] + ball_bbox[2]) / 2)
        by = int((ball_bbox[1] + ball_bbox[3]) / 2)
        ball_movement_data.append({
            'frame': frame_num,
            'x': bx,
            'y': by
        })

    # Assign Ball Possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            player = tracks['players'][frame_num][assigned_player]
            player['has_ball'] = True
            team_ball_control.append(player['team'])
        else:
            last_team = team_ball_control[-1] if team_ball_control else 0
            team_ball_control.append(last_team)
    team_ball_control = np.array(team_ball_control)

    # Initialize Visualizers
    formation_visualizer = FormationVisualizer(field_image_path="assets/football_field_with_goals.jpg")
    formation_detector = FormationDetector(team_assigner)

    # Draw output frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Draw formation overlays directly on original frames
    for frame_num in range(len(output_video_frames)):
        output_video_frames[frame_num] = formation_detector.draw_formation(
            output_video_frames[frame_num], tracks['players'][frame_num]
        )

    # Combine with bird's-eye formation visualizer
    combined_frames = []
    for frame_num, frame in enumerate(output_video_frames):
        player_detections = tracks['players'][frame_num]

        player_input = {
            player_id: {
                'team': track['team'],
                'center': track['center']
            }
            for player_id, track in player_detections.items()
            if 'center' in track and 'team' in track
        }

        formation_frame = formation_visualizer.draw_formation(player_input, frame.shape)

        # Resize both frames to same height
        h = min(frame.shape[0], formation_frame.shape[0])
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
        formation_resized = cv2.resize(formation_frame, (int(formation_frame.shape[1] * h / formation_frame.shape[0]), h))

        # Side-by-side: Original + Formation View
        combined_frame = np.hstack((frame_resized, formation_resized))
        combined_frames.append(combined_frame)

    # Save final output video
    height, width = combined_frames[0].shape[:2]
    out = cv2.VideoWriter('output_videos/output_video_combined.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))
    for f in combined_frames:
        if f.shape[:2] != (height, width):
            f = cv2.resize(f, (width, height))
        out.write(f)
    out.release()

    # Save CSVs
    os.makedirs('output_data', exist_ok=True)
    pd.DataFrame(player_movement_data).to_csv('output_data/player_tracks.csv', index=False)
    pd.DataFrame(ball_movement_data).to_csv('output_data/ball_tracks.csv', index=False)
    print("[INFO] Player and Ball tracks saved to CSV.")

if __name__ == '__main__':
    main()
