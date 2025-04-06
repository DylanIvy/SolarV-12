# dataset.py
import torch
from torch.utils.data import Dataset
from datetime import timedelta
from config import num_videos


class SolarRadiationImageDataset(Dataset):
    def __init__(self, dataframe, video_frames, video_start_time, frame_rate, forecast_offset=5, image_transform=None,cloud_coverage_data=None):
        """
        Custom Dataset for forecasting future solar radiation from a timelapse video.

        Each sample includes:
          - The current timestamp (as a string),
          - The current solar radiation reading,
          - The image corresponding to the computed frame for the current time,
          - The target: the solar radiation reading forecast_offset rows in the future.

        This version maps the full time range of the CSV (which may span a full day)
        onto the timelapse video (which is very short) by scaling the time differences.

        Args:
            dataframe (pd.DataFrame): CSV data that must include "Timestamp" (datetime)
                                      and "Solar Radiation Sensor".
            video_frames (np.ndarray): Array of video frames (shape: (num_frames, H, W, 3)).
            video_start_time (datetime): A reference time for the video playback.
            frame_rate (float): The video’s frame rate (frames per second).
            forecast_offset (int): How many CSV rows ahead to use as the forecast target.
            image_transform (callable, optional): Transformations for the image.
        """
        # Sort the CSV data chronologically
        self.dataframe = dataframe.sort_values(by="Timestamp").reset_index(drop=True)
        self.video_frames = video_frames
        self.video_start_time = video_start_time
        self.frame_rate = frame_rate
        self.forecast_offset = forecast_offset
        self.image_transform = image_transform
        self.cloud_coverage_data = cloud_coverage_data

        # Get the actual time span from the CSV data:
        self.day_start = self.dataframe["Timestamp"].min()
        self.day_end = self.dataframe["Timestamp"].max()
        self.total_actual_seconds = (self.day_end - self.day_start).total_seconds()
        self.total_video_frames = len(video_frames)

        # Adjust dataset length so every sample has a future target.
        self.length = len(self.dataframe) - self.forecast_offset

    def __len__(self):
        return min(len(self.dataframe) - self.forecast_offset, len(self.cloud_coverage_data) - self.forecast_offset)

    def __getitem__(self, idx):
        # Get the current CSV row.
        row = self.dataframe.iloc[idx]
        csv_time = row["Timestamp"]
        current_sensor = row["Solar Radiation Sensor"]
        if idx >= len(self.cloud_coverage_data):
            current_cloud_coverage = 0.0  # Default value if out of range
        else:
            current_cloud_coverage = self.cloud_coverage_data[idx]
        df = self.dataframe
        day_start = self.day_start
        active_period_seconds = 43200
        skipped_period_seconds = 43200  # Nighttime skipped

        # Total actual seconds, considering the skipped nighttime periods
        total_actual_seconds = num_videos * active_period_seconds
        elapsed_seconds = (csv_time - day_start).total_seconds()
        video_index = int(elapsed_seconds // (active_period_seconds + skipped_period_seconds))  # Determine which video
        within_video_seconds = elapsed_seconds % (active_period_seconds + skipped_period_seconds)

        # If within nighttime period, adjust to the next active period
        if within_video_seconds >= active_period_seconds:
            video_index += 1
            within_video_seconds -= active_period_seconds + skipped_period_seconds

        # Compute the adjusted elapsed time within the active periods
        adjusted_elapsed_seconds = video_index * active_period_seconds + max(0, within_video_seconds)

        # Compute frame index
        computed_frame_index = round((adjusted_elapsed_seconds / total_actual_seconds) * self.total_video_frames)
        computed_frame_index = max(0, min(computed_frame_index, self.total_video_frames - 1))
        video_frame_time = self.video_start_time + timedelta(seconds=(computed_frame_index / self.frame_rate))

        # Retrieve the video frame (the timelapse representation for the current timestamp)
        image = self.video_frames[computed_frame_index]
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Get the future sensor measurement as the forecast target.
        future_sensor = self.dataframe.iloc[idx + self.forecast_offset]["Solar Radiation Sensor"]
        if (idx + self.forecast_offset) >= len(self.cloud_coverage_data):
            future_cloud_coverage = self.cloud_coverage_data[-1]  # Use last available value
        else:
            future_cloud_coverage = self.cloud_coverage_data[idx + self.forecast_offset]

        sample = {
            "timestamp": str(csv_time),
            "current_sensor": current_sensor,
            "image": image,
            "current_cloud_coverage": current_cloud_coverage,  # New feature
            "target": torch.tensor([future_sensor, future_cloud_coverage], dtype=torch.float32),
            # Including debug info below (optional)
            "computed_frame_index": computed_frame_index,
            "computed_video_time": str(video_frame_time)
        }
        return sample