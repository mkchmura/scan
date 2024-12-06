import cv2
from PIL import Image
from transformers import pipeline
import threading
from queue import Queue
import concurrent.futures
import torch
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class CameraConfig:
    url: str
    name: str
    process_interval: int = 1  # Process every nth frame

class VideoProcessor:
    def __init__(self, batch_size: int = 4, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.frame_queue = Queue(maxsize=100)
        self.result_queue = Queue()
        self.active = True
        
        # Initialize object detection model on GPU if available
        device = 0 if torch.cuda.is_available() else -1
        self.object_detector = pipeline('object-detection', 
                                     model='facebook/detr-resnet-50', 
                                     device=device)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.ignored_objects = {
            'traffic light', 'fire hydrant', 'tv', 'television', 
            'refrigerator', 'bench', 'bird'
        }

    def process_camera_feed(self, camera_config: CameraConfig):
        """Process individual camera feed and add frames to queue"""
        cap = cv2.VideoCapture(camera_config.url)
        if not cap.isOpened():
            logging.error(f"Failed to open camera: {camera_config.name}")
            return

        frame_counter = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            while self.active:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame from {camera_config.name}")
                    time.sleep(1)  # Prevent busy waiting
                    continue

                frame_counter += 1
                if frame_counter % camera_config.process_interval == 0:
                    self.frame_queue.put({
                        'frame': frame,
                        'camera_name': camera_config.name,
                        'timestamp': time.time()
                    })
                
        except Exception as e:
            logging.error(f"Error processing camera {camera_config.name}: {str(e)}")
        finally:
            cap.release()

    def process_frame_batch(self, frame_batch: List[Dict[str, Any]]):
        """Process a batch of frames through the object detector"""
        try:
            # Convert frames to PIL Images
            images = [
                Image.fromarray(cv2.cvtColor(item['frame'], cv2.COLOR_BGR2RGB))
                for item in frame_batch
            ]
            
            # Batch inference
            batch_results = self.object_detector(images)
            
            # Process results for each frame
            for idx, (item, results) in enumerate(zip(frame_batch, batch_results)):
                relevant_detections = [
                    result for result in results 
                    if result["score"] > 0.90 and 
                    result["label"].lower() not in self.ignored_objects
                ]
                
                if relevant_detections:
                    self.result_queue.put({
                        'camera_name': item['camera_name'],
                        'timestamp': item['timestamp'],
                        'detections': relevant_detections
                    })
                
        except Exception as e:
            logging.error(f"Error in batch processing: {str(e)}")

    def process_results(self):
        """Process and log detection results"""
        while self.active:
            try:
                result = self.result_queue.get(timeout=1)
                logging.info(
                    f"Camera: {result['camera_name']} - "
                    f"Time: {time.strftime('%H:%M:%S', time.localtime(result['timestamp']))} - "
                    f"Detections: {len(result['detections'])}"
                )
                for detection in result['detections']:
                    logging.info(f"  {detection['label']}: {detection['score']:.2f}")
            except:
                continue

    def run(self, camera_configs: List[CameraConfig]):
        """Main method to run the video processing system"""
        try:
            # Start camera feed threads
            camera_threads = []
            for config in camera_configs:
                thread = threading.Thread(
                    target=self.process_camera_feed,
                    args=(config,)
                )
                thread.daemon = True
                thread.start()
                camera_threads.append(thread)

            # Start result processing thread
            result_thread = threading.Thread(target=self.process_results)
            result_thread.daemon = True
            result_thread.start()

            # Main processing loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while self.active:
                    frame_batch = []
                    # Collect frames for batch processing
                    while len(frame_batch) < self.batch_size:
                        try:
                            frame_data = self.frame_queue.get(timeout=1)
                            frame_batch.append(frame_data)
                        except:
                            break
                    
                    if frame_batch:
                        executor.submit(self.process_frame_batch, frame_batch)

        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            self.active = False
            # Wait for threads to finish
            for thread in camera_threads:
                thread.join()
            result_thread.join()

if __name__ == '__main__':
    # Example camera configurations
    cameras = [
        CameraConfig(
            url='https://wzmedia.dot.ca.gov/D8/LB-8_18_125.stream/playlist.m3u8',
            name='Camera 1'
        ),
        CameraConfig(
            url='https://wzmedia.dot.ca.gov/D8/LB-8_18_125.stream/playlist.m3u8',
            name='Camera 2'
        ),
        # Add more camera configurations as needed
    ]

    processor = VideoProcessor(batch_size=4, max_workers=4)
    processor.run(cameras)