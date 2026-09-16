#----- Libraries
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import time
import logging
import glob
import os
import def_combine_df as cb
#-----

# ---- Set up logging
logging.basicConfig(
    filename="edit_log.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)
# ---- 


# New class that inherits from FileSystemEventHandler
class WatchBirdBoard: 
    watchDirectory = "/Users/robgoodsell/Library/CloudStorage/OneDrive-UWEBristol/WECS_bird_board/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        handler = Handler()
        self.observer.schedule(handler, self.watchDirectory, recursive=True)
        self.observer.start()

        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()

        self.observer.join()


# Another new class that handles the actions
class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            return
        
        if event.event_type == "deleted":
            logging.info("Deleted:" + event.src_path)
            cb.combine_df("data/wecs_birds.csv")

        if event.event_type == "created":
            logging.info("Created:" + event.src_path)
            cb.combine_df("data/wecs_birds.csv")


        elif event.event_type == "modified":
            logging.info("Modified:" + event.src_path)
            cb.combine_df("data/wecs_birds.csv")


            

if __name__ == "__main__":
    watch = WatchBirdBoard()
    watch.run()
