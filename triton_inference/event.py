import time
from watchdog.observers import Observer
from watchdog.events import EVENT_TYPE_MODIFIED, FileSystemEventHandler
from pathlib import Path
import json
from functools import partial

class CfgHandler(FileSystemEventHandler):

    observer = Observer()
    cfg = None
    cfg_path = None

    action = {}

    def on_modified(self,  event):
        print(f'event type: {event.event_type} path : {event.src_path}')
        path = Path(event.src_path)
        if path.is_file():
            with open(path, "r") as fp:
                self.cfg = json.load(fp)

            self.action[event.event_type](self.cfg)

    # def  on_created(self,  event):
    #      print(f'event type: {event.event_type} path : {event.src_path}')
    #      self.action(event.event_type)(self.cfg)

    # def  on_deleted(self,  event):
    #      print(f'event type: {event.event_type} path : {event.src_path}')
    #      self.action(event.event_type)(self.cfg)
    
    def register_action(self, func, event_type):
        self.action[event_type] = func

    def begin_watch(self, path, recursive=False):
        self.cfg_path = path
        with open(path, "r") as fp:
            self.cfg = json.load(fp)
        self.observer.schedule(self, path=path, recursive=recursive)
        self.observer.start()

    def stop_watch(self):
        self.observer.stop()
        self.observer.join()


if __name__ ==  "__main__":

    import multiprocessing as mp

    v = mp.Value('i', 0)

    def foo(v, cfg):
        v.value += 1

    cfg_handler = CfgHandler()
    cfg_handler.register_action(partial(foo, v), EVENT_TYPE_MODIFIED)
    cfg_handler.begin_watch("/jmain01/home/JAD003/sxr06/lxx22-sxr06/model-inference/repository/meta.json")
    print(cfg_handler.cfg)
    time.sleep(30)
    print(cfg_handler.cfg)
    print(v.value)

    cfg_handler.stop_watch()