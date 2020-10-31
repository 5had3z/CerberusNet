#!/usr/bin/env python3.8

import os
import re
from pathlib import Path

from .CityScapes import CityScapesDataset

IMG_EXT = '.png'

class CityScapesDemo(CityScapesDataset):
    def __init__(self, directory: Path, output_size=(1024, 512), **kwargs):
        super(CityScapesDemo, self).__init__(directories={}, output_size=output_size)
        self.l_img = []
        self.l_seq = []

        for filename in os.listdir(directory):
            if filename.endswith(IMG_EXT):
                frame_n = int(re.split("_", filename)[2])
                seq_name = filename.replace(
                    str(frame_n).zfill(6), str(frame_n+1).zfill(6))

                if os.path.isfile(seq_name):
                    self.l_img.append(filename)
                    self.l_seq.append(seq_name)

if __name__ == "__main__":
    raise NotImplementedError
