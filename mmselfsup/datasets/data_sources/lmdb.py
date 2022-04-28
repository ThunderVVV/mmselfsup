# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource
import lmdb
import six
import PIL


@DATASOURCES.register_module
class Lmdb(BaseDataSource):
    """The implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.
    """

    def load_annotations(self):
        assert self.ann_file is not None
        

        data_infos = []
        env = lmdb.open(
            self.ann_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            print("cannot open lmdb from %s" % (self.ann_file))
            sys.exit(0)

        with env.begin(write=False) as txn:
            self.nSamples = int(txn.get("num-samples".encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = "label-%09d".encode() % index
                label = txn.get(label_key).decode("utf-8")

                # length filtering
                length_of_label = len(label)
                if length_of_label > 25:
                    continue

                self.filtered_index_list.append(index)
                data_infos.append(index)

            self.nSamples = len(self.filtered_index_list)
        return data_infos
    
    def get_img(self, index):
        env = lmdb.open(
            self.ann_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            print("cannot open lmdb from %s" % (self.ann_file))
            sys.exit(0)
            
        assert index <= len(self), "index range error"
        index = self.data_infos[index]

        with env.begin(write=False) as txn:
            # label_key = "label-%09d".encode() % index
            # label = txn.get(label_key).decode("utf-8")
            img_key = "image-%09d".encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert("RGB")

            except IOError:
                print(f"Corrupted image for {index}")
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new("RGB", (self.opt.imgW, self.opt.imgH))
        

        return img
