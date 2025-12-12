import os
import io
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)
from .volume_transforms import ClipToTensor

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

class SSVideoClsDatasetFromRawFrame(Dataset):
    """Load your own video classification dataset using Decord decoder."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=8,
                 crop_size=224, short_side_size=256, new_height=256, new_width=340,
                 keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10,
                 test_num_crop=3, filename_tmpl='img_{:05}.jpg', args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.args = args
        self.aug = False
        self.rand_erase = False

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
        self.dataset_samples = list(cleaned.values[:, 0].astype('str'))
        self.total_frames = list(cleaned.values[:, 1])
        self.label_array = list(cleaned.values[:, 2])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_total_frames = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_total_frames.append(self.total_frames[idx])
                        self.test_label_array.append(self.label_array[idx])

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_video_decord(sample, total_frame, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    total_frame = self.total_frames[index]
                    buffer = self.load_video_decord(sample, total_frame, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_video_decord(sample, total_frame)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video_decord(sample, total_frame)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            total_frame = self.test_total_frames[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video_decord(sample, total_frame)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                total_frame = self.test_total_frames[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_video_decord(sample, total_frame)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.test_num_crop - 1)
            temporal_start = chunk_nb
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::self.test_num_segment, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::self.test_num_segment, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        """Data augmentation part remains unchanged"""
        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_video_decord(self, sample, num_frames, sample_rate_scale=1):
        """Load video content using Decord decoder"""
        fname = sample
        fname = os.path.join(self.prefix, fname)
        
        # If it is a frame sequence path, convert to video file path
        # Assume video file has the same name as frame sequence folder, but with .mp4 extension
        if os.path.isdir(fname) or '/' in fname:
            # Extract directory name as video filename
            dir_name = fname.rstrip('/')
            video_fname = dir_name + '.mp4'
        else:
            video_fname = fname

        try:
            if self.keep_aspect_ratio:
                if "s3://" in video_fname:
                    video_bytes = self.client.get(video_fname)
                    vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
                else:
                    vr = VideoReader(video_fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in video_fname:
                    video_bytes = self.client.get(video_fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(video_fname, 
                                    width=self.new_width, 
                                    height=self.new_height,
                                    num_threads=1, 
                                    ctx=cpu(0))
        except Exception as e:
            print(f"Video {video_fname} cannot be loaded by decord: {e}")
            return []

        if self.mode == 'test':
            # Test mode: dense sampling, keep the same sampling strategy as original RawFrame
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index)))
            
            # Limit index not to exceed total video frames
            all_index = [min(idx, len(vr)-1) for idx in all_index]
            
        else:
            # Train and validation mode: keep the same sampling strategy as original RawFrame
            average_duration = num_frames // self.num_segment
            all_index = []
            
            if average_duration > 0:
                if self.mode == 'validation':
                    all_index = list(
                        np.multiply(list(range(self.num_segment)), average_duration) +
                        np.ones(self.num_segment, dtype=int) * (average_duration // 2))
                else:
                    all_index = list(
                        np.multiply(list(range(self.num_segment)), average_duration) +
                        np.random.randint(average_duration, size=self.num_segment))
            elif num_frames > self.num_segment:
                if self.mode == 'validation':
                    all_index = list(range(self.num_segment))
                else:
                    all_index = list(
                        np.sort(np.random.randint(num_frames, size=self.num_segment)))
            else:
                all_index = [0] * (self.num_segment - num_frames) + list(range(num_frames))
            
            # Limit index not to exceed total video frames
            all_index = [min(idx, len(vr)-1) for idx in all_index]

        try:
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except Exception as e:
            print(f"Error getting batch for video {video_fname}: {e}")
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


class SSRawFrameClsDatasetCombine(Dataset):
    """Load your own raw frame classification dataset."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=8,
                 crop_size=224, short_side_size=256, new_height=256, new_width=340,
                 keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10,
                 test_num_crop=3, filename_tmpl='img_{:05}.jpg', args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.args = args
        self.aug = False
        self.rand_erase = False

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
        self.dataset_samples = list(cleaned.values[:, 0].astype('str'))
        self.total_frames = list(cleaned.values[:, 1])
        self.label_array = list(cleaned.values[:, 2])
        # self.conf_array = list(cleaned.values[:, -1])
        # Check if conf_array exists
        if cleaned.shape[1] == 4:
            self.conf_array = list(cleaned.values[:, -1])
        else:
            self.conf_array = None

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = Compose([
                Resize(self.short_side_size,
                                        interpolation='bilinear'),
                CenterCrop(size=(self.crop_size,
                                                  self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size),
                                        interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_total_frames = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_total_frames.append(self.total_frames[idx])
                        self.test_label_array.append(self.label_array[idx])

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(sample,
                                     total_frame,
                                     sample_rate_scale=scale_t)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    total_frame = self.total_frames[index]
                    buffer = self.load_frame(sample,
                                             total_frame,
                                             sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                depth_list = []
                label_list = []
                index_list = []
                conf_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                    if self.conf_array is not None:
                        conf_list.append(self.conf_array[index])
                    depth_frames = self.depth_frame(buffer)
                    if isinstance(depth_frames, torch.Tensor) and depth_frames.numel() > 0:
                        depth_list.append(depth_frames.clone())
                    else:
                        depth_list.append(depth_frames)
                return frame_list, label_list, conf_list, depth_list
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(sample, total_frame)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".
                        format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_frame(sample, total_frame)
            depth_frames = self.depth_frame(buffer)
            buffer = self.data_transform(buffer)
            # return buffer, self.label_array[index], sample.split(
            #     "/")[-1].split(".")[0]
            return buffer, self.label_array[index], depth_frames

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            total_frame = self.test_total_frames[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_frame(sample, total_frame)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                total_frame = self.test_total_frames[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_frame(sample, total_frame)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            temporal_start = chunk_nb
            if self.test_num_crop == 1:
                # Perform center crop when test_num_crop is 1
                h, w = buffer.shape[1], buffer.shape[2]
                if h >= w:
                    offset_h = int((h - self.short_side_size) / 2)
                    offset_w = 0
                    buffer = buffer[temporal_start::self.test_num_segment, \
                           offset_h:offset_h + self.short_side_size, :, :]
                else:
                    offset_h = 0
                    offset_w = int((w - self.short_side_size) / 2)
                    buffer = buffer[temporal_start::self.test_num_segment, \
                           :, offset_w:offset_w + self.short_side_size, :]
            else:
                # Original multi-crop logic
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                  / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::self.test_num_segment, \
                           spatial_start:spatial_start + self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start::self.test_num_segment, \
                           :, spatial_start:spatial_start + self.short_side_size, :]

            depth_frames = self.depth_frame(buffer)
            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb, depth_frames
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def depth_frame(self, buffer):
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C
        buffer = tensor_normalize(buffer, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        buffer = buffer.permute(0, 3, 1, 2)# T C H W
        interpolation_mode = "bilinear"
        buffer = nn.functional.interpolate(
            buffer,
            size=(1536, 1536),
            mode=interpolation_mode,
            align_corners=False,
        )
        buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W.
        return buffer

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_frame(self, sample, num_frames, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample
        fname = os.path.join(self.prefix, fname)

        if self.mode == 'test':
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index)))
            print("all_index: ", all_index)
            imgs = []
            for idx in all_index:
                # frame_fname = os.path.join(fname, self.filename_tmpl.format(idx + 1))
                frame_fname = os.path.join(fname, self.filename_tmpl.format(idx))
                if "s3://" in fname:
                    img_bytes = self.client.get(frame_fname)
                else:
                    with open(frame_fname, 'rb') as f:
                        img_bytes = f.read()
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                imgs.append(img)
            buffer = np.array(imgs)
            return buffer

        # handle temporal segments
        average_duration = num_frames // self.num_segment
        all_index = []
        if average_duration > 0:
            if self.mode == 'validation':
                all_index = list(
                    np.multiply(list(range(self.num_segment)),
                                average_duration) +
                    np.ones(self.num_segment, dtype=int) *
                    (average_duration // 2))
            else:
                all_index = list(
                    np.multiply(list(range(self.num_segment)),
                                average_duration) +
                    np.random.randint(average_duration, size=self.num_segment))
        elif num_frames > self.num_segment:
            if self.mode == 'validation':
                all_index = list(range(self.num_segment))
            else:
                all_index = list(
                    np.sort(
                        np.random.randint(num_frames, size=self.num_segment)))
        else:
            all_index = [0] * (self.num_segment - num_frames) + list(
                range(num_frames))
        all_index = list(np.array(all_index))
        # all_index = [0, 1, 2, 3, 4, 5, 6, 7]
        imgs = []
        for idx in all_index:
            # frame_fname = os.path.join(fname, self.filename_tmpl.format(idx + 1))
            frame_fname = os.path.join(fname, self.filename_tmpl.format(idx))
            if "s3://" in fname:
                img_bytes = self.client.get(frame_fname)
            else:
                with open(frame_fname, 'rb') as f:
                    img_bytes = f.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            imgs.append(img)
        buffer = np.array(imgs)
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


class SSVideoClsDatasetCombine(Dataset):
    """Video-based variant of the combined dataset that reads clips with Decord."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=8,
                 crop_size=224, short_side_size=256, new_height=256, new_width=340,
                 keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10,
                 test_num_crop=3, filename_tmpl='img_{:05}.jpg', args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.args = args
        self.aug = False
        self.rand_erase = False
        # Optionally restore depth branch, enabled by default to maintain consistency with frame version
        self.return_depth = getattr(args, 'return_depth', True) if args is not None else False

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
        self.dataset_samples = list(cleaned.values[:, 0].astype('str'))
        
        # Unified logic: no longer read frame count from CSV, use dummy value to fill
        # Actual frame count will be obtained via len(vr) in load_video_decord
        self.total_frames = [0] * len(self.dataset_samples)

        if cleaned.shape[1] == 2:
            # Format: [path, label]
            self.label_array = list(cleaned.values[:, 1])
            self.conf_array = None
        elif cleaned.shape[1] == 3:
            # Format: [path, label, conf]
            self.label_array = list(cleaned.values[:, 1])
            self.conf_array = list(cleaned.values[:, 2])
        else:
            raise ValueError(f"Unsupported CSV format with {cleaned.shape[1]} columns. Expected 2 (path, label) or 3 (path, label, conf) columns.")

        if mode == 'train':
            pass
        elif mode == 'validation':
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_total_frames = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_total_frames.append(self.total_frames[idx])
                        self.test_label_array.append(self.label_array[idx])

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_video_decord(sample, total_frame, sample_rate_scale=scale_t)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    total_frame = self.total_frames[index]
                    buffer = self.load_video_decord(sample, total_frame, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                depth_list = []
                label_list = []
                conf_list = []
                depth_frames = self._build_depth(buffer)
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    frame_list.append(new_frames)
                    label_list.append(self.label_array[index])
                    if self.conf_array is not None:
                        conf_list.append(self.conf_array[index])
                    else:
                        conf_list.append(index)
                    depth_list.append(depth_frames)
                return frame_list, label_list, conf_list, depth_list
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_video_decord(sample, total_frame)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video_decord(sample, total_frame)
            depth_frames = self._build_depth(buffer)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], depth_frames

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            total_frame = self.test_total_frames[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video_decord(sample, total_frame)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                total_frame = self.test_total_frames[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_video_decord(sample, total_frame)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            temporal_start = chunk_nb
            if self.test_num_crop == 1:
                h, w = buffer.shape[1], buffer.shape[2]
                if h >= w:
                    offset_h = int((h - self.short_side_size) / 2)
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    offset_h:offset_h + self.short_side_size, :, :]
                else:
                    offset_w = int((w - self.short_side_size) / 2)
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    :, offset_w:offset_w + self.short_side_size, :]
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                    / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    spatial_start:spatial_start + self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    :, spatial_start:spatial_start + self.short_side_size, :]

            depth_frames = self._build_depth(buffer)
            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                chunk_nb, split_nb, depth_frames
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _build_depth(self, buffer):
        if not self.return_depth:
            return torch.empty(0)

        if isinstance(buffer, torch.Tensor):
            frames = buffer.detach().cpu()
            if frames.dim() == 4:
                if frames.shape[-1] in (3, 1) and frames.shape[0] not in (3, 1):
                    pass  # already T H W C
                elif frames.shape[1] in (3, 1):
                    frames = frames.permute(0, 2, 3, 1)
                elif frames.shape[0] in (3, 1):
                    frames = frames.permute(1, 2, 3, 0)
            buffer_np = frames.numpy()
        else:
            buffer_np = buffer

        frames = []
        for frame in buffer_np:
            if isinstance(frame, torch.Tensor):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = frame
            frames.append(transforms.ToTensor()(frame_np))

        depth_tensor = torch.stack(frames)  # T C H W
        depth_tensor = depth_tensor.permute(0, 2, 3, 1)  # T H W C
        depth_tensor = tensor_normalize(depth_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        depth_tensor = depth_tensor.permute(0, 3, 1, 2)  # T C H W

        depth_tensor = nn.functional.interpolate(
            depth_tensor,
            size=(1536, 1536),
            mode="bilinear",
            align_corners=False,
        )

        depth_tensor = depth_tensor.permute(1, 0, 2, 3)  # C T H W
        return depth_tensor

    def _aug_frame(self, buffer, args):
        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        frames = [transforms.ToPILImage()(frame) for frame in buffer]
        frames = aug_transform(frames)

        frames = [transforms.ToTensor()(img) for img in frames]
        frames = torch.stack(frames)  # T C H W
        frames = frames.permute(0, 2, 3, 1)  # T H W C

        frames = tensor_normalize(frames, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        frames = frames.permute(3, 0, 1, 2)

        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        frames = spatial_sampling(
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def load_video_decord(self, sample, num_frames, sample_rate_scale=1):
        fname = os.path.join(self.prefix, sample)

        if os.path.isdir(fname):
            video_fname = fname.rstrip('/') + '.mp4'
        else:
            if "s3://" not in fname:
                if os.path.exists(fname):
                    video_fname = fname
                elif os.path.exists(fname + '.mp4'):
                    video_fname = fname + '.mp4'
                else:
                    ext = os.path.splitext(fname)[1]
                    if ext == '':
                        video_fname = fname + '.mp4'
                    else:
                        video_fname = fname
            else:
                ext = os.path.splitext(fname)[1]
                if ext == '':
                    video_fname = fname + '.mp4'
                else:
                    video_fname = fname

        try:
            if self.keep_aspect_ratio:
                if "s3://" in video_fname:
                    video_bytes = self.client.get(video_fname)
                    vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
                else:
                    vr = VideoReader(video_fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in video_fname:
                    video_bytes = self.client.get(video_fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(video_fname,
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
        except Exception as e:
            print(f"Video {video_fname} cannot be loaded by decord: {e}")
            return []

        # Override passed num_frames with actual frame count read by decord
        # This way it doesn't matter if frame count in CSV is inaccurate or 0
        num_frames = len(vr)

        if self.mode == 'test':
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index)))
            all_index = [min(idx, len(vr) - 1) for idx in all_index]
        else:
            average_duration = num_frames // self.num_segment
            all_index = []

            if average_duration > 0:
                if self.mode == 'validation':
                    all_index = list(
                        np.multiply(list(range(self.num_segment)), average_duration) +
                        np.ones(self.num_segment, dtype=int) * (average_duration // 2))
                else:
                    all_index = list(
                        np.multiply(list(range(self.num_segment)), average_duration) +
                        np.random.randint(average_duration, size=self.num_segment))
            elif num_frames > self.num_segment:
                if self.mode == 'validation':
                    all_index = list(range(self.num_segment))
                else:
                    all_index = list(
                        np.sort(np.random.randint(num_frames, size=self.num_segment)))
            else:
                all_index = [0] * (self.num_segment - num_frames) + list(range(num_frames))

            all_index = [min(idx, len(vr) - 1) for idx in all_index]

        try:
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        except Exception as e:
            print(f"Error getting batch for video {video_fname}: {e}")
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


class SSVideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train', clip_len=8,
                crop_size=224, short_side_size=256, new_height=256,
                new_width=340, keep_aspect_ratio=True, num_segment=1,
                num_crop=1, test_num_segment=10, test_num_crop=3, filename_tmpl=None, args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        
        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=self.split)
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            temporal_start = chunk_nb # 0/1
            if self.test_num_crop == 1:
                # When test_num_crop is 1, do not crop, directly select the corresponding time segment
                buffer = buffer[temporal_start::2, :, :, :]
            else:
                # Original multi-crop logic
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::2, \
                           spatial_start:spatial_start + self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start::2, \
                           :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample
        fname = os.path.join(self.prefix, fname)

        try:
            if self.keep_aspect_ratio:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                     width=self.new_width,
                                     height=self.new_height,
                                     num_threads=1,
                                     ctx=cpu(0))
                else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                    num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            tick = len(vr) / float(self.num_segment)
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                               [int(tick * x) for x in range(self.num_segment)]))
            while len(all_index) < (self.num_segment * self.test_num_segment):
                all_index.append(all_index[-1])
            all_index = np.sort(np.array(all_index))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer
        elif self.mode == 'validation':
            tick = len(vr) / float(self.num_segment)
            all_index = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        if average_duration > 0:
            all_index = list(np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                        size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index = list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index = list(np.zeros((self.num_segment,)))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
