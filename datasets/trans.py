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
        self.total_frames = list(cleaned.values[:, 1])
        self.label_array = list(cleaned.values[:, 2])

        if cleaned.shape[1] == 4:
            self.conf_array = list(cleaned.values[:, -1])
        else:
            self.conf_array = None

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