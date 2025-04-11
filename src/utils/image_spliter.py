import torch

class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        bs, chn, height, width= im.shape
        self.chn = 1
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.height = height
        self.width = width
        self.im_res = torch.zeros([bs, self.chn, height * self.sf, width * self.sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros_like(self.im_res)
        self.weight = self._gaussian_weights(pch_size, pch_size, bs, im.device)



        # precompute the weight-tensor（no effect)
        # print(f"self.im_res shape: {self.im_res.shape}")
        # print(f"self.weight shape: {self.weight.shape}")
        # for h_start in self.height_starts_list:
        #     for w_start in self.width_starts_list:
        #         h_end = h_start + self.pch_size
        #         w_end = w_start + self.pch_size
        #
        #         h_start_sf = h_start * self.sf
        #         h_end_sf = h_end * self.sf
        #         w_start_sf = w_start * self.sf
        #         w_end_sf = w_end * self.sf
        #
        #         self.pixel_count[:, :, h_start_sf:h_end_sf, w_start_sf:w_end_sf] += self.weight

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for i in range(len(starts)):
                if starts[i] + self.pch_size > length:
                    starts[i] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        self.num_pchs = 0
        return self

    def __next__(self):
        #tried to include boundry patch, ensure boundry is covered(not useful)
        if self.num_pchs < self.length:
            total_rows = len(self.height_starts_list)
            total_cols = len(self.width_starts_list)
            row_idx = self.num_pchs // total_cols
            col_idx = self.num_pchs % total_cols

            h_start = self.height_starts_list[row_idx]
            w_start = self.width_starts_list[col_idx]

            # Check if we're at the last row or column and adjust accordingly
            if row_idx == total_rows - 1:
                h_start = self.height - self.pch_size
                h_end = self.height
            else:
                h_end = h_start + self.pch_size

            if col_idx == total_cols - 1:
                w_start = self.width - self.pch_size
                w_end = self.width
            else:
                w_end = w_start + self.pch_size

            # Ensure indices are within bounds
            h_start = max(0, h_start)
            h_end = min(self.height, h_end)
            w_start = max(0, w_start)
            w_end = min(self.width, w_end)

            pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]

            h_start_sf = h_start * self.sf
            h_end_sf = h_end * self.sf
            w_start_sf = w_start * self.sf
            w_end_sf = w_end * self.sf

            self.w_start, self.w_end = w_start_sf, w_end_sf
            self.h_start, self.h_end = h_start_sf, h_end_sf

            self.num_pchs += 1
        else:
            raise StopIteration()

        return pch, (h_start_sf, h_end_sf, w_start_sf, w_end_sf)

    def _gaussian_weights(self, tile_width, tile_height, nbatches, device):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.003
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        weights_tensor = torch.tensor(weights, device=device).unsqueeze(0).unsqueeze(0)
        weights_tensor = weights_tensor.repeat(nbatches, self.chn, 1, 1)
        return weights_tensor

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def update_gaussian(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        #self.weight /= self.weight.sum()
        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res * self.weight
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += self.weight

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

    def reset_accumulators(self):
        self.im_res.zero_()
        self.pixel_count.zero_()

