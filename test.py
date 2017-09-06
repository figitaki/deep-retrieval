import sys
import numpy as np
import caffe
import argparse
import cv2
from tqdm import tqdm
import os
from collections import OrderedDict
import subprocess


class ImageHelper:
    def __init__(self, S, L, means):
        self.S = S
        self.L = L
        self.means = means

    def prepare_image_and_grid_regions_for_network(self, fname, roi=None):
        # Extract image, resize at desired size, and extract roi region if
        # available. Then compute the rmac grid in the net format: ID X Y W H
        I, im_resized = self.load_and_prepare_image(fname, roi)
        if self.L == 0:
            # Encode query in mac format instead of rmac, so only one region
            # Regions are in ID X Y W H format
            R = np.zeros((1, 5), dtype=np.float32)
            R[0, 3] = im_resized.shape[1] - 1
            R[0, 4] = im_resized.shape[0] - 1
        else:
            # Get the region coordinates and feed them to the network.
            all_regions = []
            all_regions.append(self.get_rmac_region_coordinates(im_resized.shape[0], im_resized.shape[1], self.L))
            R = self.pack_regions_for_network(all_regions)
        return I, R

    def get_rmac_features(self, I, R, net):
        net.blobs['data'].reshape(I.shape[0], 3, int(I.shape[2]), int(I.shape[3]))
        net.blobs['data'].data[:] = I
        net.blobs['rois'].reshape(R.shape[0], R.shape[1])
        net.blobs['rois'].data[:] = R.astype(np.float32)
        net.forward(end='rmac/normalized')
        return np.squeeze(net.blobs['rmac/normalized'].data)

    def load_and_prepare_image(self, fname, roi=None):
        # Read image, get aspect ratio, and resize such as the largest side equals S
        im = cv2.imread(fname)
        im_size_hw = np.array(im.shape[0:2])
        ratio = float(self.S)/np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))
        im_resized = cv2.resize(im, (new_size[1], new_size[0]))
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[roi[1]:roi[3], roi[0]:roi[2], :]
        # Transpose for network and subtract mean
        I = im_resized.transpose(2, 0, 1) - self.means
        return I, im_resized

    def pack_regions_for_network(self, all_regions):
        n_regs = np.sum([len(e) for e in all_regions])
        R = np.zeros((n_regs, 5), dtype=np.float32)
        cnt = 0
        # There should be a check of overflow...
        for i, r in enumerate(all_regions):
            try:
                R[cnt:cnt + r.shape[0], 0] = i
                R[cnt:cnt + r.shape[0], 1:] = r
                cnt += r.shape[0]
            except:
                continue
        assert cnt == n_regs
        R = R[:n_regs]
        # regs where in xywh format. R is in xyxy format, where the last coordinate is included. Therefore...
        R[:n_regs, 3] = R[:n_regs, 1] + R[:n_regs, 3] - 1
        R[:n_regs, 4] = R[:n_regs, 2] + R[:n_regs, 4] - 1
        return R

    def get_rmac_region_coordinates(self, H, W, L):
        # Almost verbatim from Tolias et al Matlab implementation.
        # Could be heavily pythonized, but really not worth it...
        # Desired overlap of neighboring regions
        ovr = 0.4
        # Possible regions for the long dimension
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)

        b = (np.maximum(H, W) - w) / (steps - 1)
        # steps(idx) regions for long dimension. The +1 comes from Matlab
        # 1-indexing...
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

        # Region overplus per dimension
        Wd = 0
        Hd = 0
        if H < W:
            Wd = idx
        elif H > W:
            Hd = idx

        regions_xywh = []
        for l in range(1, L+1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            # Center coordinates
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            # Center coordinates
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

            for i_ in cenH:
                for j_ in cenW:
                    regions_xywh.append([j_, i_, wl, wl])

        # Round the regions. Careful with the borders!
        for i in range(len(regions_xywh)):
            for j in range(4):
                regions_xywh[i][j] = int(round(regions_xywh[i][j]))
            if regions_xywh[i][0] + regions_xywh[i][2] > W:
                regions_xywh[i][0] -= ((regions_xywh[i][0] + regions_xywh[i][2]) - W)
            if regions_xywh[i][1] + regions_xywh[i][3] > H:
                regions_xywh[i][1] -= ((regions_xywh[i][1] + regions_xywh[i][3]) - H)
        return np.array(regions_xywh).astype(np.float32)


class Dataset:
    def __init__(self, path, eval_binary_path):
        self.path = path
        self.eval_binary_path = eval_binary_path
        # Some images from the Paris dataset are corrupted. Standard practice is
        # to ignore them
        self.blacklisted = set(["paris_louvre_000136",
                            "paris_louvre_000146",
                            "paris_moulinrouge_000422",
                            "paris_museedorsay_001059",
                            "paris_notredame_000188",
                            "paris_pantheon_000284",
                            "paris_pantheon_000960",
                            "paris_pantheon_000974",
                            "paris_pompidou_000195",
                            "paris_pompidou_000196",
                            "paris_pompidou_000201",
                            "paris_pompidou_000467",
                            "paris_pompidou_000640",
                            "paris_sacrecoeur_000299",
                            "paris_sacrecoeur_000330",
                            "paris_sacrecoeur_000353",
                            "paris_triomphe_000662",
                            "paris_triomphe_000833",
                            "paris_triomphe_000863",
                            "paris_triomphe_000867"])
        self.load()

    def load(self):
        # Load the dataset GT
        self.lab_root = '{0}/lab/'.format(self.path)
        self.img_root = '{0}/jpg/'.format(self.path)
        lab_filenames = np.sort(os.listdir(self.lab_root))
        # Get the filenames without the extension
        self.img_filenames = [e[:-4] for e in np.sort(os.listdir(self.img_root)) if e[:-4] not in self.blacklisted]

        # Parse the label files. Some challenges as filenames do not correspond
        # exactly to query names. Go through all the labels to:
        # i) map names to filenames and vice versa
        # ii) get the relevant regions of interest of the queries,
        # iii) get the indexes of the dataset images that are queries
        # iv) get the relevants / non-relevants list
        self.relevants = {}
        self.junk = {}
        self.non_relevants = {}

        self.filename_to_name = {}
        self.name_to_filename = OrderedDict()
        self.q_roi = {}
        for e in lab_filenames:
            if e.endswith('_query.txt'):
                q_name = e[:-len('_query.txt')]
                q_data = file("{0}/{1}".format(self.lab_root, e)).readline().split(" ")
                q_filename = q_data[0][5:] if q_data[0].startswith('oxc1_') else q_data[0]
                self.filename_to_name[q_filename] = q_name
                self.name_to_filename[q_name] = q_filename
                good = set([e.strip() for e in file("{0}/{1}_ok.txt".format(self.lab_root, q_name))])
                good = good.union(set([e.strip() for e in file("{0}/{1}_good.txt".format(self.lab_root, q_name))]))
                junk = set([e.strip() for e in file("{0}/{1}_junk.txt".format(self.lab_root, q_name))])
                good_plus_junk = good.union(junk)
                self.relevants[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in good]
                self.junk[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] in junk]
                self.non_relevants[q_name] = [i for i in range(len(self.img_filenames)) if self.img_filenames[i] not in good_plus_junk]
                self.q_roi[q_name] = np.array(map(float, q_data[1:]), dtype=np.float32)

        self.q_names = self.name_to_filename.keys()
        self.q_index = np.array([self.img_filenames.index(self.name_to_filename[qn]) for qn in self.q_names])
        self.N_images = len(self.img_filenames)
        self.N_queries = len(self.q_index)

    def score(self, sim, temp_dir, eval_bin):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        maps = [self.score_rnk_partial(i, idx[i], temp_dir, eval_bin) for i in range(len(self.q_names))]
        for i in range(len(self.q_names)):
            print "{0}: {1:.2f}".format(self.q_names[i], 100 * maps[i])
        print 20 * "-"
        print "Mean: {0:.2f}".format(100 * np.mean(maps))

    def score_rnk_partial(self, i, idx, temp_dir, eval_bin):
        rnk = np.array(self.img_filenames)[idx]
        with open("{0}/{1}.rnk".format(temp_dir, self.q_names[i]), 'w') as f:
            f.write("\n".join(rnk)+"\n")
        cmd = "{0} {1}{2} {3}/{4}.rnk".format(eval_bin, self.lab_root, self.q_names[i], temp_dir, self.q_names[i])
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        map_ = float(p.stdout.readlines()[0])
        p.wait()
        return map_

    def get_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[i]))

    def get_query_filename(self, i):
        return os.path.normpath("{0}/{1}.jpg".format(self.img_root, self.img_filenames[self.q_index[i]]))

    def get_query_roi(self, i):
        return self.q_roi[self.q_names[i]]


def extract_features(dataset, image_helper, net, args):
    Ss = [args.S, ] if not args.multires else [args.S - 250, args.S, args.S + 250]
    # First part, queries
    for S in Ss:
        # Set the scale of the image helper
        image_helper.S = S
        out_queries_fname = "{0}/{1}_S{2}_L{3}_queries.npy".format(args.temp_dir, args.dataset_name, S, args.L)
        if not os.path.exists(out_queries_fname):
            dim_features = net.blobs['rmac/normalized'].data.shape[1]
            N_queries = dataset.N_queries
            features_queries = np.zeros((N_queries, dim_features), dtype=np.float32)
            for i in tqdm(range(N_queries), file=sys.stdout, leave=False, dynamic_ncols=True):
                # Load image, process image, get image regions, feed into the network, get descriptor, and store
                I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_query_filename(i), roi=dataset.get_query_roi(i))
                features_queries[i] = image_helper.get_rmac_features(I, R, net)
            np.save(out_queries_fname, features_queries)
    features_queries = np.dstack([np.load("{0}/{1}_S{2}_L{3}_queries.npy".format(args.temp_dir, args.dataset_name, S, args.L)) for S in Ss]).sum(axis=2)
    features_queries /= np.sqrt((features_queries * features_queries).sum(axis=1))[:, None]

    # Second part, dataset
    for S in Ss:
        image_helper.S = S
        out_dataset_fname = "{0}/{1}_S{2}_L{3}_dataset.npy".format(args.temp_dir, args.dataset_name, S, args.L)
        if not os.path.exists(out_dataset_fname):
            dim_features = net.blobs['rmac/normalized'].data.shape[1]
            N_dataset = dataset.N_images
            features_dataset = np.zeros((N_dataset, dim_features), dtype=np.float32)
            for i in tqdm(range(N_dataset), file=sys.stdout, leave=False, dynamic_ncols=True):
                # Load image, process image, get image regions, feed into the network, get descriptor, and store
                I, R = image_helper.prepare_image_and_grid_regions_for_network(dataset.get_filename(i), roi=None)
                features_dataset[i] = image_helper.get_rmac_features(I, R, net)
            np.save(out_dataset_fname, features_dataset)
    features_dataset = np.dstack([np.load("{0}/{1}_S{2}_L{3}_dataset.npy".format(args.temp_dir, args.dataset_name, S, args.L)) for S in Ss]).sum(axis=2)
    features_dataset /= np.sqrt((features_dataset * features_dataset).sum(axis=1))[:, None]
    # Restore the original scale
    image_helper.S = args.S
    return features_queries, features_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Oxford / Paris')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use (e.g. 0)')
    parser.add_argument('--S', type=int, required=True, help='Resize larger side of image to S pixels (e.g. 800)')
    parser.add_argument('--L', type=int, required=True, help='Use L spatial levels (e.g. 2)')
    parser.add_argument('--proto', type=str, required=True, help='Path to the prototxt file')
    parser.add_argument('--weights', type=str, required=True, help='Path to the caffemodel file')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the Oxford / Paris directory')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--eval_binary', type=str, required=True, help='Path to the compute_ap binary to evaluate Oxford / Paris')
    parser.add_argument('--temp_dir', type=str, required=True, help='Path to a temporary directory to store features and scores')
    parser.add_argument('--multires', dest='multires', action='store_true', help='Enable multiresolution features')
    parser.add_argument('--aqe', type=int, required=False, help='Average query expansion with k neighbors')
    parser.add_argument('--dbe', type=int, required=False, help='Database expansion with k neighbors')
    parser.set_defaults(multires=False)
    args = parser.parse_args()

    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    # Load and reshape the means to subtract to the inputs
    args.means = np.array([103.93900299,  116.77899933,  123.68000031], dtype=np.float32)[None, :, None, None]

    # Configure caffe and load the network
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(args.proto, args.weights, caffe.TEST)

    # Load the dataset and the image helper
    dataset = Dataset(args.dataset, args.eval_binary)
    image_helper = ImageHelper(args.S, args.L, args.means)

    # Extract features
    features_queries, features_dataset = extract_features(dataset, image_helper, net, args)

    # Database side expansion?
    if args.dbe is not None and args.dbe > 0:
        # Extend the database features
        # With larger datasets this has to be done in a batched way.
        # and using smarter ways than sorting to take the top k results.
        # For 5k images, not really a problem to do it by brute force
        X = features_dataset.dot(features_dataset.T)
        idx = np.argsort(X, axis=1)[:, ::-1]
        weights = np.hstack(([1], (args.dbe - np.arange(0, args.dbe)) / float(args.dbe)))
        weights_sum = weights.sum()
        features_dataset = np.vstack([np.dot(weights, features_dataset[idx[i, :args.dbe + 1], :]) / weights_sum for i in range(len(features_dataset))])

    # Compute similarity
    sim = features_queries.dot(features_dataset.T)
    # Average query expansion?
    if args.aqe is not None and args.aqe > 0:
        # Sort the results to get the nearest neighbors, compute average
        # representations, and query again.
        # No need to L2-normalize as we are on the query side, so it doesn't
        # affect the ranking
        idx = np.argsort(sim, axis=1)[:, ::-1]
        features_queries = np.vstack([np.vstack((features_queries[i], features_dataset[idx[i, :args.aqe]])).mean(axis=0) for i in range(len(features_queries))])
        #for i in range(features_queries.shape[0]):
        #    features_queries[i] = np.vstack((features_queries[i], features_dataset[idx[i, :args.aqe]])).mean(axis=0)
        sim = features_queries.dot(features_dataset.T)

    # Score
    dataset.score(sim, args.temp_dir, args.eval_binary)
