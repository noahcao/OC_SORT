from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np 
import os
import glob
from sklearn import preprocessing
import scipy 
import sys 
import scipy.spatial


def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)


def write_results_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            frame_id = int(frame_data[0])
            track_id = int(frame_data[1])
            x1, y1, w, h = frame_data[2:6]
            score = frame_data[6]
            line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
            f.write(line)


def median_trick(X):
    """
        median trick for computing the bandwith for kernel regression.
    """
    N = len(X)
    perm = np.random.choice(N, N, replace=False)
    dsample = X[perm]
    pd = scipy.spatial.distance.pdist(dsample)
    sigma = np.median(pd)
    return sigma


def gp_interpolation(txt_path, save_path, reference_dir, n_min=30, n_dti=20):
    seq_txts = sorted(glob.glob(os.path.join(txt_path, '*.txt')))
    for seq_txt in seq_txts:
        seq_name = seq_txt.split('/')[-1]
        ref_seq_data = np.loadtxt(os.path.join(reference_dir,
                "{}".format(seq_name)), delimiter=",")
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=',')
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)

        track_count = 0
        for track_id in range(min_id, max_id + 1):
            track_count += 1
            print("{} {}/{}".format(seq_name, track_count, max_id-min_id))
            index = (seq_data[:, 1] == track_id)
            to_fill_tracklet = seq_data[index]
            ref_index = (ref_seq_data[:, 1] == track_id)
            tracklet = ref_seq_data[ref_index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            boxes = tracklet[:, 2:6].reshape((-1, 4))
            center_x = boxes[:, 0] + 0.5 * boxes[:, 2]
            center_y = boxes[:, 1] + 0.5 * boxes[:, 3]
            center_x = center_x.reshape((-1, 1))
            center_y = center_y.reshape((-1, 1))
            time_steps = tracklet[:, 0].reshape((-1, 1))

            n_frame = tracklet.shape[0]
            l = n_frame if n_frame < 500 else 500

            bandwidth = median_trick(boxes)


            """
                change the following to use your own kernel for GPR
            """
            l = 1000.0 / n_frame
            kernel = 20 * RBF(l)
            scaler_boxes = preprocessing.StandardScaler().fit(boxes)

            scaler_x = preprocessing.StandardScaler().fit(center_x)
            scaler_y = preprocessing.StandardScaler().fit(center_y)

            x_scaled = scaler_x.transform(center_x)
            y_scaled = scaler_y.transform(center_y)

            gp_x = GaussianProcessRegressor(kernel, n_restarts_optimizer=2)
            gp_y = GaussianProcessRegressor(kernel, n_restarts_optimizer=2)
            gp_x.fit(time_steps, x_scaled)
            gp_y.fit(time_steps, y_scaled)
            
            if n_frame > n_min:
                frames = tracklet[:, 0]
                to_fill_frames = to_fill_tracklet[:,0]
                frames_dti = {}
                for frame in frames:
                    if frame not in to_fill_frames:
                        """
                            Smooth the steps which are made up by the linear interpolation
                        """
                        curr_frame = frame
                        width, height = tracklet[np.where(tracklet[:,0]==curr_frame)][0][4:6]
                        curr_frame = np.array([curr_frame]).reshape((-1,1))

                        curr_x = gp_x.predict(curr_frame)
                        curr_y = gp_y.predict(curr_frame)
                        curr_x = scaler_x.inverse_transform(curr_x)
                        curr_y = scaler_y.inverse_transform(curr_y)
                        curr_bbox = np.array([[curr_x - 0.5 * width, curr_y - 0.5 * height,
                                width, height]]).reshape((4,))
                        tracklet = tracklet[np.where(tracklet[:,0]!=curr_frame)[1]]
                        frames_dti[int(curr_frame.item())] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        write_results_score(save_seq_txt, seq_results)

if __name__ == "__main__":
    """
        Input:
            * txt_path: the raw tracking output path 
            * save_path: path to saved the interpolated result files
            * li_path: the path to results after linear interpolation
    """
    txt_path, li_path, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
    mkdir_if_missing(save_path)
    gp_interpolation(txt_path, save_path, li_path, n_min=30, n_dti=20)