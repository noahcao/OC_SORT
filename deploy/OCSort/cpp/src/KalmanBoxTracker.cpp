#include <utility>
#include "../include/KalmanBoxTracker.hpp"
namespace ocsort {
    int KalmanBoxTracker::count = 0;
    KalmanBoxTracker::KalmanBoxTracker(Eigen::VectorXf bbox_, int cls_, int delta_t_) {
        bbox = std::move(bbox_);
        delta_t = delta_t_;
        kf = new KalmanFilterNew(7, 4);
        kf->F << 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1;
        kf->H << 1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
        kf->R.block(2, 2, 2, 2) *= 10.0;
        kf->P.block(4, 4, 3, 3) *= 1000.0;
        kf->P *= 10.0;
        kf->Q.bottomRightCorner(1, 1)(0, 0) *= 0.01;
        kf->Q.block(4, 4, 3, 3) *= 0.01;
        kf->x.head<4>() = convert_bbox_to_z(bbox);
        time_since_update = 0;
        id = KalmanBoxTracker::count;
        KalmanBoxTracker::count += 1;
        history.clear();
        hits = 0;      
        hit_streak = 0; 
        age = 0;        
        conf = bbox(4); 
        cls = cls_;
        last_observation.fill(-1);  
        observations.clear();        
        history_observations.clear();
        velocity.fill(0);            
    }

    void KalmanBoxTracker::update(Eigen::Matrix<float, 5, 1>* bbox_, int cls_) {
        if (bbox_ != nullptr) {
            conf = (*bbox_)[4];
            cls = cls_;
            if (int(last_observation.sum()) >= 0) {
                Eigen::VectorXf previous_box_tmp;
                for (int i = 0; i < delta_t; ++i) {
                    int dt = delta_t - i;
                    if (observations.count(age - dt) > 0) {
                        previous_box_tmp = observations[age - dt];
                        break;
                    }
                }
                if (0 == previous_box_tmp.size()) {     
                    previous_box_tmp = last_observation;
                }
                velocity = speed_direction(previous_box_tmp, *bbox_);
            }
            last_observation = *bbox_; 
            observations[age] = *bbox_;
            history_observations.push_back(*bbox_);
            time_since_update = 0;
            history.clear();
            hits += 1;
            hit_streak += 1;
            Eigen::VectorXf tmp = convert_bbox_to_z(*bbox_);
            kf->update(&tmp);
        }
        else {
            kf->update(nullptr);
        }
    }

    Eigen::RowVectorXf KalmanBoxTracker::predict() {
        if (kf->x[6] + kf->x[2] <= 0) kf->x[6] *= 0.0;
        kf->predict();
        age += 1;
        if (time_since_update > 0) hit_streak = 0;
        time_since_update += 1;
        history.push_back(convert_x_to_bbox(kf->x));
        return convert_x_to_bbox(kf->x);
    }
    Eigen::VectorXf KalmanBoxTracker::get_state() {
        return convert_x_to_bbox(kf->x);
    }
}// namespace ocsort