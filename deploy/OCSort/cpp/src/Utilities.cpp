#include "../include/Utilities.hpp"
namespace ocsort {
    Eigen::VectorXf convert_bbox_to_z(Eigen::VectorXf bbox) {
        double w = bbox[2] - bbox[0];
        double h = bbox[3] - bbox[1];
        double x = bbox[0] + w / 2.0;
        double y = bbox[1] + h / 2.0;
        double s = w * h;
        double r = w / (h + 1e-6);
        Eigen::MatrixXf z(4, 1);
        z << x, y, s, r;
        return z;
    }
    Eigen::VectorXf speed_direction(Eigen::VectorXf bbox1, Eigen::VectorXf bbox2) {
        double cx1 = (bbox1[0] + bbox1[2]) / 2.0;
        double cy1 = (bbox1[1] + bbox1[3]) / 2.0;
        double cx2 = (bbox2[0] + bbox2[2]) / 2.0;
        double cy2 = (bbox2[1] + bbox2[3]) / 2.0;
        Eigen::VectorXf speed(2, 1);
        speed << cy2 - cy1, cx2 - cx1;
        double norm = sqrt(pow(cy2 - cy1, 2) + pow(cx2 - cx1, 2)) + 1e-6;
        return speed / norm;
    }
    Eigen::VectorXf convert_x_to_bbox(Eigen::VectorXf x) {
        float w = std::sqrt(x(2) * x(3));
        float h = x(2) / w;
        Eigen::VectorXf bbox = Eigen::VectorXf::Ones(4, 1);
        bbox << x(0) - w / 2, x(1) - h / 2, x(0) + w / 2, x(1) + h / 2;
        return bbox;
    }
    Eigen::VectorXf k_previous_obs(std::unordered_map<int, Eigen::VectorXf> observations_, int cur_age, int k) {
        if (observations_.size() == 0) return Eigen::VectorXf::Constant(5, -1.0);
        for (int i = 0; i < k; i++) {
            int dt = k - i;
            if (observations_.count(cur_age - dt) > 0) return observations_.at(cur_age - dt);
        }
        auto iter = std::max_element(observations_.begin(), observations_.end(), [](const std::pair<int, Eigen::VectorXf>& p1, const std::pair<int, Eigen::VectorXf>& p2) { return p1.first < p2.first; });
        int max_age = iter->first;
        return observations_[max_age];
    }
}// namespace ocsort