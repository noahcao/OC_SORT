#ifndef OC_SORT_CPP_ASSOCIATION_HPP
#define OC_SORT_CPP_ASSOCIATION_HPP

#include "Eigen/Dense"
#include <algorithm>
#include "vector"
#include "lapjv.hpp"
#define pi 3.1415926

namespace ocsort {
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> speed_direction_batch(const Eigen::MatrixXf& dets,
        const Eigen::MatrixXf& tracks);
    Eigen::MatrixXf iou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2);
    Eigen::MatrixXf giou_batch(const Eigen::MatrixXf& bboxes1, const Eigen::MatrixXf& bboxes2);
    std::tuple<std::vector<Eigen::Matrix<int, 1, 2>>, std::vector<int>, std::vector<int>> associate(Eigen::MatrixXf detections, Eigen::MatrixXf trackers, float iou_threshold, Eigen::MatrixXf velocities, Eigen::MatrixXf previous_obs_, float vdc_weight);
}// namespace ocsort

#endif//OC_SORT_CPP_ASSOCIATION_HPP