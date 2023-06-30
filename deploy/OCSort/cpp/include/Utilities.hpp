#ifndef OC_SORT_CPP_UTILITIES_HPP
#define OC_SORT_CPP_UTILITIES_HPP
#include "Eigen/Dense"
namespace ocsort {
    /**
     * Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
     * @param bbox
     * @return z
     */
    Eigen::VectorXf convert_bbox_to_z(Eigen::VectorXf bbox);
    Eigen::VectorXf speed_direction(Eigen::VectorXf bbox1, Eigen::VectorXf bbox2);
    Eigen::VectorXf convert_x_to_bbox(Eigen::VectorXf x);
    Eigen::VectorXf k_previous_obs(std::unordered_map<int, Eigen::VectorXf> observations_, int cur_age, int k);
}// namespace ocsort
#endif//OC_SORT_CPP_UTILITIES_HPP