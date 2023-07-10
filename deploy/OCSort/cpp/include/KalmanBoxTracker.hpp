#ifndef OC_SORT_CPP_KALMANBOXTRACKER_HPP
#define OC_SORT_CPP_KALMANBOXTRACKER_HPP
////////////// KalmanBoxTracker /////////////
#include "../include/kalmanfilter.hpp"
#include "../include/Utilities.hpp"
#include "iostream"
/*
This class represents the internal state of individual
tracked objects observed as bbox.
*/
namespace ocsort {

    class KalmanBoxTracker {
    public:
        /*method*/
        KalmanBoxTracker() {};
        KalmanBoxTracker(Eigen::VectorXf bbox_, int cls_, int delta_t_ = 3);
        void update(Eigen::Matrix<float, 5, 1>* bbox_, int cls_);
        Eigen::RowVectorXf predict();  
        Eigen::VectorXf get_state();

    public:
        /*variable*/
        static int count;
        Eigen::VectorXf bbox;// [5,1]
        KalmanFilterNew* kf;
        int time_since_update;
        int id;
        std::vector<Eigen::VectorXf> history;
        int hits;
        int hit_streak;
        int age = 0;
        float conf;
        int cls;
        Eigen::RowVectorXf last_observation = Eigen::RowVectorXf::Zero(5);
        std::unordered_map<int, Eigen::VectorXf> observations;
        std::vector<Eigen::VectorXf> history_observations;
        Eigen::RowVectorXf velocity = Eigen::RowVectorXf::Zero(2);// [2,1]
        int delta_t;
    };
}// namespace ocsort

#endif//OC_SORT_CPP_KALMANBOXTRACKER_HPP