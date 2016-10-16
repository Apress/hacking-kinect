//////////////////////////////////////////////////////////////////
// Simple Kinect SLAM demo
// by Daniel Herrera C.
//////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>
#include <list>

#include <pthread.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <boost/container/vector.hpp>

#include "kinect_slam_freenect.h"
#include "kinect_calibration.h"

namespace kinect_slam {

//////////////////////////////////////////////////////////////////
// CFeatureTrack : data structure that contains a reconstructed
//   point cloud and the camera pose for a registered frame.
//////////////////////////////////////////////////////////////////
class CFeatureTrack {
public:
    cv::Point2f base_position;
    cv::Point2f active_position;
    cv::Mat1f descriptor;
    int missed_frames;
};

//////////////////////////////////////////////////////////////////
// CTrackedView: data structure that contains a reconstructed
//   point cloud and the camera pose for a registered frame.
//////////////////////////////////////////////////////////////////
class CTrackedView {
public:
	boost::container::vector<pcl::PointXYZRGB> cloud;
    //std::vector<pcl::PointXYZRGB> cloud;
    cv::Matx33f R;
    cv::Matx31f T;
};

//////////////////////////////////////////////////////////////////
// CTrackingSharedData: data structure that contains shared
//   variables between the tracking module and other modules.
//////////////////////////////////////////////////////////////////
class CTrackingSharedData {
public:
    //Commands
    bool is_data_new;   //True if this class has new data that should be rendered
    bool is_tracking_enabled; //True if the tracking thread should process images

    //Base image
    cv::Matx33f base_R;
    cv::Matx31f base_T;
    cv::Mat3b *base_rgb;
    cv::Mat3f *base_pointmap;

    //Last tracked image
    cv::Mat3b *active_rgb;
    cv::Mat1s *active_depth;
    
    //Model
    std::list<CFeatureTrack> tracks; //Tracked features since last base frame
    std::vector<CTrackedView> views; //All registered views

    CTrackingSharedData();
    ~CTrackingSharedData();
};

//////////////////////////////////////////////////////////////////
// CTrackingModule: Executes the SLAM algorithm. Extracts 2D 
//   features, builds point clouds, and registers the point clouds.
//////////////////////////////////////////////////////////////////
class CTrackingModule
{
public:
    volatile bool die;

    pthread_mutex_t shared_mutex;
    CTrackingSharedData shared;

    CFreenectSharedData *freenect_data;
    
    CTrackingModule();
    ~CTrackingModule();

    void run();
    static void *thread_entry(void *instance);

private:
    CKinectCalibration calib;
    cv::Mat3b *rgb_buffer;
    cv::Mat1s *depth_buffer;

    void compute_pointmap(const cv::Mat1s &depth, cv::Mat3f &pointmap);
    void cloud_from_pointmap(const cv::Mat3b &rgb, const cv::Mat3f &pointmap, boost::container::vector<pcl::PointXYZRGB> &cloud);
    
    void match_features(const cv::Mat1f &new_descriptors, std::vector<int> &match_idx);
    static bool is_track_stale(const CFeatureTrack &track);
    void update_tracks(const std::vector<cv::KeyPoint> &feature_points, const cv::Mat1f &feature_descriptors, const std::vector<int> &match_idx);
    float get_median_feature_movement();
    
    void absolute_orientation(cv::Mat1f &X, cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T);
    void ransac_orientation(const cv::Mat1f &X, const cv::Mat1f &Y, cv::Matx33f &R, cv::Matx31f &T);
    void transformation_from_tracks(const cv::Mat3f &active_pointmap, cv::Matx33f &R, cv::Matx31f &T);
};

}