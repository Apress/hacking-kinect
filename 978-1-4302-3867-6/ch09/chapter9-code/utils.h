// Author: Nicolas Burrus
// Hacking the Kinect
#pragma once

#include <string>
#include <cassert>

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Geometry>

cv::Mat1f imread_Mat1f_raw(const std::string& filename);
void imwrite_Mat1f_raw(const std::string& filename, const cv::Mat1f& m);

void show_vector3f(const char* name, const Eigen::Vector3f& v);

void removePointsWithNanNormals(pcl::PointCloud<pcl::PointNormal>& cloud);
pcl::PointCloud<pcl::PointNormal>::Ptr preprocessImageCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr image_cloud);
pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr image_cloud);

template <class PointT>
typename pcl::PointCloud<PointT>::Ptr subsampleCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud, float voxel_size)
{
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    grid.setInputCloud(cloud);
    typename pcl::PointCloud<PointT>::Ptr subsampled_cloud (new pcl::PointCloud<PointT>);
    grid.filter(*subsampled_cloud);
    return subsampled_cloud;
}

