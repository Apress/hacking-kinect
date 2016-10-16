// Author: Nicolas Burrus
// Hacking the Kinect

#include "utils.h"

#include <fstream>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>

void removePointsWithNanNormals(pcl::PointCloud<pcl::PointNormal>& cloud)
{
    pcl::PointCloud<pcl::PointNormal> tmp;
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        if (pcl_isfinite(cloud.points[i].normal_x))
            tmp.push_back(cloud.points[i]);
    }
    cloud = tmp;
}

pcl::PointCloud<pcl::PointNormal>::Ptr preprocessImageCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr image_cloud)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr image_cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    image_cloud_with_normals = computeNormals(image_cloud);

    pcl::PassThrough<pcl::PointNormal> bbox_filter;
    bbox_filter.setFilterFieldName("z");
    bbox_filter.setFilterLimits(-1.5, -0.5);
    bbox_filter.setInputCloud(image_cloud_with_normals->makeShared());
    bbox_filter.filter(*image_cloud_with_normals);

    removePointsWithNanNormals(*image_cloud_with_normals);
    return image_cloud_with_normals;
}

pcl::PointCloud<pcl::PointNormal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr image_cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (image_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    // pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.01);
    ne.compute (*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr image_cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*image_cloud, *normals, *image_cloud_with_normals);
    return image_cloud_with_normals;
}

cv::Mat1f imread_Mat1f_raw(const std::string& filename)
{
  std::ifstream f (filename.c_str(), std::ios::binary);
  assert (f.good());
  int32_t rows = -1, cols = -1;
  f.read((char*)&rows, sizeof(int32_t));
  f.read((char*)&cols, sizeof(int32_t));

  cv::Mat1f m(rows, cols);
  f.read((char*)m.data, m.rows*m.cols*sizeof(float));
  assert(f.good());
  return m;
}

void imwrite_Mat1f_raw(const std::string& filename, const cv::Mat1f& m)
{
  std::ofstream f (filename.c_str(), std::ios::binary);
  assert(f.good());
  int32_t rows = m.rows, cols = m.cols;
  f.write((char*)&rows, sizeof(int32_t));
  f.write((char*)&cols, sizeof(int32_t));
  f.write((char*)m.data, m.rows*m.cols*sizeof(float));
  assert(f.good());
}

void show_vector3f(const char* name, const Eigen::Vector3f& v)
{
    PCL_INFO("%s: [%f %f %f]\n", name, v.x(), v.y(), v.z());
}
