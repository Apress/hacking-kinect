// Author: Nicolas Burrus
// Hacking the Kinect
// Listings 9-16 to 9-20

#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>

#include <boost/thread/thread.hpp>

#include <Eigen/Geometry>

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "aruco/aruco.h"

#include "utils.h"

const float cx = 320.0;
const float cy = 267.0;
const float fx = 522.995;
const float fy = 522.995;

class MarkedViewpoint
{
public:
    MarkedViewpoint()
    {}

    const cv::Size2f& boardSize() const { return board_size_; }

    void computeAlignedPointCloud(const cv::Mat3b& rgb_image,
                                  const cv::Mat1f& depth_image,
                                  pcl::PointCloud<pcl::PointXYZ>::ConstPtr image_cloud,
                                  pcl::PointCloud<pcl::PointNormal>& output_cloud)
    {
        Eigen::Affine3f pose;
        bool ok = estimateCameraToBoardTransform(pose, rgb_image, depth_image);
        if (!ok)
            return;

        cv::waitKey(10);

        pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal> normals;
        normal_estimator.setNormalEstimationMethod (normal_estimator.AVERAGE_3D_GRADIENT);
        normal_estimator.setMaxDepthChangeFactor(0.05f);
        normal_estimator.setNormalSmoothingSize(5.0f);
        normal_estimator.setInputCloud(image_cloud);
        normal_estimator.compute(normals);

        pcl::concatenateFields(*image_cloud, normals, output_cloud);
        removePointsWithNanNormals(output_cloud);
        pcl::transformPointCloudWithNormals(output_cloud, output_cloud, pose);

        cropCloudOnAxis(output_cloud, "z", 0.03, 0.4); // Maximal height set to 40cm
        cropCloudOnAxis(output_cloud, "y", 0.00, board_size_.height);
        cropCloudOnAxis(output_cloud, "x", 0.00, board_size_.width);
    }

private:
    Eigen::Vector3f computeMarkerCenter(const aruco::Marker& marker, const cv::Mat1f& depth_image)
    {
        cv::Point2f image_center (0,0);
        for (int i = 0; i < 4; ++i)
            image_center += marker[i];
        image_center *= 1/4.f;

        float mean_depth = 0;
        int n = 0;
        for (int dy = -3; dy < 3; ++dy)
        for (int dx = -3; dx < 3; ++dx)
        {
            int row = roundf(image_center.y + dy);
            int col = roundf(image_center.x + dx);

            if (depth_image(row, col) > 1e-5) // valid depth point
            {
                mean_depth += depth_image(row, col);
                n += 1;
            }
        }
        assert(n > 0);
        mean_depth /= n;
        Eigen::Vector3f center_3d (mean_depth * (image_center.x-cx)/fx, mean_depth * -(image_center.y-cy)/fy, -mean_depth);
        show_vector3f("center3d", center_3d);
        return center_3d;
    }

    bool estimateCameraToBoardTransform(Eigen::Affine3f& camera_to_board,
                                        const cv::Mat3b& rgb_image,
                                        const cv::Mat1f& depth_image)
    {
        aruco::MarkerDetector detector;
        std::vector<aruco::Marker> markers;
        cv::Mat tmp (rgb_image);
        detector.detect(tmp, markers);

        cv::Mat3b debug_img; rgb_image.copyTo(debug_img);
        for (size_t i = 0; i < markers.size(); ++i)
            markers[i].draw(debug_img, Scalar(255,0,0));
        cv::imshow("markers", debug_img);

        if (markers.size() < 3)
            return false; // at least 3 markers needed.

        // Index markers by id name.
        std::vector<aruco::Marker*> markers_by_id (4, (aruco::Marker*) 0);
        for (size_t i = 0; i < markers.size(); ++i)
            markers_by_id[markers[i].id] = &markers[i];

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > marker_centers (4);
        for (size_t i = 0; i < markers_by_id.size(); ++i)
        {
            if (markers_by_id[i])
                marker_centers[i] = computeMarkerCenter(*markers_by_id[i], depth_image);
        }

        Eigen::Vector3f x_axis;
        Eigen::Vector3f y_axis;
        Eigen::Vector3f origin;

        if (markers_by_id[0] && markers_by_id[2])            
            x_axis = marker_centers[2] - marker_centers[0];
        else
            x_axis = marker_centers[3] - marker_centers[1];

        if (markers_by_id[0] && markers_by_id[1])
            y_axis = marker_centers[0] - marker_centers[1];
        else
            y_axis = marker_centers[2] - marker_centers[3];

        if (markers_by_id[1])
            origin = marker_centers[1];
        else
            origin = marker_centers[0] - y_axis;

        board_size_.width = x_axis.norm();
        board_size_.height = y_axis.norm();

        x_axis.normalize();
        y_axis.normalize();
        Eigen::Vector3f z_axis = x_axis.cross(y_axis);

        show_vector3f("X", x_axis);
        show_vector3f("Y", y_axis);
        show_vector3f("Z", z_axis);
        show_vector3f("origin", origin);

        Eigen::Affine3f board_to_camera (Eigen::Affine3f::Identity());
        board_to_camera.matrix().block<3,1>(0,0) = x_axis;
        board_to_camera.matrix().block<3,1>(0,1) = y_axis;
        board_to_camera.matrix().block<3,1>(0,2) = z_axis;
        board_to_camera.translation() = origin;
        camera_to_board = board_to_camera.inverse();

        return true;
    }

    void cropCloudOnAxis(pcl::PointCloud<pcl::PointNormal>& cloud, const char* axis, float min_value, float max_value)
    {
        pcl::PointCloud<pcl::PointNormal> tmp;
        pcl::PassThrough<pcl::PointNormal> bbox_filter;
        bbox_filter.setFilterFieldName(axis);
        bbox_filter.setFilterLimits(min_value, max_value);
        bbox_filter.setInputCloud(cloud.makeShared());
        bbox_filter.filter(tmp);
        cloud = tmp;
    }

private:
    cv::Size2f board_size_;
};

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        std::cerr << "Output cloud will contain the cropped object area in global coordinate system." << std::endl;
        std::cerr << "\nUsage: marker_modeling rgb_image depth_image image_cloud.pcd output_cloud" << std::endl;
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr image_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(argv[3], *image_cloud);

    cv::Mat3b rgb_image = imread(argv[1]);
    assert(rgb_image.data);

    // Depth image as output by RGBDemo.
    cv::Mat1f depth_image = imread_Mat1f_raw(argv[2]);
    assert(depth_image.data);

    MarkedViewpoint modeler;
    pcl::PointCloud<pcl::PointNormal>::Ptr view_cloud (new pcl::PointCloud<pcl::PointNormal>);
    modeler.computeAlignedPointCloud(rgb_image, depth_image, image_cloud, *view_cloud);

    pcl::io::savePLYFile(std::string(argv[4]) + ".ply", *view_cloud);
    pcl::io::savePCDFile(std::string(argv[4]) + ".pcd", *view_cloud);

    // Show the point cloud.
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    cv::Vec3b color (0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color2(view_cloud, color[0], color[1], color[2]);
    viewer->addPointCloud<pcl::PointNormal> (view_cloud, single_color2, "cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // Show the crop box.
    pcl::ModelCoefficients cube;
    cube.values.resize(10);
    cube.values[3] = cube.values[4] = cube.values[5] = 0;
    cube.values[6] = 0;
    cube.values[7] = modeler.boardSize().width;
    cube.values[8] = modeler.boardSize().height;
    cube.values[9] = 0.4 - 0.03;
    cube.values[0] = 0 + cube.values[7]/2;
    cube.values[1] = 0 + cube.values[8]/2;
    cube.values[2] = 0.03 + cube.values[9]/2;
    viewer->addCube(cube, "crop");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, "crop");

    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return 0;
}
