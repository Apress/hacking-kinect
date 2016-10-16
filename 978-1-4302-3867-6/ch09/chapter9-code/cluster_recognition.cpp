// Author: Nicolas Burrus
// Hacking the Kinect
// Listings 9-21 to 9-25

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/vfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>

#include <boost/thread/thread.hpp>

#include <Eigen/Geometry>

#include "ply.h"
#include "utils.h"
#include "table_top_detector.h"

class CloudRecognizer
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef pcl::VFHSignature308 DescriptorType;
    typedef pcl::VFHEstimation<pcl::PointNormal, pcl::PointNormal, DescriptorType> FeatureExtractor;

public:
    struct Model
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        pcl::PointCloud<pcl::PointNormal>::ConstPtr cloud;
        DescriptorType descriptor;
        std::string name;
    };

public:
    CloudRecognizer()
    {}

    void addModel(pcl::PointCloud<pcl::PointNormal>::ConstPtr cloud, const std::string& name)
    {
        Model model;
        model.cloud = cloud;
        model.name = name;
        computeDescriptor(model.cloud, model.descriptor);
        models_.push_back(model);
    }

    // Return true if an object was detected.
    const Model* recognizeCloud(pcl::PointCloud<pcl::PointNormal>::ConstPtr cluster_cloud,
                                Eigen::Affine3f& model_pose)
    {
        DescriptorType cluster_descriptor;
        computeDescriptor(cluster_cloud, cluster_descriptor);
        const Model* closest_model = findClosestModel(cluster_descriptor);
        assert(closest_model);
        model_pose = computeModelPose(cluster_cloud, closest_model);
        return closest_model;
    }

private:
    Eigen::Vector3f toVector3f(const Eigen::Vector4f& v) const
    { return Eigen::Vector3f(v.x(), v.y(), v.z()); }

    Eigen::Affine3f computeModelPose(pcl::PointCloud<pcl::PointNormal>::ConstPtr cluster_cloud,
                                     const Model* model) const
    {
        Eigen::Vector4f cluster_centroid;
        pcl::compute3DCentroid(*cluster_cloud, cluster_centroid);

        Eigen::Vector4f model_centroid;
        pcl::compute3DCentroid(*model->cloud, model_centroid);

        Eigen::Affine3f transform (Eigen::Affine3d::Identity());
        transform.translate(toVector3f(cluster_centroid-model_centroid));

        pcl::PointCloud<pcl::PointNormal> centered_model_cloud;
        pcl::transformPointCloudWithNormals(*model->cloud, centered_model_cloud, transform);

        pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> reg;
        reg.setMaximumIterations (50);
        reg.setTransformationEpsilon (1e-7);
        reg.setMaxCorrespondenceDistance (0.05);
        reg.setInputCloud (centered_model_cloud.makeShared());
        reg.setInputTarget (cluster_cloud);
        pcl::PointCloud<pcl::PointNormal> aligned_cloud;
        reg.align (aligned_cloud);
        transform = reg.getFinalTransformation() * transform.matrix();
        return transform;
    }

    const Model* findClosestModel(const DescriptorType& descriptor) const
    {
        const Model* best_model = 0;
        float best_distance = FLT_MAX;
        for (size_t i = 0; i < models_.size(); ++i)
        {
            float dist = computeDistance(models_[i].descriptor, descriptor);
            std::clog << "dist with " << i << " = " << dist << std::endl;
            if (dist < best_distance)
            {
                best_distance = dist;
                best_model = &models_[i];
            }
        }
        return best_model;
    }

    // Euclidian distance
    float computeDistance(const DescriptorType& d1, const DescriptorType& d2) const
    {
        float dist = 0;
        const int descriptor_size = sizeof(d1.histogram) / sizeof(float);
        for (int i = 0; i < descriptor_size; ++i)
        {
            float diff = d1.histogram[i] - d2.histogram[i];
            dist += diff * diff;
        }
        return dist;
    }

    void computeDescriptor(pcl::PointCloud<pcl::PointNormal>::ConstPtr cloud, DescriptorType& descriptor) const
    {
        pcl::search::KdTree<pcl::PointNormal>::Ptr search_tree (new pcl::search::KdTree<pcl::PointNormal> ());
        pcl::PointCloud<DescriptorType> descriptors;

        pcl::PointCloud<pcl::PointNormal>::Ptr subsampled_object_cloud;
        subsampled_object_cloud = subsampleCloud<pcl::PointNormal>(cloud, 0.003);

        pcl::PointCloud<pcl::PointNormal>::Ptr object_normals (new pcl::PointCloud<pcl::PointNormal> ());
        pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> normal_estimator;
        normal_estimator.setInputCloud (subsampled_object_cloud);
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal> ());
        normal_estimator.setSearchMethod (tree);
        normal_estimator.setRadiusSearch (0.01);
        normal_estimator.compute (*object_normals);

        FeatureExtractor extractor;
        extractor.setInputCloud (subsampled_object_cloud);
        extractor.setInputNormals (object_normals);
        extractor.setSearchMethod (search_tree);
        extractor.compute (descriptors);

        assert(descriptors.points.size() == 1); // should be single descriptor

        descriptor = descriptors.points[0];
    }

private:
    std::vector<Model> models_;
};

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Match each cluster in image.pcd with the provided models." << std::endl;
        std::cerr << "\nUsage: cluster_recognition image.pcd model_1.pcd model_2.pcd ..." << std::endl;
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr image_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile(argv[1], *image_cloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr image_cloud_with_normals;
    image_cloud_with_normals = preprocessImageCloud(image_cloud);

    CloudRecognizer recognizer;

    for (int i = 2; i < argc; ++i)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr model_cloud (new pcl::PointCloud<pcl::PointNormal>);
        pcl::io::loadPCDFile(argv[i], *model_cloud);
        recognizer.addModel(model_cloud, argv[i]);
    }

    TableTopDetector<pcl::PointNormal> detector;
    detector.initialize();
    detector.detect(*image_cloud_with_normals);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    Eigen::Vector3i color (0, 0, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color1(image_cloud_with_normals, color[0], color[1], color[2]);
    viewer->addPointCloud<pcl::PointNormal> (image_cloud_with_normals, single_color1, "image");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0, "image");

    for (size_t i = 0; i < detector.objects().size(); ++i)
    {
        pcl::PointCloud<pcl::PointNormal>::ConstPtr cluster = detector.objects()[i];
        Eigen::Affine3f pose;
        const CloudRecognizer::Model* model = recognizer.recognizeCloud(cluster, pose);

        pcl::PointCloud<pcl::PointNormal>::Ptr aligned_model (new pcl::PointCloud<pcl::PointNormal>);
        pcl::transformPointCloudWithNormals(*model->cloud, *aligned_model, pose);

        Eigen::Vector3i color (255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> single_color(aligned_model, color[0], color[1], color[2]);
        pcl::PointCloud<pcl::PointNormal>::Ptr subsampled_model;
        subsampled_model = subsampleCloud<pcl::PointNormal>(aligned_model, 0.01);
        viewer->addPointCloud<pcl::PointNormal> (subsampled_model, single_color, cv::format("object %d", i));
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cv::format("object %d", i));

        pcl::PointNormal text_pos;
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid (*cluster, centroid);
        text_pos.getVector4fMap() = centroid;
        std::string model_name = model->name;
        // remove the .pcd extension
        if (model_name.size() > 4)
            model_name.erase(model_name.size()-3, model_name.size());
        viewer->addText3D(model_name, text_pos, 0.02, 255, 0, 0, cv::format("text %d", i));
    }

    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return 0;
}
