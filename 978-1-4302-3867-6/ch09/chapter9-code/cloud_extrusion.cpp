// Author: Nicolas Burrus
// Hacking the Kinect
// Listings 9-5 to 9-15

#include "table_top_detector.h"

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/octree/octree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/msac.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/ply_io.h>

#include <boost/thread/thread.hpp>

// Options
bool show_table_hull = true;
bool show_voxels = false;

class Extruder
{
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloudType;
    typedef PointCloudType::Ptr PointCloudPtr;
    typedef PointCloudType::ConstPtr PointCloudConstPtr;

    struct Voxel
    {
        Voxel(int c = 0, int r = 0, int d = 0) : c(c), r(r), d(d) {}

        bool operator<(const Voxel& rhs) const
        {
            if (d < rhs.d) return true;
            if (rhs.d < d) return false;
            if (r < rhs.r) return true;
            if (rhs.r < r) return false;
            if (c < rhs.c) return true;
            if (rhs.c < c) return false;
            return false; // equals, return false.
        }

        int c, r, d; // col, row, depth.
    };

public:
    Extruder() : voxel_size_ (0.01) {}

public:
    void setInputCloud(PointCloudConstPtr cloud)
    { input_cloud_ = cloud; }

    void setTablePlane(const Eigen::Vector4f& coeffs)
    { plane_coeffs_ = coeffs; }

    void setVoxelSize(float size) { voxel_size_ = size; }
    float voxelSize() const { return voxel_size_; }

public:
    void compute(pcl::PointCloud<pcl::PointNormal>& output_cloud,
                 pcl::PolygonMesh& output_mesh)
    {
        buildVoxelMap(input_cloud_);
        PCL_INFO("%d voxels\n", voxels_.size());

        extrudeVoxelSet();
        PCL_INFO("%d voxels after filling\n", voxels_.size());

        dilateVoxelMap(3);
        erodeVoxelMap(3);

        buildOutputCloud(output_cloud);
        buildOutputMesh(output_cloud, output_mesh);
    }

    void extrudeVoxelSet()
    {
        // Determine the plane normal.
        Eigen::Vector3f plane_normal (plane_coeffs_[0], plane_coeffs_[1], plane_coeffs_[2]);
        plane_normal.normalize();

        std::set<Voxel> filled_voxels;
        for (std::set<Voxel>::const_iterator it = voxels_.begin();
             it != voxels_.end();
             ++it)
        {
            Eigen::Vector3f origin = voxelToPoint(*it);
            Eigen::Vector3f end = lineIntersectionWithPlane(origin, origin + plane_normal);

            // Walk along the line segment using small steps.
            float step_size = voxel_size_ * 0.1f;

            // Determine the number of steps.
            int nsteps = (end-origin).norm() / step_size;

            for (int k = 0; k < nsteps; ++k)
            {
                Eigen::Vector3f p = origin + plane_normal * (float) step_size * (float) k;
                filled_voxels.insert(pointToVoxel(p));
            }
        }
        voxels_ = filled_voxels;
    }

protected:
    void buildOutputCloud(pcl::PointCloud<pcl::PointNormal>& output)
    {
        for (std::set<Voxel>::const_iterator it = voxels_.begin();
             it != voxels_.end();
             ++it)
        {
            if (isInnerVoxel(*it))
                continue;

            pcl::PointNormal p;
            p.getVector3fMap() = voxelToPoint(*it);
            p.getNormalVector3fMap() = computeNormal(*it);
            output.push_back(p);
        }
    }

    void buildOutputMesh(const pcl::PointCloud<pcl::PointNormal>& cloud,
                         pcl::PolygonMesh& mesh)
    {
        pcl::Poisson<pcl::PointNormal> poisson;
        poisson.setInputCloud(cloud.makeShared());
        poisson.performReconstruction(mesh);
    }

    void dilateVoxelMap(int k)
    {
        std::set<Voxel> dilated_set;
        for (std::set<Voxel>::const_iterator it = voxels_.begin();
             it != voxels_.end();
             ++it)
        {
            const Voxel& v = *it;
            for (int dc = -k; dc <= k; ++dc)
                for (int dr = -k; dr <= k; ++dr)
                    for (int dd = -k; dd <= k; ++dd)
                    {
                        Voxel dv (v.c + dc, v.r + dr, v.d + dd);
                        dilated_set.insert(dv);
                    }
        }
        voxels_ = dilated_set;
    }

    void erodeVoxelMap(int k)
    {
        std::set<Voxel> eroded_set;
        for (std::set<Voxel>::const_iterator it = voxels_.begin();
             it != voxels_.end();
             ++it)
        {
            const Voxel& v = *it;
            bool has_all_neighbors = true;

            for (int dc = -k; dc <= k; ++dc)
                for (int dr = -k; dr <= k; ++dr)
                    for (int dd = -k; dd <= k; ++dd)
                    {
                        Voxel dv (v.c + dc, v.r + dr, v.d + dd);
                        if (voxels_.find(dv) == voxels_.end())
                            has_all_neighbors = false;
                    }

            if (has_all_neighbors)
                eroded_set.insert(*it);
        }
        voxels_ = eroded_set;
    }

    bool isInnerVoxel(const Voxel& v) const
    {
        for (int dc = -1; dc <= 1; ++dc)
            for (int dr = -1; dr <= 1; ++dr)
                for (int dd = -1; dd <= 1; ++dd)
                {
                    // Skip v itself.
                    if (dc == 0 && dr == 0 && dd == 0)
                        continue;

                    Voxel dv (v.c + dc, v.r + dr, v.d + dd);
                    if (voxels_.find(dv) == voxels_.end())
                    {
                        return false;
                    }
                }

        return true;
    }

    Eigen::Vector3f computeNormal(const Voxel& v)
    {
        Eigen::Vector3f p = voxelToPoint(v);
        Eigen::Vector3f normal (0, 0, 0);
        int nb_neighbors = 0;

        for (int dc = -1; dc <= 1; ++dc)
            for (int dr = -1; dr <= 1; ++dr)
                for (int dd = -1; dd <= 1; ++dd)
                {
                    if (dc == 0 && dr == 0 && dd == 0)
                        continue;

                    Voxel dv (v.c + dc, v.r + dr, v.d + dd);
                    if (voxels_.find(dv) != voxels_.end())
                        continue;

                    Eigen::Vector3f neighbor = voxelToPoint(dv);
                    Eigen::Vector3f direction = neighbor - p;
                    normal += direction;

                    nb_neighbors += 1;
                }

        if (nb_neighbors < 1)
            return Eigen::Vector3f(0,0,0);

        normal *= (1.0f / nb_neighbors);
        normal.normalize();
        return normal;
    }

    Eigen::Vector3f lineIntersectionWithPlane(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) const
    {
        const float a = plane_coeffs_[0];
        const float b = plane_coeffs_[1];
        const float c = plane_coeffs_[2];
        const float d = plane_coeffs_[3];

        double u = a*p1.x() + b*p1.y() + c*p1.z() + d;
        u /= a*(p1.x()-p2.x()) + b*(p1.y()-p2.y()) + c*(p1.z()-p2.z());
        Eigen::Vector3f r = p1 + u * (p2-p1);
        return r;
    }

    void buildVoxelMap(PointCloudConstPtr cloud)
    {
        for (int i = 0; i < cloud->points.size(); ++i)
        {
            Voxel v = pointToVoxel(cloud->points[i].getVector3fMap());
            voxels_.insert(v);
            assert(voxels_.find(v) != voxels_.end());
        }
    }

    Eigen::Vector3f voxelToPoint(const Voxel& v) const
    { return Eigen::Vector3f(v.c * voxel_size_, v.r * voxel_size_, v.d * voxel_size_); }

    Voxel pointToVoxel(const Eigen::Vector3f& p) const
    { return Voxel(p.x() / voxel_size_, p.y() / voxel_size_, p.z() / voxel_size_); }

private:
    PointCloudConstPtr input_cloud_;
    Eigen::Vector4f plane_coeffs_;
    float voxel_size_;
    std::set<Voxel> voxels_;
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: table_top_detector_main cloud_file" << std::endl;
        return -1;
    }

    // initialize PointClouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PCDReader reader;
    reader.read(argv[1], *cloud);

    TableTopDetector<pcl::PointXYZ> detector;
    detector.initialize();
    detector.detect(*cloud);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);

    if (show_table_hull)
    {
        viewer->addPolygonMesh(*detector.tableHullMesh(), "table");
    }

    for (int cluster_id = 0; cluster_id < detector.objects().size(); ++cluster_id)
    {
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud = detector.objects()[cluster_id];

        Extruder extruder;
        extruder.setInputCloud(cloud);
        extruder.setTablePlane(detector.plane());
        extruder.setVoxelSize(0.004);
        pcl::PointCloud<pcl::PointNormal>::Ptr extruded_cloud (new pcl::PointCloud<pcl::PointNormal>);
        pcl::PolygonMesh extruded_mesh;
        extruder.compute(*extruded_cloud, extruded_mesh);
        pcl::io::savePLYFile(cv::format("model%02d.ply", cluster_id), extruded_mesh);
        std::clog << cv::format("Extruded_cloud.size(): %d\n", extruded_cloud->points.size());

        viewer->addPolygonMesh(extruded_mesh, cv::format("object_mesh_%d", cluster_id));

        // Show one cube per voxel.
        if (show_voxels)
        {
            pcl::ModelCoefficients cube;
            cube.values.resize(10);
            cube.values[3] = cube.values[4] = cube.values[5] = 0;
            cube.values[6] = 0;
            cube.values[7] = cube.values[8] = cube.values[9] = extruder.voxelSize();
            for (int i = 0; i < extruded_cloud->points.size(); ++i)
            {
                cube.values[0] = extruded_cloud->points[i].x;
                cube.values[1] = extruded_cloud->points[i].y;
                cube.values[2] = extruded_cloud->points[i].z;
                viewer->addCube(cube, cv::format("cube %d", i));
            }
        }
    }

    viewer->initCameraParameters ();

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    return 0;
}
