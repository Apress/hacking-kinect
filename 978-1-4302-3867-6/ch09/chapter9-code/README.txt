This source code corresponds to the four programs introduced in the chapter 9 of Hacking the Kinect.

COMPILATION
===========

You first need to install PCL >= 1.5 and OpenCV >= 2.3.1. Then use cmake to compile the sample programs.

Example on Linux:
cd chapter9
mkdir build
cmake ..
make

USAGE
=====

table_top_detector_main
=======================

Extracts objects clouds that lie on top of a table.

Example of usage:
./table_top_detector_main data/table/cloud.pcd

=> generates one point cloud per object model00.pcd, model01.pcd, ...

cloud_extrusion
===============

Extrude object point clouds.

Example of usage:
./cloud_extrusion data/table/cloud.pcd

=> generates one mesh per object model00.ply, model01.ply, ...
=> these ply files can be opened with Meshlab or Blender

marked_viewpoint_modeler
========================

Crop and locate an object view in a consistent global coordinate
frame using markers.

Example of usage:
./marked_viewpoint_modeler data/statue/view0000/raw/color.png data/statue/view0000/raw/depth.raw data/statue/view0000/cloud.pcd aligned_view0000

This kind of dataset is easily generated using RGBDemo.

=> generates aligned_view0000.pcd and aligned_view0000.ply

=> applying it to all images in data/statue will generates a set
   of aligned partial views that can be loaded into Meshlab for
   further refinement

cloud_recognition
=================

Associate the objects lying on a table with a set of reference models.

Example of usage:
./cloud_recognition data/table/cloud.pcd data/models/*.pcd
