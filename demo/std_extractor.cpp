#include <Eigen/Geometry>
#include <mutex>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>

#include "include/STDesc.h"
#include "ros/init.h"

#include <chrono>

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;

std::mutex laser_mtx;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  std::unique_lock<std::mutex> lock(laser_mtx);

  laser_buffer.push(msg);
}

bool readPC(PointCloud::Ptr &cloud) {
  if (laser_buffer.empty())
    return false;

  auto laser_msg = laser_buffer.front();
  double laser_timestamp = laser_msg->header.stamp.toSec();
  pcl::fromROSMsg(*laser_msg, *cloud);
  std::unique_lock<std::mutex> l_lock(laser_mtx);
  laser_buffer.pop();
  return true;
}

void convertToMarkers(const std::vector<STDesc>& stds, visualization_msgs::MarkerArray& marker_array) {
        int id = 0;

    for (const auto& std : stds) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "velodyne";
        marker.header.stamp = ros::Time::now();
        marker.ns = "std_descriptors";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.03;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        geometry_msgs::Point p1, p2, p3;
        p1.x = std.vertex_A_(0);
        p1.y = std.vertex_A_(1);
        p1.z = std.vertex_A_(2);
        p2.x = std.vertex_B_(0);
        p2.y = std.vertex_B_(1);
        p2.z = std.vertex_B_(2);
        p3.x = std.vertex_C_(0);
        p3.y = std.vertex_C_(1);
        p3.z = std.vertex_C_(2);

        marker.points.push_back(p1);
        marker.points.push_back(p2);
        marker.points.push_back(p2);
        marker.points.push_back(p3);
        marker.points.push_back(p3);
        marker.points.push_back(p1);

        marker_array.markers.push_back(marker);
    }
}

void convertToPointCloud(const std::vector<STDesc>& stds, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    for (const auto& std : stds) {
        pcl::PointXYZ p1, p2, p3;
        p1.x = std.vertex_A_(0);
        p1.y = std.vertex_A_(1);
        p1.z = std.vertex_A_(2);
        p2.x = std.vertex_B_(0);
        p2.y = std.vertex_B_(1);
        p2.z = std.vertex_B_(2);
        p3.x = std.vertex_C_(0);
        p3.y = std.vertex_C_(1);
        p3.z = std.vertex_C_(2);

        cloud->points.push_back(p1);
        cloud->points.push_back(p2);
        cloud->points.push_back(p3);
    }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "online_demo");
  ros::NodeHandle nh;

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);

  ros::Publisher pubkeycurrent = nh.advertise<visualization_msgs::MarkerArray>("/std_descriptors", 1);
  ros::Publisher pub_points = nh.advertise<sensor_msgs::PointCloud2>("/std_points", 1);
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

  STDescManager *std_manager = new STDescManager(config_setting);

  while (ros::ok()) {
    ros::spinOnce();

    PointCloud::Ptr current_cloud(new PointCloud);

    if (readPC(current_cloud)) {
        down_sampling_voxel(*current_cloud, 0.1);
    
        // step1. Descriptor Extraction
        std::vector<STDesc> stds_vec;
        auto start = std::chrono::high_resolution_clock::now();
        std_manager->GenerateSTDescs(current_cloud, stds_vec);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        ROS_INFO("Extracted %lu ST descriptors in %f seconds", stds_vec.size(), elapsed.count());

        ////// visualizacion de los keypoints
        visualization_msgs::MarkerArray marker_array;
        convertToMarkers(stds_vec, marker_array);
        pubkeycurrent.publish(marker_array);
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.clear();  // Asegúrate de que el array de marcadores esté vacío
        marker_array.markers.push_back(delete_marker);
        pubkeycurrent.publish(marker_array);
        //////////////////////////////////////////

        ////// publicacion de nube de puntos en los vertices de los stds
        pcl::PointCloud<pcl::PointXYZ>::Ptr std_points(new pcl::PointCloud<pcl::PointXYZ>);
        convertToPointCloud(stds_vec, std_points);
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*std_points, output);
        output.header.frame_id = "velodyne";
        pub_points.publish(output);
        //////////////////////////////////////////
    }
  }

  return 0;
}