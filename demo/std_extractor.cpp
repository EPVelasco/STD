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
bool init_std = true;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
sensor_msgs::PointCloud2::ConstPtr msg_point;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  std::unique_lock<std::mutex> lock(laser_mtx);
  msg_point = msg;
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

void convertToMarkers(const std::vector<STDesc>& stds, visualization_msgs::MarkerArray& marker_array, const Eigen::Vector3f& color, float alpha = 1.0) {
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
        marker.color.r = color(0);  
        marker.color.g = color(1);  
        marker.color.b = color(2);  
        marker.color.a = alpha;     

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

  ros::Publisher pubkeycurr = nh.advertise<visualization_msgs::MarkerArray>("std_curr", 10);
  ros::Publisher pubkeyprev = nh.advertise<visualization_msgs::MarkerArray>("std_prev", 10);
  
  ros::Publisher pub_curr_points = nh.advertise<sensor_msgs::PointCloud2>("std_curr_points", 10);
  ros::Publisher pub_prev_points = nh.advertise<sensor_msgs::PointCloud2>("std_prev_points", 10);
  ros::Publisher pubSTD =   nh.advertise<visualization_msgs::MarkerArray>("pair_std", 10);

  ros::Publisher marker_pub_prev = nh.advertise<visualization_msgs::MarkerArray>("Axes_prev_STD", 10);
  ros::Publisher marker_pub_curr = nh.advertise<visualization_msgs::MarkerArray>("Axes_curr_STD", 10);

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

  STDescManager *std_manager = new STDescManager(config_setting);
  std::vector<STDesc> stds_curr;
  std::vector<STDesc> stds_prev;

  while (ros::ok()) {
    ros::spinOnce();

    PointCloud::Ptr current_cloud(new PointCloud);

    if (readPC(current_cloud)) {
        down_sampling_voxel(*current_cloud, 0.1);
    
        // step1. Descriptor Extraction

        auto start = std::chrono::high_resolution_clock::now();
        std_manager->GenerateSTDescs(current_cloud, stds_curr);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if(init_std){
          init_std = false;
          stds_prev = stds_curr;
          ROS_INFO("Inicial...");
        }

        ROS_INFO("Extracted %lu ST", stds_curr.size());
        ROS_INFO("Extracted %lu ST descriptors in %f seconds", stds_prev.size(), elapsed.count());


        ////// Llenando el std_pair con los stds prev y current
        std::vector<STDesc> std_pair;
        std_pair.insert(std_pair.end(), stds_prev.begin(), stds_prev.end());
        std_pair.insert(std_pair.end(), stds_curr.begin(), stds_curr.end());
        
        //////////////////////////////////////////////


        // step2. Searching pairs
        // std::pair<int, double> search_result(-1, 0);
        // std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        // loop_transform.first << 0, 0, 0;
        // loop_transform.second = Eigen::Matrix3d::Identity();
        // std::vector<std::pair<STDesc, STDesc>> loop_std_pair;

        // std_manager->SearchLoop(std_pair, search_result, loop_transform, loop_std_pair);
        // ROS_INFO("Pairs %lu STD", loop_std_pair.size());
        // publish_std_pairs(loop_std_pair, pubSTD);

        // // step3. Add descriptors to the database
        // std_manager->AddSTDescs(std_pair);


        ////// OK

        std::vector<std::pair<STDesc, STDesc>> matched_pairs;
        std_manager->MatchConsecutiveFrames(stds_prev, stds_curr, matched_pairs);
        ROS_INFO("Pairs %lu ST", matched_pairs.size());
        publish_std_pairs(matched_pairs, pubSTD);

        std_manager->publishAxes(marker_pub_prev, stds_prev, msg_point->header);
        std_manager->publishAxes(marker_pub_curr, stds_curr, msg_point->header);




        /////////////////// Data visualization //////////////////////////////////////////////
      
          ////// visualizacion de los keypoints current
          visualization_msgs::MarkerArray marker_array_curr;
          Eigen::Vector3f colorVector_curr(0.0f, 0.0f, 1.0f);  // azul

          convertToMarkers(stds_curr, marker_array_curr,colorVector_curr ,0.5 );
          pubkeycurr.publish(marker_array_curr);
          visualization_msgs::Marker delete_marker_curr;
          delete_marker_curr.action = visualization_msgs::Marker::DELETEALL;
          marker_array_curr.markers.clear();  // Asegúrate de que el array de marcadores esté vacío
          marker_array_curr.markers.push_back(delete_marker_curr);
          pubkeycurr.publish(marker_array_curr);
          //////////////////////////////////////////

          ////// publicacion de nube de puntos en los vertices de los stds
          pcl::PointCloud<pcl::PointXYZ>::Ptr std_points(new pcl::PointCloud<pcl::PointXYZ>);
          convertToPointCloud(stds_curr, std_points);
          sensor_msgs::PointCloud2 output_curr;
          pcl::toROSMsg(*std_points, output_curr);
          output_curr.header.frame_id = "velodyne";
          pub_curr_points.publish(output_curr);
          //////////////////////////////////////////

          /////////////////////// Previous std
          ////// visualizacion de los keypoints prev
          visualization_msgs::MarkerArray marker_array_prev;
          Eigen::Vector3f colorVector_prev(1.0f, 0.0f, 0.0f);  // azul
          convertToMarkers(stds_prev, marker_array_prev,colorVector_prev ,0.5 );
          pubkeyprev.publish(marker_array_prev);
          visualization_msgs::Marker delete_marker_prev;
          delete_marker_prev.action = visualization_msgs::Marker::DELETEALL;
          marker_array_prev.markers.clear();  // Asegúrate de que el array de marcadores esté vacío
          marker_array_prev.markers.push_back(delete_marker_prev);
          pubkeyprev.publish(marker_array_prev);
          //////////////////////////////////////////

          ////// publicacion de nube de puntos en los vertices de los stds
          pcl::PointCloud<pcl::PointXYZ>::Ptr std_points_prev(new pcl::PointCloud<pcl::PointXYZ>);
          convertToPointCloud(stds_prev, std_points_prev);
          sensor_msgs::PointCloud2 output_prev;
          pcl::toROSMsg(*std_points_prev, output_prev);
          output_prev.header.frame_id = "velodyne";
          pub_prev_points.publish(output_prev);
          //////////////////////////////////////////

        //////////////////////////////////////////////////////////////////////////////////////////////////

        stds_prev = stds_curr;
    }
  }

  return 0;
}