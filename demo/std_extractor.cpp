#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <unordered_map>
#include <vector>

// Incluye tus funciones de extracción de descriptores aquí
#include "include/STDesc.h"

class STDExtractor {
public:
    STDExtractor() {
        sub_ = nh_.subscribe("/velodyne_points", 1, &STDExtractor::pointCloudCallback, this);
        pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/std_descriptors", 1);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        if (cloud->empty()) {
            ROS_ERROR("Received an empty point cloud!");
            return;
        }

        ROS_INFO("Received point cloud with %lu points", cloud->points.size());

        // Aquí se realiza la extracción de descriptores
        std::vector<STDesc> stds;
        STDescManager manager;
        manager.GenerateSTDescs(cloud, stds);

        ROS_INFO("Extracted %lu ST descriptors", stds.size());

        // Convertir descriptores a marcadores para RVIZ
        visualization_msgs::MarkerArray marker_array;
        convertToMarkers(stds, marker_array);

        pub_.publish(marker_array);
    }

    void convertToMarkers(const std::vector<STDesc>& stds, visualization_msgs::MarkerArray& marker_array) {
        int id = 0;
        for (const auto& std : stds) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "base_link";
            marker.header.stamp = ros::Time::now();
            marker.ns = "std_descriptors";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::LINE_LIST;
            marker.action = visualization_msgs::Marker::ADD;
            marker.scale.x = 0.01;
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

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "std_extractor_node");
    ros::NodeHandle nh;
    ConfigSetting config_setting;
    read_parameters(nh, config_setting);

    STDExtractor extractor;
    ros::spin();
    return 0;
}
