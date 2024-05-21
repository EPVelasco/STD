#include <Eigen/Geometry>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <iostream>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// PCL
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Internal library
#include "include/STDesc.h"
#include "include/KDTree_STD.h"

// ROS
#include <ros/ros.h>
#include "ros/init.h"
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <pcl/filters/voxel_grid.h>


// Time
#include <chrono>

#include <random>

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;

std::mutex laser_mtx;
std::mutex odom_mtx;

bool init_std = true;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

sensor_msgs::PointCloud2::ConstPtr msg_point;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    std::unique_lock<std::mutex> lock(laser_mtx);
    msg_point = msg;
    laser_buffer.push(msg);
}

void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg) {
    std::unique_lock<std::mutex> lock(odom_mtx);
    odom_buffer.push(msg);
}

//////////////////////////////// Sincronización de los datos:
bool syncPackages(PointCloud::Ptr &cloud, Eigen::Affine3d &pose) {
    if (laser_buffer.empty() || odom_buffer.empty())
        return false;

    auto laser_msg = laser_buffer.front();
    double laser_timestamp = laser_msg->header.stamp.toSec();

    auto odom_msg = odom_buffer.front();
    double odom_timestamp = odom_msg->header.stamp.toSec();

    // check if timestamps are matched
    if (abs(odom_timestamp - laser_timestamp) < 1e-3) {
        pcl::fromROSMsg(*laser_msg, *cloud);

        Eigen::Quaterniond r(
            odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
            odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
        Eigen::Vector3d t(odom_msg->pose.pose.position.x,
                          odom_msg->pose.pose.position.y,
                          odom_msg->pose.pose.position.z);

        pose = Eigen::Affine3d::Identity();
        pose.translate(t);
        pose.rotate(r);

        std::unique_lock<std::mutex> l_lock(laser_mtx);
        std::unique_lock<std::mutex> o_lock(odom_mtx);

        laser_buffer.pop();
        odom_buffer.pop();

    } else if (odom_timestamp < laser_timestamp) {
        ROS_WARN("Current odometry is earlier than laser scan, discard one "
                 "odometry data.");
        std::unique_lock<std::mutex> o_lock(odom_mtx);
        odom_buffer.pop();
        return false;
    } else {
        ROS_WARN(
            "Current laser scan is earlier than odometry, discard one laser scan.");
        std::unique_lock<std::mutex> l_lock(laser_mtx);
        laser_buffer.pop();
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////

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

void printSTDesc(const STDesc& desc) {
    std::cout << "Side Lengths: " << desc.side_length_.transpose() << std::endl;
    std::cout << "Angles: " << desc.angle_.transpose() << std::endl;
    std::cout << "Center: " << desc.center_.transpose() << std::endl;
    std::cout << "Vertex A: " << desc.vertex_A_.transpose() << std::endl;
    std::cout << "Vertex B: " << desc.vertex_B_.transpose() << std::endl;
    std::cout << "Vertex C: " << desc.vertex_C_.transpose() << std::endl;
    std::cout << "Frame ID: " << desc.frame_id_ << std::endl;
}

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

void addDescriptorToMatrix(Eigen::MatrixXf& mat, const STDesc& desc, int row) {
    Eigen::Vector3f side_length = desc.side_length_.cast<float>();
    Eigen::Vector3f angle = desc.angle_.cast<float>();
    Eigen::Vector3f center = desc.center_.cast<float>();
    Eigen::Vector3f vertex_A = desc.vertex_A_.cast<float>();
    Eigen::Vector3f vertex_B = desc.vertex_B_.cast<float>();
    Eigen::Vector3f vertex_C = desc.vertex_C_.cast<float>();

    mat.block<1, 3>(row, 0) = side_length.transpose();
    mat.block<1, 3>(row, 3) = angle.transpose();
    mat.block<1, 3>(row, 6) = center.transpose();
    mat.block<1, 3>(row, 9) = vertex_A.transpose();
    mat.block<1, 3>(row, 12) = vertex_B.transpose();
    mat.block<1, 3>(row, 15) = vertex_C.transpose();
}

void updateMatrixAndKDTree(Eigen::MatrixXf& mat, std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>& index, const std::deque<STDesc>& std_local_map) {
    int num_desc = std_local_map.size();
    mat.resize(num_desc, 18);

    // Rellenar la matriz con los descriptores actuales
    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }

    // Recrear el KD-Tree con la matriz actualizada
    index = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>(18, std::cref(mat), 10 /* max leaf */);
    index->index_->buildIndex();
}

// Función para generar colores aleatorios
std::tuple<float, float, float> getRandomColor() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return std::make_tuple(dis(gen), dis(gen), dis(gen));
}

void generateArrow(const STDesc& desc1, const STDesc& desc2, visualization_msgs::MarkerArray& marker_array, int& id, const std_msgs::Header& header) {
    visualization_msgs::Marker arrow;
    arrow.header = header;
    arrow.ns = "std_matches";
    arrow.id = id++;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.scale.x = 0.05;  // Grosor del cuerpo de la flecha
    arrow.scale.y = 0.2;  // Grosor de la cabeza de la flecha
    arrow.scale.z = 0.4;   // Longitud de la cabeza de la flecha
    
    // Generar color aleatorio
    auto [r, g, b] = getRandomColor();
    arrow.color.r = r;
    arrow.color.g = g;
    arrow.color.b = b;
    arrow.color.a = 1.0;

    // Punto de inicio (centro del descriptor 1)
    geometry_msgs::Point start;
    start.x = desc1.center_(0);
    start.y = desc1.center_(1);
    start.z = desc1.center_(2);

    // Punto final (centro del descriptor 2)
    geometry_msgs::Point end;
    end.x = desc2.center_(0);
    end.y = desc2.center_(1);
    end.z = desc2.center_(2);

    arrow.points.push_back(start);
    arrow.points.push_back(end);

    marker_array.markers.push_back(arrow);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "STD_descriptor");
    ros::NodeHandle nh;

    ConfigSetting config_setting;
    read_parameters(nh, config_setting);

    ros::Publisher pubkeycurr = nh.advertise<visualization_msgs::MarkerArray>("std_curr", 10);
    ros::Publisher pubkeyprev = nh.advertise<visualization_msgs::MarkerArray>("std_prev", 10);    
    ros::Publisher pub_curr_points = nh.advertise<sensor_msgs::PointCloud2>("std_curr_points", 10);
    ros::Publisher pub_prev_points = nh.advertise<sensor_msgs::PointCloud2>("std_prev_points", 10);
    ros::Publisher pubSTD = nh.advertise<visualization_msgs::MarkerArray>("pair_std", 10);
    ros::Publisher marker_pub_prev = nh.advertise<visualization_msgs::MarkerArray>("Axes_prev_STD", 10);
    ros::Publisher marker_pub_curr = nh.advertise<visualization_msgs::MarkerArray>("Axes_curr_STD", 10);
    ros::Publisher pose_pub_prev = nh.advertise<geometry_msgs::PoseArray>("std_prev_poses", 10);
    ros::Publisher pose_pub_curr = nh.advertise<geometry_msgs::PoseArray>("std_curr_poses", 10);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/odom", 100, OdomHandler);

    STDescManager *std_manager = new STDescManager(config_setting);
    std::vector<STDesc> stds_curr;
    std::vector<STDesc> stds_prev;

    // matrix of the std_descrptor to std_descriptor
    using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    std::deque<STDesc> std_local_map;
    std::deque<int> counts_per_iteration;

    Eigen::MatrixXf mat(0, 18);
    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>> index;
    
    PointCloud::Ptr current_cloud_world(new PointCloud);
    PointCloud::Ptr current_cloud(new PointCloud);
    Eigen::Affine3d pose; // odometria 
    Eigen::Affine3d pose_prev = Eigen::Affine3d::Identity(); // odometria 
    int cont_itera = 0;

    while (ros::ok()) {
        ros::spinOnce();
        std::vector<STDesc> stds_curr_pair;
        std::vector<STDesc> stds_prev_pair;

        if (syncPackages(current_cloud, pose)) {           

            // step1. Descriptor Extraction
            auto start = std::chrono::high_resolution_clock::now();                     
            
            int cont_desc_pairs = 0;
            if (init_std) {
                init_std = false;
                down_sampling_voxel(*current_cloud, config_setting.ds_size_);
                std_manager->GenerateSTDescs(current_cloud, stds_curr);
                stds_prev = stds_curr;
                ROS_INFO("++++++++++ Iniciando Extraccion de STD ++++++++");
            } else { 
                  // //////////////////////////////////////////////////

                pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose_prev);
                pose_prev = pose;
                down_sampling_voxel(*current_cloud_world, config_setting.ds_size_);
                std_manager->GenerateSTDescs(current_cloud_world, stds_curr);

               

                // Comparar stds_curr con std_local_map usando el KD-Tree actualizado
                if (!stds_curr.empty()) {
                    visualization_msgs::MarkerArray marker_array;
                    int id = 0;
 
                    for (const auto& desc : stds_curr) {
                        std::vector<float> query;
                        Eigen::Vector3f side_length = desc.side_length_.cast<float>();
                        Eigen::Vector3f angle = desc.angle_.cast<float>();
                        Eigen::Vector3f center = desc.center_.cast<float>();
                        Eigen::Vector3f vertex_A = desc.vertex_A_.cast<float>();
                        Eigen::Vector3f vertex_B = desc.vertex_B_.cast<float>();
                        Eigen::Vector3f vertex_C = desc.vertex_C_.cast<float>();

                        query.insert(query.end(), side_length.data(), side_length.data() + 3);
                        query.insert(query.end(), angle.data(), angle.data() + 3);
                        query.insert(query.end(), center.data(), center.data() + 3);
                        query.insert(query.end(), vertex_A.data(), vertex_A.data() + 3);
                        query.insert(query.end(), vertex_B.data(), vertex_B.data() + 3);
                        query.insert(query.end(), vertex_C.data(), vertex_C.data() + 3);

                        // Buscar el descriptor más cercano
                        const size_t num_results = 1;
                        std::vector<size_t> ret_indexes(num_results);
                        std::vector<float> out_dists_sqr(num_results);

                        nanoflann::KNNResultSet<float> resultSet(num_results);
                        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
                        index->index_->findNeighbors(resultSet, query.data());

                        for (size_t i = 0; i < resultSet.size(); i++) {
                           
                            if (ret_indexes[i] < std_local_map.size() && out_dists_sqr[i] < config_setting.kdtree_threshold_) {
                                cont_desc_pairs++;
                                // std::cout << "ret_index[" << i << "]=" << ret_indexes[i]<< " out_dist_sqr=" << out_dists_sqr[i] << std::endl;
                                // Llamar a generateArrow para crear la flecha entre descriptores

                                generateArrow(desc,std_local_map[ret_indexes[i]], marker_array, id, msg_point->header );

                                stds_prev_pair.push_back(std_local_map[ret_indexes[i]]);
                                stds_curr_pair.push_back(desc);
                            }
                            //else {
                            //     std::cerr << "Error: ret_indexes[" << i << "] está fuera de los límites de std_local_map." << std::endl;
                            // }
                        }
                    }

                    // Publicar las flechas en RViz
                    pubSTD.publish(marker_array);
                    visualization_msgs::Marker delete_marker_curr;
                    delete_marker_curr.action = visualization_msgs::Marker::DELETEALL;
                    marker_array.markers.clear();  // Asegúrate de que el array de marcadores esté vacío
                    marker_array.markers.push_back(delete_marker_curr);
                    pubSTD.publish(marker_array);
                }
                
                
                
                


            }
            std::cout<<"Pares encontrados: "<<cont_desc_pairs<<std::endl;

            // Añadir los nuevos descriptores de stds_curr a std_local_map
            std_local_map.insert(std_local_map.end(), stds_curr.begin(), stds_curr.end());
            counts_per_iteration.push_back(stds_curr.size());

            // Si el tamaño de counts_per_iteration excede max_window_size, eliminar los descriptores más antiguos
            while (counts_per_iteration.size() > config_setting.max_window_size_) {
                int count_to_remove = counts_per_iteration.front();
                counts_per_iteration.pop_front();
                // Eliminar los descriptores más antiguos de std_local_map
                for (int i = 0; i < count_to_remove; ++i) {
                    std_local_map.pop_front();
                }
            }
            updateMatrixAndKDTree(mat, index, std_local_map);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            ROS_INFO("Extracted %lu ST", stds_curr.size());
            ROS_INFO("Extracted %lu ST descriptors in %f seconds", stds_prev.size(), elapsed.count());

          //std_manager->publishAxes(marker_pub_prev, stds_prev_pair, msg_point->header);
          //std_manager->publishAxes(marker_pub_curr, stds_curr_pair, msg_point->header);

          std_manager->publishPoses(pose_pub_prev, stds_prev_pair, msg_point->header);
          std_manager->publishPoses(pose_pub_curr, stds_curr_pair, msg_point->header);


         /* ///////////////// Data visualization //////////////////////////////////////////////

            //// visualizacion de los keypoints current
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

            ///////////////////// Previous std
            ////// visualizacion de los keypoints prev
            visualization_msgs::MarkerArray marker_array_prev;
            Eigen::Vector3f colorVector_prev(1.0f, 0.0f, 0.0f);  // rojo
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
            //////////////////////////////////////////*/

            
            ////////////////////////////////////////////////////////////////////////////////////////////////
            // Actualizar stds_prev
            stds_prev = stds_curr;
            std::cout<<"Iteracion: "<<cont_itera++<<std::endl;
            
        }
    }

    return 0;
}

