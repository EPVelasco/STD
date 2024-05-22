#include <nanoflann.hpp>
#include <deque>
#include <Eigen/Dense>

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

    // la matriz tiene 27 elementos
    Eigen::Vector3f side_length = desc.side_length_.cast<float>();
    Eigen::Vector3f angle = desc.angle_.cast<float>();
    Eigen::Vector3f center = desc.center_.cast<float>();
    Eigen::Vector3f vertex_A = desc.vertex_A_.cast<float>();
    Eigen::Vector3f vertex_B = desc.vertex_B_.cast<float>();
    Eigen::Vector3f vertex_C = desc.vertex_C_.cast<float>();
    Eigen::Matrix3d axes = desc.calculateReferenceFrame();
    Eigen::Matrix<float, 9, 1> axes_vec;
    axes_vec << axes(0),axes(1),axes(2),axes(3),axes(4),axes(5),axes(6),axes(7),axes(8);
    mat.block<1, 3>(row, 0) = side_length.transpose();
    mat.block<1, 3>(row, 3) = angle.transpose();
    mat.block<1, 3>(row, 6) = center.transpose();
    mat.block<1, 3>(row, 9) = vertex_A.transpose();
    mat.block<1, 3>(row, 12) = vertex_B.transpose();
    mat.block<1, 3>(row, 15) = vertex_C.transpose();
    mat.block<1, 9>(row, 18) = axes_vec.transpose();
}

void updateMatrixAndKDTree(Eigen::MatrixXf& mat, std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>& index, const std::deque<STDesc>& std_local_map) {
    int num_desc = std_local_map.size();
    mat.resize(num_desc, 27);

    // Rellenar la matriz con los descriptores actuales
    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }

    // Recrear el KD-Tree con la matriz actualizada
    index = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>(27, std::cref(mat), 10 /* max leaf */);
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

// DBSCAN parameters
const double EPSILON = 100.0; // Radio de búsqueda
const int MIN_POINTS = 1; // Mínimo número de puntos para formar un cluster

void DBSCAN(const Eigen::MatrixXf &data, std::vector<int> &labels) {
    using namespace nanoflann;
    const int num_points = data.rows();
    const int dim = data.cols();

    labels.assign(num_points, -1); // Inicializar etiquetas a -1 (no visitado)

    typedef KDTreeEigenMatrixAdaptor<Eigen::MatrixXf> KDTree;
    KDTree kdtree(dim, std::cref(data), 10 /* max leaf */);
    kdtree.index_->buildIndex();

    int cluster_id = 0;

    for (int i = 0; i < num_points; ++i) {
        if (labels[i] != -1) continue; // Ya visitado

        // Buscar vecinos dentro de EPSILON
        nanoflann::SearchParameters params;
        std::vector<nanoflann::ResultItem<long int, float>> ret_matches;

        const size_t nMatches = kdtree.index_->radiusSearch(data.row(i).data(), EPSILON * EPSILON, ret_matches, params);
        //std::cout << "Punto " << i << " tiene " << nMatches << " vecinos dentro del radio " << EPSILON << std::endl;


        //std::cout << "nMatches: " << nMatches;

        if (nMatches < MIN_POINTS) {
            labels[i] = -2; // Ruido
            continue;
        }

        // Asignar un nuevo ID de cluster
        labels[i] = cluster_id;
        std::deque<size_t> seeds;
        for (const auto& match : ret_matches) {
            seeds.push_back(match.first);
        }

        while (!seeds.empty()) {
            const int curr_point = seeds.front();
            seeds.pop_front();

            if (labels[curr_point] == -2) {
                labels[curr_point] = cluster_id; // Cambiar de ruido a borde
            }

            if (labels[curr_point] != -1) continue; // Ya procesado

            labels[curr_point] = cluster_id;

            ret_matches.clear();
            const size_t nMatchesInner = kdtree.index_->radiusSearch(data.row(curr_point).data(), EPSILON * EPSILON, ret_matches, params);

            if (nMatchesInner >= MIN_POINTS) {
                for (const auto& match : ret_matches) {
                    seeds.push_back(match.first);
                }
            }
        }

        ++cluster_id;
    }
    std::cout<<"Clusters: "<<cluster_id<<std::endl;
}

void updateMatrixAndKDTreeWithFiltering(Eigen::MatrixXf& mat, std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>& index, std::deque<STDesc>& std_local_map) {
    // Convertir std_local_map a una matriz Eigen
    std::cout << "Tamaño de std_local_map: " << std_local_map.size() << std::endl;

    std::cout << "Tamaño de  a mat: " << mat.size() << std::endl;

    int num_desc = std_local_map.size();
    mat.resize(num_desc, 27);

    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }
    std::cout << "Tamaño de std_local_map a mat: " << mat.size() << std::endl;

    // Aplicar DBSCAN
    std::vector<int> labels;
    DBSCAN(mat, labels);

    std::cout << "Labels: " << labels.size() << std::endl;


    // Filtrar std_local_map según los clusters identificados
    std::deque<STDesc> filtered_std_local_map;

    // Busco el maximo de labels, si es diferente de -2 es que hubo asociacion de datos si no no
    int max_label = *std::max_element(labels.begin(), labels.end());
    if(max_label<0){
        filtered_std_local_map = std_local_map;
    }
    else{
        for (int i = 0; i < labels.size(); ++i) {
            if (labels[i] >= 0) { // Si el descriptor no es ruido
                filtered_std_local_map.push_back(std_local_map[i]);
            }
        }
    }

    // Actualizar std_local_map con los descriptores filtrados
    std_local_map = std::move(filtered_std_local_map);

    // Actualizar la matriz y el KD-Tree con los descriptores filtrados
    num_desc = std_local_map.size();
    mat.resize(num_desc, 27);

    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }

    std::cout << "mat: " << mat.size() << std::endl;

    // Recrear el KD-Tree con la matriz actualizada
    index = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>(27, std::cref(mat), 10 /* max leaf */);
    index->index_->buildIndex();
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
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("output_cloud", 1);


    STDescManager *std_manager = new STDescManager(config_setting);
    std::vector<STDesc> stds_curr;
    std::vector<STDesc> stds_prev;
    

    using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    std::deque<STDesc> std_local_map;
    std::deque<int> counts_per_iteration;

    Eigen::MatrixXf mat(0, 27);
    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>> index;

    PointCloud::Ptr current_cloud_world(new PointCloud);
    PointCloud::Ptr current_cloud(new PointCloud);
    Eigen::Affine3d pose;
    Eigen::Affine3d pose_iden = Eigen::Affine3d::Identity();
    std::deque<Eigen::Affine3d> pose_vec;
    int cont_itera = 0;

    while (ros::ok()) {
        ros::spinOnce();
        std::vector<STDesc> stds_curr_pair;
        std::vector<STDesc> stds_prev_pair;
        std::vector<STDesc> stds_map; 

        if (syncPackages(current_cloud, pose)) {           
            pose_vec.push_back(pose);
            auto start = std::chrono::high_resolution_clock::now();       
            down_sampling_voxel(*current_cloud, config_setting.ds_size_);              
            
            int cont_desc_pairs = 0;
            if (init_std) {
                init_std = false;
                
                std_manager->GenerateSTDescs(current_cloud, stds_curr);
                stds_prev = stds_curr;
                stds_map = stds_curr;
                ROS_INFO("++++++++++ Iniciando Extraccion de STD ++++++++");
            } else { 
                
            //    if (pose_vec.size()>0)                
            //        pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose_vec[pose_vec.size()-1]);  
            //    else{
            //        pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose_iden);  
            //    }      
                pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose);
                std_manager->GenerateSTDescs(current_cloud_world, stds_curr);

                sensor_msgs::PointCloud2 output_cloud;
                pcl::toROSMsg(*current_cloud_world, output_cloud);
                output_cloud.header.frame_id = "velodyne";  // O el frame_id que desees
                cloud_pub.publish(output_cloud);



                if (!stds_prev.empty()) {
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
                        Eigen::Matrix3d axes_f = desc.calculateReferenceFrame();

                        query.insert(query.end(), side_length.data(), side_length.data() + 3);
                        query.insert(query.end(), angle.data(), angle.data() + 3);
                        query.insert(query.end(), center.data(), center.data() + 3);
                        query.insert(query.end(), vertex_A.data(), vertex_A.data() + 3);
                        query.insert(query.end(), vertex_B.data(), vertex_B.data() + 3);
                        query.insert(query.end(), vertex_C.data(), vertex_C.data() + 3);
                        query.insert(query.end(), axes_f.data(), axes_f.data() + axes_f.size());

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
                                generateArrow(desc, std_local_map[ret_indexes[i]], marker_array, id, msg_point->header);

                                stds_prev_pair.push_back(std_local_map[ret_indexes[i]]);
                                stds_curr_pair.push_back(desc);
                            }
                            else{
                                // elementos que no tuvieron match para ser añadidos a std_map para hacer el mapa robusto:
                                stds_map.push_back(desc);
                            }
                        }
                    }

                    // Publicar las flechas en RViz
                    pubSTD.publish(marker_array);
                    visualization_msgs::Marker delete_marker_curr;
                    delete_marker_curr.action = visualization_msgs::Marker::DELETEALL;
                    marker_array.markers.clear();
                    marker_array.markers.push_back(delete_marker_curr);
                    pubSTD.publish(marker_array);
                }
            }

            std::cout << "Pares encontrados: " << cont_desc_pairs << std::endl;

            std::cout << "Tamaño de std_local_map1 : " << std_local_map.size() << std::endl;
            std::cout << "Tamaño de stds_curr : " << stds_curr.size() << std::endl;

            // Añadir los nuevos descriptores de stds_curr a std_local_map
            std_local_map.insert(std_local_map.end(), stds_curr.begin(), stds_curr.end());
            counts_per_iteration.push_back(stds_curr.size());

            std::cout << "Tamaño de std_local_map2 : " << std_local_map.size() << std::endl;

            while (counts_per_iteration.size() > config_setting.max_window_size_) {
                std::cout << "counts_per_iteration: " << counts_per_iteration.size() <<std::endl;
                int count_to_remove = counts_per_iteration.front();
                 std::cout << "count_to_remove: " << count_to_remove <<std::endl;
                counts_per_iteration.pop_front();
                for (int i = 0; i < count_to_remove; ++i) {
               // for (int i = 0; i < std_local_map.size()/2; ++i) {    
                    std_local_map.pop_front();
                    
                }
            }

            std::cout << "Tamaño de std_local_map3: " << std_local_map.size() << std::endl;
            

            // Actualizar la matriz y el KD-Tree con filtrado DBSCAN
            //updateMatrixAndKDTreeWithFiltering(mat, index, std_local_map);
            updateMatrixAndKDTree(mat, index, std_local_map);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            //ROS_INFO("Extracted %lu ST", stds_curr.size());
            ROS_INFO("Extracted %lu ST descriptors in %f seconds", stds_curr.size(), elapsed.count());
            

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

