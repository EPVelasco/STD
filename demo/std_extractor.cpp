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
int current_frame_id_=0;

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

void convertToMarkers(const std::vector<STDesc>& stds, visualization_msgs::MarkerArray& marker_array, const Eigen::Vector3f& color, float alpha = 1.0, float scale = 0.03) {
    int id = 0;

    for (const auto& std : stds) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "std_descriptors";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = scale;
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

void MAPconvertToMarkers(const Eigen::MatrixXf& data, visualization_msgs::MarkerArray& marker_array, const Eigen::Vector3f& color, float alpha = 1.0, float scale = 0.03) {
    int id = 0;

    for (int i = 0; i < data.rows(); ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "std_descriptors";
        marker.id = id++;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = scale;
        marker.color.r = color(0);  
        marker.color.g = color(1);  
        marker.color.b = color(2);  
        marker.color.a = alpha;     

        geometry_msgs::Point p1, p2, p3;
        p1.x = data(i,9);
        p1.y = data(i,10);
        p1.z = data(i,11);
        p2.x = data(i,12);
        p2.y = data(i,13);
        p2.z = data(i,14);
        p3.x = data(i,15);
        p3.y = data(i,16);
        p3.z = data(i,17);

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

void MAPconvertToPointCloud(const Eigen::MatrixXf& data, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    for (int i = 0; i < data.rows(); ++i) {
        pcl::PointXYZ p1, p2, p3;

        // Asumiendo que los vertices estan en las posiciones 9-17
        p1.x = data(i, 9);
        p1.y = data(i, 10);
        p1.z = data(i, 11);
        
        p2.x = data(i, 12);
        p2.y = data(i, 13);
        p2.z = data(i, 14);

        p3.x = data(i, 15);
        p3.y = data(i, 16);
        p3.z = data(i, 17);

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
    //std::cout << "Norms : " << desc.norms.transpose() << std::endl;
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

    // la matriz tiene 36 elementos
    Eigen::Vector3f side_length = desc.side_length_.cast<float>();
    Eigen::Vector3f angle = desc.angle_.cast<float>();
    Eigen::Vector3f center = desc.center_.cast<float>();
    Eigen::Vector3f vertex_A = desc.vertex_A_.cast<float>();
    Eigen::Vector3f vertex_B = desc.vertex_B_.cast<float>();
    Eigen::Vector3f vertex_C = desc.vertex_C_.cast<float>();
    Eigen::Vector3f normal1 = desc.normal1_.cast<float>();
    Eigen::Vector3f normal2 = desc.normal2_.cast<float>();
    Eigen::Vector3f normal3 = desc.normal3_.cast<float>();
    Eigen::Matrix3d axes = desc.calculateReferenceFrame();
    Eigen::Matrix<float, 9, 1> axes_vec;
    axes_vec << axes(0),axes(1),axes(2),axes(3),axes(4),axes(5),axes(6),axes(7),axes(8);
    mat.block<1, 3>(row, 0) = side_length.transpose();
    mat.block<1, 3>(row, 3) = angle.transpose();
    mat.block<1, 3>(row, 6) = center.transpose();
    mat.block<1, 3>(row, 9) = vertex_A.transpose();
    mat.block<1, 3>(row, 12) = vertex_B.transpose();
    mat.block<1, 3>(row, 15) = vertex_C.transpose();
    mat.block<1, 3>(row, 18) = normal1.transpose();
    mat.block<1, 3>(row, 21) = normal2.transpose();
    mat.block<1, 3>(row, 24) = normal3.transpose();
    mat.block<1, 9>(row, 27) = axes_vec.transpose();
}

void updateMatrixAndKDTree(Eigen::MatrixXf& mat, std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>& index, const std::deque<STDesc>& std_local_map) {
    int num_desc = std_local_map.size();
    mat.resize(num_desc, 36);

    // Rellenar la matriz con los descriptores actuales
    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }

    // Recrear el KD-Tree con la matriz actualizada
    index = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>(36, std::cref(mat), 10 /* max leaf */);
    index->index_->buildIndex();
}

void publishLocalMap(const std::deque<STDesc>& std_local_map, visualization_msgs::MarkerArray& marker_array, const Eigen::Vector3f& color, float alpha = 1.0) {
    std::vector<STDesc> temp_vector;
    temp_vector.reserve(std_local_map.size());
    for (const auto& desc : std_local_map) {
        temp_vector.push_back(desc);
    }
    // std::cout << "publishLocalMap**********: " << std_local_map.size() << std::endl;
    // std::cout << "temp_vector " << temp_vector.size() << std::endl;


    
    convertToMarkers(temp_vector, marker_array, color, alpha,0.06);
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
    arrow.header.frame_id = "map";
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
//const double EPSILON = 10.0; // Radio de búsqueda
const int MIN_POINTS = 1; // Mínimo número de puntos para formar un cluster

void DBSCAN(const Eigen::MatrixXf &data, std::vector<int> &labels, const double EPSILON) {
    using namespace nanoflann;
    const int num_points = data.rows();

    // // Utilizar los vértices de los std para filtrar el mapa (están en la posición 6-8 y 18-26)
    // Eigen::MatrixXf data_vertex(num_points, 12); // 3 columnas para el centro y 9 para los axes

    // // Copiar los datos correspondientes
    // data_vertex << data.block(0, 6, num_points, 3),   // Datos del centro del triángulo (columnas 6-8)
    //                data.block(0, 18, num_points, 9);  // Datos de los axes del triángulo (columnas 18-26)

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
        std::cout << "Punto " << i << " tiene " << nMatches << " vecinos dentro del radio " << EPSILON << std::endl;

        // Filtrar el punto actual de los vecinos
        std::vector<size_t> valid_neighbors;
        for (const auto& match : ret_matches) {
            if (match.first != i) {
                valid_neighbors.push_back(match.first);
            }
        }

        if (valid_neighbors.size() < MIN_POINTS) {
            labels[i] = -2; // Ruido
            continue;
        }

        // Asignar un nuevo ID de cluster
        labels[i] = cluster_id;
        std::deque<size_t> seeds(valid_neighbors.begin(), valid_neighbors.end());

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

            std::vector<size_t> valid_neighbors_inner;
            for (const auto& match : ret_matches) {
                if (match.first != curr_point) {
                    valid_neighbors_inner.push_back(match.first);
                }
            }

            if (valid_neighbors_inner.size() >= MIN_POINTS) {
                for (const auto& match : valid_neighbors_inner) {
                    seeds.push_back(match);
                }
            }
        }

        ++cluster_id;
    }

    std::cout << "Clusters: " << cluster_id << std::endl;
    for (const auto& label_str : labels) {
        if (label_str != -2)
            std::cout << "label_str: " << label_str << std::endl;
    }
}

// Función para calcular la distancia euclidiana entre dos vértices
float calcularDistancia(const Eigen::Vector3f &v1, const Eigen::Vector3f &v2) {
    return (v1 - v2).norm();
}



 
void extractVerticesToMatrix(const std::deque<STDesc>& std_local_map, Eigen::MatrixXf& all_vertices) {
    const int num_desc = std_local_map.size();
    all_vertices.resize(3 * num_desc, 3); // 3 filas por descriptor, cada una con 3 coordenadas

    for (size_t i = 0; i < num_desc; ++i) {
        all_vertices.row(3 * i) = std_local_map[i].vertex_A_.transpose().cast<float>();   // vertex_A
        all_vertices.row(3 * i + 1) = std_local_map[i].vertex_B_.transpose().cast<float>(); // vertex_B
        all_vertices.row(3 * i + 2) = std_local_map[i].vertex_C_.transpose().cast<float>(); // vertex_C
    }
}

void build_stdesc_EPVS(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
    std::vector<STDesc> &stds_vec, ConfigSetting config_setting_) {
  stds_vec.clear();
  double scale = 1.0 / config_setting_.std_side_resolution_;
  int near_num = 1;
  double max_dis_threshold = 100000;
  double min_dis_threshold = 0 ;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
  kd_tree->setInputCloud(corner_points);
  std::vector<int> pointIdxNKNSearch(near_num);
  std::vector<float> pointNKNSquaredDistance(near_num);
  // Search N nearest corner points to form stds.
  for (size_t i = 0; i < corner_points->size(); i++) {
    pcl::PointXYZINormal searchPoint = corner_points->points[i];
    if (kd_tree->nearestKSearch(searchPoint, near_num, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < near_num - 1; m++) {
        for (int n = m + 1; n < near_num; n++) {
          pcl::PointXYZINormal p1 = searchPoint;
          pcl::PointXYZINormal p2 = corner_points->points[pointIdxNKNSearch[m]];
          pcl::PointXYZINormal p3 = corner_points->points[pointIdxNKNSearch[n]];
          Eigen::Vector3d normal_inc1(p1.normal_x - p2.normal_x,
                                      p1.normal_y - p2.normal_y,
                                      p1.normal_z - p2.normal_z);
          Eigen::Vector3d normal_inc2(p3.normal_x - p2.normal_x,
                                      p3.normal_y - p2.normal_y,
                                      p3.normal_z - p2.normal_z);
          Eigen::Vector3d normal_add1(p1.normal_x + p2.normal_x,
                                      p1.normal_y + p2.normal_y,
                                      p1.normal_z + p2.normal_z);
          Eigen::Vector3d normal_add2(p3.normal_x + p2.normal_x,
                                      p3.normal_y + p2.normal_y,
                                      p3.normal_z + p2.normal_z);
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > max_dis_threshold || b > max_dis_threshold ||
              c > max_dis_threshold || a < min_dis_threshold ||
              b < min_dis_threshold || c < min_dis_threshold) {
            continue;
          }
          // re-range the vertex by the side length
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          // check augnmentation
          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          if (iter == feat_map.end()) {
            Eigen::Vector3d vertex_attached;
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              normal_1 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[0] = p1.intensity;
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              normal_1 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[0] = p2.intensity;
            } else {
              A << p3.x, p3.y, p3.z;
              normal_1 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[0] = p3.intensity;
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              normal_2 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[1] = p1.intensity;
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              normal_2 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[1] = p2.intensity;
            } else {
              B << p3.x, p3.y, p3.z;
              normal_2 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[1] = p3.intensity;
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              normal_3 << p1.normal_x, p1.normal_y, p1.normal_z;
              vertex_attached[2] = p1.intensity;
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              normal_3 << p2.normal_x, p2.normal_y, p2.normal_z;
              vertex_attached[2] = p2.intensity;
            } else {
              C << p3.x, p3.y, p3.z;
              normal_3 << p3.normal_x, p3.normal_y, p3.normal_z;
              vertex_attached[2] = p3.intensity;
            }
            STDesc single_descriptor;
            current_frame_id_++;
            single_descriptor.vertex_A_ = A;
            single_descriptor.vertex_B_ = B;
            single_descriptor.vertex_C_ = C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.vertex_attached_ = vertex_attached;
            single_descriptor.side_length_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.normal1_ = normal_1;
            single_descriptor.normal2_ = normal_2;
            single_descriptor.normal3_ = normal_3;
            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_id_ = current_frame_id_;
            Eigen::Matrix3d triangle_positon;
            feat_map[position] = true;
            stds_vec.push_back(single_descriptor);
          }
        }
      }
    }
  }
};

// Función para verificar y agrupar vértices dentro de un radio
void verifyvertx(const Eigen::MatrixXf &vertices, std::vector<int> &labels, const float EPSILON) {
    const int num_points = vertices.rows();
    labels.assign(num_points, -1); // Inicializar etiquetas a -1 (no visitado)
    int current_label = 0;

    for (int i = 0; i < num_points; ++i) {
        if (labels[i] != -1) continue; // Si ya está etiquetado, continuar al siguiente

        // Etiquetar el punto actual con una nueva etiqueta de cluster
        labels[i] = current_label;

        // Recorrer todos los puntos desde el siguiente al actual para encontrar vecinos
        for (int j = i + 1; j < num_points; ++j) {
            if (calcularDistancia(vertices.row(i).transpose(), vertices.row(j).transpose()) <= EPSILON) {
                labels[j] = current_label; // Etiquetar el punto vecino con la misma etiqueta de cluster
            }
        }
        current_label++;
    }
}
// Función para promediar los vértices agrupados por labels
Eigen::MatrixXf promediarVertices(const Eigen::MatrixXf &vertices, const std::vector<int> &labels) {
    std::map<int, Eigen::Vector3f> sum_vertices;
    std::map<int, int> count_vertices;

    // Sumarizar los vértices por label
    for (int i = 0; i < vertices.rows(); ++i) {
        int label = labels[i];
        if (label >= 0) {
            if (sum_vertices.find(label) == sum_vertices.end()) {
                sum_vertices[label] = Eigen::Vector3f::Zero();
                count_vertices[label] = 0;
            }
            sum_vertices[label] += vertices.row(i).transpose();
            count_vertices[label] += 1;
        }
    }

    // Crear una nueva matriz de vértices con los promedios
    Eigen::MatrixXf new_vertices(vertices.rows(), vertices.cols());

    for (int i = 0; i < vertices.rows(); ++i) {
        int label = labels[i];
        if (label >= 0 && count_vertices[label] > 0) {
            new_vertices.row(i) = (sum_vertices[label] / count_vertices[label]).transpose();
        } else {
            new_vertices.row(i) = vertices.row(i);
        }
    }

    return new_vertices;
}

void updateMatrixAndKDTreeWithFiltering(Eigen::MatrixXf& mat, std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>& index, std::deque<STDesc>& std_local_map,  ConfigSetting config_setting) {
    std::cout << "Tamaño de std_local_map: " << std_local_map.size() << std::endl;
    std::cout << "Tamaño de  a mat: " << mat.size()/36 << std::endl;

    int num_desc = std_local_map.size();
    mat.resize(num_desc, 36);

    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }
    std::cout << "Tamaño de std_local_map a mat: " << mat.size()/36 << std::endl;

 

    /////////////////// Filtrado de vértices
    Eigen::MatrixXf all_vertices;
    extractVerticesToMatrix(std_local_map, all_vertices);

    // Aplicar DBSCAN a todos los vértices
    std::vector<int> vertex_labels;
    std::cout << "Vertices antes de la agrupación y el promedio:" << std::endl;
    std::cout << all_vertices << std::endl;
    verifyvertx(all_vertices, vertex_labels, config_setting.epsilon_);
    std::cout << "Vertices después de la agrupación y el promedio:" << std::endl;
    std::cout << all_vertices << std::endl;

     std::cout << "Labels después de la agrupación:" << std::endl;
    for (int label : vertex_labels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    Eigen::MatrixXf new_vertices = promediarVertices(all_vertices, vertex_labels);


    // Reconstruir std_local_map con los vértices fusionados
    std::deque<STDesc> filtered_std_local_map;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_points(new pcl::PointCloud<pcl::PointXYZINormal>);

    for (size_t i = 0; i < std_local_map.size(); ++i) {
        pcl::PointXYZINormal p1, p2, p3;

        Eigen::Vector3d vertex_A = new_vertices.row(3 * i).cast<double>();
        Eigen::Vector3d vertex_B = new_vertices.row(3 * i + 1).cast<double>();
        Eigen::Vector3d vertex_C = new_vertices.row(3 * i + 2).cast<double>();

        // Supongamos que las normales originales están en std_local_map
        p1.x = vertex_A[0]; p1.y = vertex_A[1]; p1.z = vertex_A[2];
        p1.normal_x = std_local_map[i].normal1_[0]; p1.normal_y = std_local_map[i].normal1_[1]; p1.normal_z = std_local_map[i].normal1_[2];
        p1.intensity = std_local_map[i].vertex_attached_[0];

        p2.x = vertex_B[0]; p2.y = vertex_B[1]; p2.z = vertex_B[2];
        p2.normal_x = std_local_map[i].normal2_[0]; p2.normal_y = std_local_map[i].normal2_[1]; p2.normal_z = std_local_map[i].normal2_[2];
        p2.intensity = std_local_map[i].vertex_attached_[1];

        p3.x = vertex_C[0]; p3.y = vertex_C[1]; p3.z = vertex_C[2];
        p3.normal_x = std_local_map[i].normal3_[0]; p3.normal_y = std_local_map[i].normal3_[1]; p3.normal_z = std_local_map[i].normal3_[2];
        p3.intensity = std_local_map[i].vertex_attached_[2];

        corner_points->points.push_back(p1);
        corner_points->points.push_back(p2);
        corner_points->points.push_back(p3);
    }


    //Usar corner_points para construir stds_vec
    std::vector<STDesc> stds_vec;
    build_stdesc_EPVS(corner_points, stds_vec, config_setting);
    filtered_std_local_map.assign(stds_vec.begin(), stds_vec.end());

    std::cout << "Antes : " << std_local_map.size() << std::endl;

    // Actualizar std_local_map con los descriptores filtrados y fusionados
    std_local_map = std::move(filtered_std_local_map);

    std::cout << "despues : " << std_local_map.size() << std::endl;

    // Actualizar la matriz y el KD-Tree con los descriptores filtrados y fusionados
    num_desc = std_local_map.size();
    mat.resize(num_desc, 36);

    for (size_t i = 0; i < std_local_map.size(); ++i) {
        addDescriptorToMatrix(mat, std_local_map[i], i);
    }

    std::cout << "mat: " << mat.size() / 36 << std::endl;

    // Recrear el KD-Tree con la matriz actualizada
    index = std::make_unique<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>>(36, std::cref(mat), 10 /* max leaf */);
    index->index_->buildIndex();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "STD_descriptor");
    ros::NodeHandle nh;

    ConfigSetting config_setting;
    read_parameters(nh, config_setting);

    ros::Publisher pubkeycurr = nh.advertise<visualization_msgs::MarkerArray>("std_curr", 10);
    ros::Publisher pubkeyprev = nh.advertise<visualization_msgs::MarkerArray>("std_prev", 10);    
    ros::Publisher pubkeymap = nh.advertise<visualization_msgs::MarkerArray>("std_map", 10); 
    ros::Publisher pubkeymap_filter = nh.advertise<visualization_msgs::MarkerArray>("std_map_filter", 10);  
    
    ros::Publisher pub_curr_points = nh.advertise<sensor_msgs::PointCloud2>("std_curr_points", 10);
    ros::Publisher pub_prev_points = nh.advertise<sensor_msgs::PointCloud2>("std_prev_points", 10);
    ros::Publisher pub_map_points = nh.advertise<sensor_msgs::PointCloud2>("std_map_points", 10);
    
    ros::Publisher pubSTD = nh.advertise<visualization_msgs::MarkerArray>("pair_std", 10);
    ros::Publisher marker_pub_prev = nh.advertise<visualization_msgs::MarkerArray>("Axes_prev_STD", 10);
    ros::Publisher marker_pub_curr = nh.advertise<visualization_msgs::MarkerArray>("Axes_curr_STD", 10);
    ros::Publisher pose_pub_prev = nh.advertise<geometry_msgs::PoseArray>("std_prev_poses", 10);
    ros::Publisher pose_pub_curr = nh.advertise<geometry_msgs::PoseArray>("std_curr_poses", 10);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/odom", 100, OdomHandler);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("output_cloud", 1);
    ros::Publisher cloud_pub_prev = nh.advertise<sensor_msgs::PointCloud2>("output_cloud_prev", 1);


    STDescManager *std_manager = new STDescManager(config_setting);
    std::vector<STDesc> stds_curr;
    std::vector<STDesc> stds_prev;
    

    using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
    std::deque<STDesc> std_local_map;
    std::deque<int> counts_per_iteration;

    Eigen::MatrixXf mat(0, 36);
    std::unique_ptr<nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>> index;

    PointCloud::Ptr current_cloud_world(new PointCloud);
    PointCloud::Ptr current_cloud_diff(new PointCloud);    
    PointCloud::Ptr current_cloud(new PointCloud);
    Eigen::Affine3d pose;
    Eigen::Affine3d pose_prev = Eigen::Affine3d::Identity();
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
                *current_cloud_world = *current_cloud;
                stds_prev = stds_curr;
                stds_map = stds_curr;
                 
                ROS_INFO("++++++++++ Iniciando Extraccion de STD ++++++++");
            } else { 
                
            //    if (pose_vec.size()>0)                
            //        pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose_vec[pose_vec.size()-1]);  
            //    else{
            //        pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose_iden);  
            //    }      

                // Eigen::Affine3d pose_diff = (pose_prev.inverse() * pose);

                pcl::transformPointCloud(*current_cloud, *current_cloud_world, pose);
                // pcl::transformPointCloud(*current_cloud, *current_cloud_diff, pose_diff);
                std_manager->GenerateSTDescs(current_cloud_world, stds_curr);


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
                        Eigen::Vector3f norms1 = desc.normal1_.cast<float>();
                        Eigen::Vector3f norms2 = desc.normal2_.cast<float>();
                        Eigen::Vector3f norms3 = desc.normal3_.cast<float>();
                        Eigen::Matrix3d axes_f = desc.calculateReferenceFrame();


                        query.insert(query.end(), side_length.data(), side_length.data() + 3);
                        query.insert(query.end(), angle.data(), angle.data() + 3);
                        query.insert(query.end(), center.data(), center.data() + 3);
                        query.insert(query.end(), vertex_A.data(), vertex_A.data() + 3);
                        query.insert(query.end(), vertex_B.data(), vertex_B.data() + 3);
                        query.insert(query.end(), vertex_C.data(), vertex_C.data() + 3);
                        query.insert(query.end(), norms1.data(), norms1.data() + 3);
                        query.insert(query.end(), norms2.data(), norms2.data() + 3);
                        query.insert(query.end(), norms3.data(), norms3.data() + 3);
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
            // //////////// nube de puntos anterior
            // pcl::transformPointCloud(*current_cloud, *current_cloud_world_prev, pose_prev);
            // sensor_msgs::PointCloud2 output_cloud;
            // pcl::toROSMsg(*current_cloud_world_prev, output_cloud);
            // output_cloud.header.frame_id = "velodyne";  // O el frame_id que desees
            // cloud_pub_prev.publish(output_cloud);}

            sensor_msgs::PointCloud2 output_cloud;
            pcl::toROSMsg(*current_cloud_world, output_cloud);
            output_cloud.header.frame_id = "map";  // O el frame_id que desees
            cloud_pub.publish(output_cloud);
            



          ///////////////// Data visualization //////////////////////////////////////////////

            //// visualizacion de los keypoints current
            visualization_msgs::MarkerArray marker_array_curr;
            Eigen::Vector3f colorVector_curr(0.0f, 0.0f, 1.0f);  // azul

            convertToMarkers(stds_curr, marker_array_curr,colorVector_curr ,0.5);
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
            output_curr.header.frame_id = "map";
            pub_curr_points.publish(output_curr);
            //////////////////////////////////////////

            

            ///////////////////// Previous std
            ////// visualizacion de los keypoints prev
            visualization_msgs::MarkerArray marker_array_prev;
            Eigen::Vector3f colorVector_prev(1.0f, 0.0f, 0.0f);  // rojo
            convertToMarkers(stds_prev, marker_array_prev,colorVector_prev ,0.5);
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
            output_prev.header.frame_id = "map";
            pub_prev_points.publish(output_prev);
            //////////////////////////////////////////

            /////////////// plot stds_map
            visualization_msgs::MarkerArray marker_array_map;
            Eigen::Vector3f colorVector_map(0.0f, 0.0f, 0.0f);  // negro
            publishLocalMap(std_local_map, marker_array_map,colorVector_map ,0.2);
            pubkeymap.publish(marker_array_map);
            // visualization_msgs::Marker delete_marker_map;
            // delete_marker_map.action = visualization_msgs::Marker::DELETEALL;
            // marker_array_map.markers.clear();  // Asegúrate de que el array de marcadores esté vacío
            // marker_array_map.markers.push_back(delete_marker_map);
            // pubkeymap.publish(marker_array_map);
            

            
            ////////////////////////////////////////////////////////////////////////////////////////////////

             std::cout << "Pares encontrados: " << cont_desc_pairs << std::endl;
            // std::cout << "Tamaño de std_local_map1 : " << std_local_map.size() << std::endl;
            // std::cout << "Tamaño de stds_curr : " << stds_curr.size() << std::endl;

            // Añadir los nuevos descriptores de stds_curr a std_local_map
            std_local_map.insert(std_local_map.end(), stds_curr.begin(), stds_curr.end());

            std::cout << "Tamaño de std_local_map3: " << std_local_map.size() << std::endl;
            

        

            counts_per_iteration.push_back(std_local_map.size());

            // std::cout << "Tamaño de std_local_map2 : " << std_local_map.size() << std::endl;

            while (counts_per_iteration.size() > config_setting.max_window_size_) {
                std::cout << "counts_per_iteration: " << counts_per_iteration.size() <<std::endl;
                int count_to_remove = counts_per_iteration.front();
                 std::cout << "count_to_remove: " << count_to_remove <<std::endl;
                counts_per_iteration.pop_front();
                //for (int i = 0; i < count_to_remove; ++i) {
                for (int i = 0; i < std_local_map.size()/2; ++i) {    
                    std_local_map.pop_front();
                    
                }
            }

            // Actualizar la matriz y el KD-Tree con filtrado DBSCAN
            updateMatrixAndKDTreeWithFiltering(mat, index, std_local_map, config_setting);
            //updateMatrixAndKDTree(mat, index, std_local_map);



            ////// publicacion de nube de puntos en los vertices de los stds del MAPA filtrado
            pcl::PointCloud<pcl::PointXYZ>::Ptr std_map_pcl(new pcl::PointCloud<pcl::PointXYZ>);
            MAPconvertToPointCloud(mat, std_map_pcl);
            sensor_msgs::PointCloud2 output_map_point;
            pcl::toROSMsg(*std_map_pcl, output_map_point);
            output_map_point.header.frame_id = "map";
            pub_map_points.publish(output_map_point);

            ////// visualizacion de los triangulos del mapa filtrado
            visualization_msgs::MarkerArray marker_map_filter;
            Eigen::Vector3f colorVector_map_fil(1.0f, 0.0f, 1.0f);  // rojo
            MAPconvertToMarkers(mat, marker_map_filter,colorVector_map_fil ,0.5);
            pubkeymap_filter.publish(marker_map_filter);
            visualization_msgs::Marker delete_map_filter;
            delete_map_filter.action = visualization_msgs::Marker::DELETEALL;
            marker_map_filter.markers.clear();  // Asegúrate de que el array de marcadores esté vacío
            marker_map_filter.markers.push_back(delete_map_filter);
            pubkeymap_filter.publish(marker_map_filter);
            //////////////////////////////////////////


            //////////////////////////////////////////



            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            //ROS_INFO("Extracted %lu ST", stds_curr.size());
            ROS_INFO("Extracted %lu ST descriptors in %f seconds", stds_curr.size(), elapsed.count());
            

            std_manager->publishPoses(pose_pub_prev, stds_prev_pair, msg_point->header);
            std_manager->publishPoses(pose_pub_curr, stds_curr_pair, msg_point->header);

            // Actualizar stds_prev
            stds_prev = stds_curr;
            pose_prev = pose;
            std::cout<<"Iteracion: "<<cont_itera++<<std::endl;
            
        }
    }

    return 0;
}

