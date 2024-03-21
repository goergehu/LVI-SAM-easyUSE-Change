#include "utility.h"
#include "lvi_sam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subOdometry;
    std::ofstream out_file;  // hcc

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lvi_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    std::vector<std::pair<double, double>> time_yaw;
    std::mutex time_bf;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 5, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>(PROJECT_NAME + "/lidar/mapping/odometry", 5, &FeatureExtraction::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info> (PROJECT_NAME + "/lidar/feature/cloud_info", 5);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 5);
        
        initializationValue();

        // hcc:add
        std::string file_path = "/home/hcc/ws_LVI_SAM_easyUse/yaw_data.txt";
        out_file.open(file_path);
        if (out_file.is_open())
            ROS_INFO_STREAM(file_path << " is opened.");
        else {
            ROS_ERROR("Failed to open the file. Exiting...");
            ROS_BREAK();
        }
        // hcc:add
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        time_bf.lock();
        
        nav_msgs::Odometry pose_temp = *odomMsg;
        Eigen::Quaternionf q(pose_temp.pose.pose.orientation.w, pose_temp.pose.pose.orientation.x, pose_temp.pose.pose.orientation.y, pose_temp.pose.pose.orientation.z);
        Eigen::Matrix3f rotationMatrix = q.toRotationMatrix();
        // out_file << std::fixed << std::setprecision(9) << rotationMatrix(0,0) << " "<< rotationMatrix(0,1) << " "<< rotationMatrix(0,2) << " " << pose_temp.pose.pose.position.x
        // << " " << rotationMatrix(1,0) << " "<< rotationMatrix(1,1) << " "<< rotationMatrix(1,2) << " " << pose_temp.pose.pose.position.y
        // << " " << rotationMatrix(2,0) << " "<< rotationMatrix(2,1) << " "<< rotationMatrix(2,2) << " " << pose_temp.pose.pose.position.z << std::endl;
        // Eigen::Vector3f euler_angles =  rotationMatrix.eulerAngles(0, 1, 2);
        double yaw_angle = -atan2(rotationMatrix(1, 0),rotationMatrix(0, 0))*180.0/M_PI;
        // double yaw_temp = euler_angles[2]*180.0/M_PI;
        double time_temp = odomMsg->header.stamp.toSec();
        time_yaw.push_back(std::make_pair(time_temp, yaw_angle));
        // out_file << std::fixed << std::setprecision(9) << time_temp << " " << yaw_temp << " " << yaw_angle << endl;
        time_bf.unlock();
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        
        // hcc:begin
        time_bf.lock();
        double curtime = cloudHeader.stamp.toSec();
        bool no_average_flag = false;
        double cur_yaw, cur_yaw_another;
        int size_time = time_yaw.size();
        if (size_time>=2)
        {

            no_average_flag = true;
            cur_yaw = time_yaw[size_time-1].second - time_yaw[size_time-2].second + 90.0;
            if(cur_yaw > 0)
                cur_yaw_another = cur_yaw - 180.0;
            else
                cur_yaw_another = cur_yaw + 180.0;
        }
        time_bf.unlock();
        // hcc:end

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        int split = 12; // hcc : split number, change this

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            for (int j = 0; j < split; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (split - j) + cloudInfo.endRingIndex[i] * j) / split;
                int ep = (cloudInfo.startRingIndex[i] * (split - 1 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / split - 1;
                //hcc:begin
                int begin_ind = cloudSmoothness[sp].ind;
                int end_ind = cloudSmoothness[ep].ind;
                int init_num = calculatePoints(begin_ind, end_ind);
                /*
                int init_num = 10;
                
                int begin_ind = cloudSmoothness[sp].ind;
                int end_ind = cloudSmoothness[ep].ind;
                
                PointType begin_Point = extractedCloud->points[begin_ind];
                PointType end_Point =  extractedCloud->points[end_ind];

                float begin_yaw = atan2(begin_Point.y, begin_Point.x) * 180.0 / M_PI;
                float end_yaw = atan2(end_Point.y, end_Point.x) * 180.0 / M_PI;
                float diff_yaw = abs(begin_yaw - end_yaw);
                float min_yaw, max_yaw;
                min_yaw = min(begin_yaw, end_yaw);
                max_yaw = max(begin_yaw, end_yaw);
                if (diff_yaw > 180.0)
                {
                    min_yaw += 180.0;
                    float temp = max_yaw;
                    max_yaw = min_yaw;
                    min_yaw = temp - 180.0; 
                }
                if (no_average_flag)
                {
                    if((min_yaw < cur_yaw && max_yaw > cur_yaw) || (min_yaw < cur_yaw_another && max_yaw > cur_yaw_another))
                    {
                        init_num = 15;
                    }
                    // out_file << fixed << setprecision(6) << "cur_yaw, begin_yaw, end_yaw" << cur_yaw << " " << begin_yaw << " " << end_yaw << endl;
                }
                */
                // hcc:end

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= init_num){
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.startRingIndex.shrink_to_fit();
        cloudInfo.endRingIndex.clear();
        cloudInfo.endRingIndex.shrink_to_fit();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointColInd.shrink_to_fit();
        cloudInfo.pointRange.clear();
        cloudInfo.pointRange.shrink_to_fit();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }

    int calculatePoints(int sp, int ep)
    {
        map<int, int> map_temp;
        PointType first_point = extractedCloud->points[sp];
        Eigen::Vector3d vec_a(first_point.x, first_point.y, first_point.z);
        for(int i=sp+1;i<ep;i++)
        {
            PointType cur_point = extractedCloud->points[i];
            Eigen::Vector3d vec_b(cur_point.x-first_point.x, cur_point.y-first_point.y, cur_point.z-first_point.z);
            
            double dot_product = vec_a.dot(vec_b);

            double norm_a = vec_a.norm();
            double norm_b = vec_b.norm();

            double angle_rad = acos(dot_product / (norm_a * norm_b));

            double angle_deg = angle_rad * 180.0 / M_PI;
            int angle_dif = static_cast<int>(angle_deg / 5);
            if(map_temp.find(angle_dif)==map_temp.end())
                map_temp[angle_dif] = 1;
        }
        return min(max(static_cast<int>(map_temp.size())-1, 0), 20);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");
   
    ros::spin();
    FE.out_file.close();

    return 0;
}