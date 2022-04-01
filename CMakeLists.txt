cmake_minimum_required(VERSION 3.1)

project(test_RGBD)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    cv_bridge
    message_filters
    )

SET(OpenCV_DIR "/media/hongfeng/Storage/Code/ADynaSLAM/Thirdparty/opencv-4.5.1/build")
message("PROJECT_SOURCE_DIR: " ${OpenCV_DIR})
find_package(OpenCV 4.5.1 REQUIRED)
message(STATUS "OPENCV VERSION: " ${OpenCV_DIR})

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

catkin_package(
    INCLUDE_DIRS include
    #  LIBRARIES SLAM01
        CATKIN_DEPENDS roscpp rospy
    #  DEPENDS system_lib
    )

add_executable(test_RGBD src/RGBD.cpp)

target_link_libraries(test_RGBD ${OpenCV_LIBS} ${catkin_LIBRARIES})
