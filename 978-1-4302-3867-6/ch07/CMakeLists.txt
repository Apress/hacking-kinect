cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cloud_wind)

set(FREENECT_DIR "c:/code/libfreenect")
set(FREENECT_BUILD_DIR "c:/code/libfreenect_build")
list(APPEND CMAKE_MODULE_PATH "${FREENECT_DIR}/cmake_modules/")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

#Boost
set(BOOST_ROOT "c:/code/boost_1_47_0")
set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS Date_Time)
include_directories(${Boost_INCLUDE_DIRS})

#VTK
#find_package(VTK)
#if(VTK_FOUND)
#   INCLUDE(${VTK_USE_FILE})
#endif() 

#PCL
set(PCL_DIR "C:/code/PCL1.5.1")
find_package(PCL 1.5 REQUIRED common visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#OpenCV
set(OpenCV_DIR "C:/code/OpenCV2.2/build")
find_package(OpenCV COMPONENTS filesystem)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

#OpenGL
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${USB_INCLUDE_DIRS})

#PTHREAD
set(THREADS_PTHREADS_INCLUDE_DIR "c:/code/pthreads-2-8-0/include")
set(THREADS_PTHREADS_WIN32_LIBRARY "c:/code/pthreads-2-8-0/lib/pthreadVC2.lib")
set(THREADS_USE_PTHREADS_WIN32 true)
find_package(Threads REQUIRED)
include_directories(${THREADS_PTHREADS_INCLUDE_DIR})

#USB
set(LIBUSB_1_INCLUDE_DIR "C:/code/libusb-1.2.5.0/include")
set(LIBUSB_1_LIBRARY "C:/code/libusb-1.2.5.0/lib/msvc/libusb.lib")
find_package(libusb-1.0 REQUIRED)
include_directories(${LIBUSB_1_INCLUDE_DIRS})
include_directories("${FREENECT_DIR}/platform/windows")
include_directories("${FREENECT_DIR}/platform/windows/libusb10emu")

#Freenect
include_directories("${FREENECT_DIR}/include")
#set(FREENECT_LIBRARY_DIR optimized "${FREENECT_BUILD_DIR}/lib/Release/" debug "${FREENECT_BUILD_DIR}/lib/Debug/")
set(FREENECT_LIBRARY_DIR "${FREENECT_BUILD_DIR}/lib/")
link_directories(${FREENECT_LIBRARY_DIR})
set(FREENECT_LIBRARIES "freenect.lib")

#Build
add_executable (cloud_wind cloud_wind.cpp)
target_link_libraries (cloud_wind ${FREENECT_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${LIBUSB_1_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${VTK_LIBRARIES})