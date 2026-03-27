cmake_minimum_required(VERSION 3.16)
project(qr_localizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(qr_localizer
    main.cpp
    src/qr_localize.cpp
)

target_link_libraries(qr_localizer PRIVATE ${OpenCV_LIBS})
target_link_libraries(qr_localizer PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(qr_localizer PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
