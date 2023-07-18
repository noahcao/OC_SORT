# OC-SORT C++ Library Tracker 
## Example with CMake

Assuming the file directory is as follows:

```
├───include
├───src
├───multithread.cpp
└───CMakeLists.txt
```

## Compile with Cmake

You can use CMAKE GUI to configure and set the paths to the OpenCV and Eigen3 libraries on your machine. Here, I'm assuming you are using a Windows system and planning to update for Linux in the future. Bạn có thể dịch sang tiếng anh Và tôi sử dụng một inference từ YOLOv8 để có thể xác thực khả năng theo dõi đối tượng của OC SORT từ video thay vì tập dữ liệu MOT  bạn có thể tham khảo tại `mutilthread.cpp`
`CMakeLists.txt` Content:

```cmake
cmake_minimum_required(VERSION 3.10)
project(libocsort)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /O2") 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2")

set(CMAKE_CXX_STANDARD 17)

# Linker external library
SET(Eigen3_DIR "C:/eigen-3.4.0/build")
find_package(Eigen3 REQUIRED)
set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

# add_subdirectory(src)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

# note：test with yolo inference
add_executable(test mutilthread.cpp <inference_header> <inference_source>)
target_link_libraries(test PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS})

# note：test with MOT
# add_executable(test testMOT.cpp)
# target_link_libraries(test PUBLIC Eigen3::Eigen ${PROJECT_NAME} ${OpenCV_LIBS})
```

After a successful compilation without any issues, you will see some information about the library documentation and execution documentation listed below. 

Download `model.onnx` & `class.txt` video for verify [yolov8s](https://drive.google.com/drive/folders/1ke7xyawZ8N1sIXh4WGXMvv6AAfAoSR3v?usp=sharing)

```cmd
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         6/30/2023  12:58 PM         423936 libocsort.dll
-a----         6/30/2023  12:58 PM         434904 libocsort.exp
-a----         6/30/2023   1:52 AM         723294 libocsort.lib
-a----         6/30/2023  12:58 PM          66048 test.exe
```

Result show:
![result](ocsort-inference.png)

# About input-output formats
There are slight differences between the input-output formats of the modified version of OCSORT and the original version:

## Input format
Type of input: `Eigen::Matrix<double,Eigen::Dynamic,6>`

Format: `<x1>,<y1>,<x2>,<y2>,<confidence>,<class>`

## Output format
Type of output: `Eigen::Matrix<double,Eigen::Dynamic>`

Format: `<x1>,<y1>,<x2>,<y2>,<ID>,<class>,<confidence>`
This modification is done to facilitate the integration of OCSORT with other object detectors to form a complete object tracking pipeline.

## Example with C++:

### Vector2Matrix

The code you provided defines a function named `Vector2Matrix` that converts a 2D `std::vector` of float data into an `Eigen::Matrix<float, Eigen::Dynamic, 6>`.

Here's a breakdown of the code:

```cpp
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    // Create an Eigen::Matrix with the same number of rows as the data and 6 columns
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());

    // Iterate over the rows and columns of the data vector
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            // Assign the value at position (i, j) of the matrix to the corresponding value from the data vector
            matrix(i, j) = data[i][j];
        }
    }

    // Return the resulting matrix
    return matrix;
}

```

The function takes a 2D `std::vector` of float data as input, represented by the data parameter. It creates an `Eigen::Matrix<float, Eigen::Dynamic, 6>` named matrix with the same number of rows as the data vector (`data.size()`) and 6 columns.

Next, it uses nested for loops to iterate over the rows and columns of the data vector. It assigns the value at position `(i, j)` of the matrix to the corresponding value from the data vector (`data[i][j]`).

After iterating over all the elements of the data vector and assigning them to the matrix, the function returns the resulting matrix.

Note that this code assumes that the input data vector is non-empty and that all rows in the data vector have the same number of columns. It does not perform any input validation or error handling, so you should ensure that the input meets these assumptions before using the function.

### Operator 

The code you provided defines an overload of the `<<` operator for output streams (`std::ostream`) that allows you to print the contents of a `std::vector<AnyCls>` to the stream.

Here's a breakdown of the code:

```cpp
template<typename AnyCls>
std::ostream& operator<<(std::ostream& os, const std::vector<AnyCls>& v) {
    os << "{"; // Start printing with a curly brace

    // Iterate over the vector elements
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")"; // Print each element inside parentheses

        if (it != v.end() - 1) {
            os << ", "; // Add a comma and space if it's not the last element
        }
    }

    os << "}"; // End printing with a curly brace
    return os; // Return the stream
}
```
This code defines a templated function `operator<<` that takes two arguments: an output stream `(std::ostream& os)` and a constant reference to a `std::vector<AnyCls>` (`const std::vector<AnyCls>& v`). The AnyCls template parameter allows the vector to hold elements of any type.

Inside the function, it starts by outputting an opening curly brace (`{`) to the stream. Then it iterates over the elements of the vector using a for loop. For each element, it prints the element surrounded by parentheses (`(` and `)`), using *it to dereference the iterator and get the actual element. If the current element is not the last element in the vector, it adds a comma and a space after printing the element.

After iterating over all the elements, it outputs a closing curly brace (`}`) to the stream. Finally, it returns the stream itself (`os`), allowing for chaining of output operations.
### Final Process Object Detector + OC SORT 

The code you provided defines a function named `processFrame` that processes a frame using object detection and tracking algorithms. Let's break down the code step by step:
```cpp
void processFrame(const cv::Mat& frame, std::vector<Detection>& output, ocsort::OCSort& tracker) {
    std::vector<std::vector<float>> data;

    // Iterate over the output vector starting from index 1 (skipping the first element)
    for (int i = 1; i < output.size(); ++i) {
        Detection detection = output[i];
        cv::Rect box = detection.box;
        std::vector<float> row;
        for (;;) {
            // Push the coordinates, confidence, and class ID of the detection to the row vector
            row.push_back(output[i].box.x);
            row.push_back(output[i].box.y);
            row.push_back(output[i].box.x + output[i].box.width);
            row.push_back(output[i].box.y + output[i].box.height);
            row.push_back(output[i].confidence);
            row.push_back(output[i].class_id);
        }
        data.push_back(row); // Add the row vector to the data vector

        // Update the tracker with the data converted to an Eigen matrix
        std::vector<Eigen::RowVectorXf> res = tracker.update(Vector2Matrix(data));

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); // Draw a rectangle around the detection

        // Add text indicating the class name and confidence to the frame
        std::string classString = detection.className + '(' + std::to_string(detection.confidence).substr(0, 4) + ')';
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y + box.height - 10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 0);

        // Process the results from the tracker
        for (auto j : res) {
            int ID = int(j[4]);
            int Class = int(j[5]);
            float conf = j[6];

            // Add ID text and draw a rectangle for each tracked object
            cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), cv::Scalar(0, 0, 255), 1);
        }

        data.clear(); // Clear the data vector for the next iteration
    }
}
```
The function processFrame takes three parameters: a constant reference to a `cv::Mat` object representing a frame, a reference to a vector of Detection objects named output, and a reference to an `ocsort::OCSort` object named tracker.

Inside the function, a `std::vector<std::vector<float>>` named `data` is created to store the converted detection data. Then, the function iterates over the output vector starting from index 1 (skipping the first element) using a for loop. For each `Detection` object, the coordinates (x, y, width, height), confidence, and class ID are extracted and stored in a `std::vector<float>` named `row`. 

# Reference author
Thank you to the original author for building a great codebase. The code below is a modified version of their code, see the original code at [OC SORT CPP](https://github.com/Postroggy/OC_SORT_CPP)
