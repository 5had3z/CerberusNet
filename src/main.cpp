#include <iostream>
#include <opencv2/opencv.hpp>
#include "lib/nvx_pipeline.hpp"
#include <chrono>
#include <thread>

int main(int argc, char** argv)
{
    cv::VideoCapture cap("/home/bryce/Documents/basler_test_2020-02-29_20-39.mkv");
    if (cap.isOpened() == false){
       std::cerr << "Error opening vid file:";
        return -1;
    }

    std::cout << "Opened frame!\n";
    cv::Mat cap_frame;
    PROCESSOR_NVX processor(1920, 1080);
    std::cout << "Built Processor!\n";

    while (cap.isOpened())
    {
        auto time = std::chrono::system_clock::now();
        cap >> cap_frame;
        processor.ProcessFrame(cap_frame);

        processor.DisplayFrame();
        
        std::this_thread::sleep_until(time + std::chrono::milliseconds(33));
    }

    return 0;
}