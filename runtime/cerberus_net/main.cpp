#include "src/cerberus.hpp"

#include <iostream>
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>

int main(int argc, char** argv)
{
    CERBERUS nnet;
    std::cout << "Init Success!" << std::endl;

    const std::string base_path = "/home/bryce/Documents/Cityscapes Data/leftImg8bit_sequence/train/aachen/";
    cv::Mat image1 = cv::imread(base_path+"aachen_000000_000020_leftImg8bit.png");
    cv::Mat image2 = cv::imread(base_path+"aachen_000001_000020_leftImg8bit.png");

    assert(!image1.empty());
    assert(!image2.empty());

    cv::Size net_input { nnet.getInputH(), nnet.getInputW() };
    cv::resize(image1, image1, net_input);

    std::cout << "Doing inference" << std::endl;

    std::chrono::high_resolution_clock timer;
    auto begin = timer.now();
    nnet.doInference(image1, image2);
    std::cout << "End Time: " << (timer.now() - begin).count() / 1e6 << " ms" << std::endl;

    std::cout << "Showing Images" << std::endl;
    cv::imshow("Sample Input", image1);
    cv::imshow("Sample Depth", nnet.get_depth());

    cv::waitKey(0);
}