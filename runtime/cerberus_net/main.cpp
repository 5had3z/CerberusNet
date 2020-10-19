#include "src/cerberus.hpp"

#include <iostream>
#include <cassert>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv)
{
    CERBERUS nnet;
    std::cout << "Init Success!";

    cv::Mat image1 = cv::imread("/home/bryce/aachen_000000_000020_leftImg8bit.png");
    cv::Mat image2 = cv::imread("/home/bryce/aachen_000001_000020_leftImg8bit.png");

    assert(!image1.empty());
    assert(!image2.empty());

    cv::Size net_input { nnet.getInputH(), nnet.getInputW() };
    cv::resize(image1, image1, net_input);

    cv::imshow("Sample Input", image1);
    cv::waitKey(0);
}