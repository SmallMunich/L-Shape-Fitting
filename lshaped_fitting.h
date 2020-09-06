#ifndef _L_SHAPED_FITTING_H
#define _L_SHAPED_FITTING_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class LShapedFIT
{
public:
    LShapedFIT();

    ~LShapedFIT();
	
    // For Each Cluster.
    cv::RotatedRect FitBox(std::vector<cv::Point2f>* pointcloud_ptr);

    std::vector<cv::Point2f> getRectVertex();

private:
    // Different Criterion For Cluster BBox.
    double calc_area_criterion(const cv::Mat& c1, const cv::Mat& c2);

    double calc_nearest_criterion(const cv::Mat& c1, const cv::Mat& c2);

    double calc_variances_criterion(const cv::Mat& c1, const cv::Mat& c2);

    double calc_var(const std::vector<double>& v);

    void calc_cross_point(const double a0, const double a1, const double b0, const double b1, const double c0,
                          const double c1, double& x, double& y);

    cv::RotatedRect calc_rect_contour();

public:
    enum Criterion
    {
        AREA,
        NEAREST,
        VARIANCE
    };

private:
    double min_dist_of_nearest_crit_;
    double dtheta_deg_for_search_;

    Criterion criterion_;

    std::vector<double> a_;
    std::vector<double> b_;
    std::vector<double> c_;

    std::vector<cv::Point2f> vertex_pts_;
    cv::Point2f              hot_pt_;

};  // class LShapedFIT

#endif