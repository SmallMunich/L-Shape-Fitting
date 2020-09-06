#include <glog/logging.h>
#include <lshaped_fitting.h>

#include <algorithm>
#include <numeric>
#include <opencv2/core/core.hpp>

LShapedFIT::LShapedFIT()
{
    min_dist_of_nearest_crit_ = 0.01;
    dtheta_deg_for_search_    = 1.0;

    criterion_ = LShapedFIT::VARIANCE;

    vertex_pts_.clear();
}

LShapedFIT::~LShapedFIT()
{
}

cv::RotatedRect LShapedFIT::FitBox(std::vector<cv::Point2f>* pointcloud_ptr)
{
    std::vector<cv::Point2f>& points = *pointcloud_ptr;

    if (points.size() < 3)
        return cv::RotatedRect();

    // Initialize Contour Points Matrix.
    cv::Mat Matrix_pts = cv::Mat::zeros(points.size(), 2, CV_64FC1);

    for (size_t i = 0; i < points.size(); ++i)
    {
        Matrix_pts.at<double>(i, 0) = points[i].x;
        Matrix_pts.at<double>(i, 1) = points[i].y;
    }

    double dtheta = dtheta_deg_for_search_ * M_PI / 180;  // np.deg2rad(x) --> x * pi/180

    double minimal_cost = (-1.0) * std::numeric_limits<double>::max();
    double best_theta   = std::numeric_limits<double>::max();

    // Search This Best Direction For ENUM.
    int loop_number = ceil((M_PI / 2.0 - dtheta) / dtheta);

    cv::Mat e1 = cv::Mat::zeros(1, 2, CV_64FC1);
    cv::Mat e2 = cv::Mat::zeros(1, 2, CV_64FC1);

    for (int k = 0; k < loop_number; ++k)
    {
        double theta = k * dtheta;
        double cost  = std::numeric_limits<double>::min();
        // Be Sure Yaw Is In Range.
        if (theta < (M_PI / 2.0 - dtheta))
        {
            e1.at<double>(0, 0) = cos(theta);
            e1.at<double>(0, 1) = sin(theta);
            e2.at<double>(0, 0) = -sin(theta);
            e2.at<double>(0, 1) = cos(theta);

            cv::Mat c1 = Matrix_pts * e1.t();
            cv::Mat c2 = Matrix_pts * e2.t();

            if (criterion_ == Criterion::AREA)
            {
                cost = calc_area_criterion(c1, c2);
            }
            else if (criterion_ == Criterion::NEAREST)
            {
                cost = calc_nearest_criterion(c1, c2);
            }
            else if (criterion_ == Criterion::VARIANCE)
            {
                cost = calc_variances_criterion(c1, c2);
            }
            else
            {
                std::cout << "L-Shaped Algorithm Criterion Is Not Supported." << std::endl;
                break;
            }

            if (minimal_cost < cost)
            {
                minimal_cost = cost;
                best_theta   = theta;
            }
        }
        else
        {
            break;
        }
    }

    if (minimal_cost > (-1.0) * std::numeric_limits<double>::max() && best_theta < std::numeric_limits<double>::max())
    {
        ;  // Do Nothing, Continue Run As Follows.
    }
    else
    {
        std::cout << "RotatedRect Fit Failed." << std::endl;
        return cv::RotatedRect();
    }

    double sin_s = sin(best_theta);
    double cos_s = cos(best_theta);

    cv::Mat e1_s          = cv::Mat::zeros(1, 2, CV_64FC1);
    e1_s.at<double>(0, 0) = cos_s;
    e1_s.at<double>(0, 1) = sin_s;

    cv::Mat e2_s          = cv::Mat::zeros(1, 2, CV_64FC1);
    e2_s.at<double>(0, 0) = -sin_s;
    e2_s.at<double>(0, 1) = cos_s;

    cv::Mat c1_s = Matrix_pts * e1_s.t();
    cv::Mat c2_s = Matrix_pts * e2_s.t();

    double min_c1_s = std::numeric_limits<double>::max();
    double max_c1_s = (-1.0) * std::numeric_limits<double>::max();
    double min_c2_s = std::numeric_limits<double>::max();
    double max_c2_s = (-1.0) * std::numeric_limits<double>::max();

    cv::minMaxLoc(c1_s, &min_c1_s, &max_c1_s, NULL, NULL);
    cv::minMaxLoc(c2_s, &min_c2_s, &max_c2_s, NULL, NULL);

    a_.clear();
    b_.clear();
    c_.clear();

    if (min_c1_s < std::numeric_limits<double>::max() && min_c2_s < std::numeric_limits<double>::max() &&
        max_c1_s > (-1.0) * std::numeric_limits<double>::max() &&
        max_c2_s > (-1.0) * std::numeric_limits<double>::max())
    {
        a_.push_back(cos_s);
        b_.push_back(sin_s);
        c_.push_back(min_c1_s);

        a_.push_back(-sin_s);
        b_.push_back(cos_s);
        c_.push_back(min_c2_s);

        a_.push_back(cos_s);
        b_.push_back(sin_s);
        c_.push_back(max_c1_s);

        a_.push_back(-sin_s);
        b_.push_back(cos_s);
        c_.push_back(max_c2_s);

        return calc_rect_contour();
    }
    else
    {
        return cv::RotatedRect();
    }
}

double LShapedFIT::calc_area_criterion(const cv::Mat& c1, const cv::Mat& c2)
{
    std::vector<double> c1_deep;
    std::vector<double> c2_deep;

    for (int i = 0; i < c1.rows; i++)
    {
        for (int j = 0; j < c1.cols; j++)
        {
            c1_deep.push_back(c1.at<double>(i, j));
            c2_deep.push_back(c2.at<double>(i, j));
        }
    }

    // sort vector from min to max.
    sort(c1_deep.begin(), c1_deep.end());
    sort(c2_deep.begin(), c2_deep.end());

    int n_c1 = c1_deep.size();
    int n_c2 = c2_deep.size();

    double c1_min = c1_deep[0];
    double c2_min = c2_deep[0];

    double c1_max = c1_deep[n_c1 - 1];
    double c2_max = c2_deep[n_c2 - 1];

    double alpha = -(c1_max - c1_min) * (c2_max - c2_min);

    return alpha;
}

double LShapedFIT::calc_nearest_criterion(const cv::Mat& c1, const cv::Mat& c2)
{
    std::vector<double> c1_deep;
    std::vector<double> c2_deep;

    for (int i = 0; i < c1.rows; i++)
    {
        for (int j = 0; j < c1.cols; j++)
        {
            c1_deep.push_back(c1.at<double>(i, j));
            c2_deep.push_back(c2.at<double>(i, j));
        }
    }

    // sort vector from min to max.
    sort(c1_deep.begin(), c1_deep.end());
    sort(c2_deep.begin(), c2_deep.end());

    int n_c1 = c1_deep.size();
    int n_c2 = c2_deep.size();

    double c1_min = c1_deep[0];
    double c2_min = c2_deep[0];

    double c1_max = c1_deep[n_c1 - 1];
    double c2_max = c2_deep[n_c2 - 1];

    std::vector<double> d1;
    std::vector<double> d2;

    for (int i = 0; i < n_c1; i++)
    {
        double temp = std::min(sqrt(pow((c1_max - c1_deep[i]), 2)), sqrt(pow((c1_deep[i] - c1_min), 2)));
        d1.push_back(temp);
    }

    for (int i = 0; i < n_c2; i++)
    {
        double temp = std::min(sqrt(pow((c2_max - c2_deep[i]), 2)), sqrt(pow((c2_deep[i] - c2_min), 2)));
        d2.push_back(temp);
    }

    double beta = 0;

    for (size_t i = 0; i < d1.size(); i++)
    {
        double d = std::max(std::min(d1[i], d2[i]), min_dist_of_nearest_crit_);
        beta += (1.0 / d);
    }

    return beta;
}

double LShapedFIT::calc_variances_criterion(const cv::Mat& c1, const cv::Mat& c2)
{
    std::vector<double> c1_deep;
    std::vector<double> c2_deep;

    for (int i = 0; i < c1.rows; i++)
    {
        for (int j = 0; j < c1.cols; j++)
        {
            c1_deep.push_back(c1.at<double>(i, j));
            c2_deep.push_back(c2.at<double>(i, j));
        }
    }

    // sort vector from min to max.
    sort(c1_deep.begin(), c1_deep.end());
    sort(c2_deep.begin(), c2_deep.end());

    int n_c1 = c1_deep.size();
    int n_c2 = c2_deep.size();

    double c1_min = c1_deep[0];
    double c2_min = c2_deep[0];

    double c1_max = c1_deep[n_c1 - 1];
    double c2_max = c2_deep[n_c2 - 1];

    std::vector<double> d1;
    std::vector<double> d2;

    // D1 = [ min( [np.linalg.norm(c1_max - ic1), np.linalg.norm(ic1 - c1_min)] ) for ic1 in c1 ]
    for (int i = 0; i < n_c1; i++)
    {
        double temp = std::min(sqrt(pow((c1_max - c1_deep[i]), 2)), sqrt(pow((c1_deep[i] - c1_min), 2)));
        d1.push_back(temp);
    }

    for (int i = 0; i < n_c2; i++)
    {
        double temp = std::min(sqrt(pow((c2_max - c2_deep[i]), 2)), sqrt(pow((c2_deep[i] - c2_min), 2)));
        d2.push_back(temp);
    }

    std::vector<double> e1;
    std::vector<double> e2;

    assert(d1.size() == d2.size());

    // d1.size() || d2.size() Is equals.
    for (size_t i = 0; i < d1.size(); i++)
    {
        if (d1[i] < d2[i])
        {
            e1.push_back(d1[i]);
        }
        else
        {
            e2.push_back(d2[i]);
        }
    }

    double v1 = 0.0;
    if (!e1.empty())
    {
        v1 = (-1.0) * calc_var(e1);
    }

    double v2 = 0.0;
    if (!e2.empty())
    {
        v2 = (-1.0) * calc_var(e2);
    }

    double gamma = v1 + v2;

    return gamma;
}

double LShapedFIT::calc_var(const std::vector<double>& v)
{
    double sum  = std::accumulate(std::begin(v), std::end(v), 0.0);
    double mean = sum / v.size();

    double acc_var_num = 0.0;

    std::for_each(std::begin(v), std::end(v), [&](const double d) { acc_var_num += (d - mean) * (d - mean); });

    double var = sqrt(acc_var_num / (v.size() - 1));

    return var;
}

void LShapedFIT::calc_cross_point(const double a0, const double a1, const double b0, const double b1, const double c0,
                                  const double c1, double& x, double& y)
{
    x = (b0 * (-c1) - b1 * (-c0)) / (a0 * b1 - a1 * b0);
    y = (a1 * (-c0) - a0 * (-c1)) / (a0 * b1 - a1 * b0);
}

cv::RotatedRect LShapedFIT::calc_rect_contour()
{
    // std::vector<cv::Point2f> vertex_pts_;
    vertex_pts_.clear();

    double top_left_x = 0.0, top_left_y = 0.0;
    calc_cross_point(a_[0], a_[1], b_[0], b_[1], c_[0], c_[1], top_left_x, top_left_y);
    vertex_pts_.push_back(cv::Point2f(top_left_x, top_left_y));

    double top_right_x = 0.0, top_right_y = 0.0;
    calc_cross_point(a_[1], a_[2], b_[1], b_[2], c_[1], c_[2], top_right_x, top_right_y);
    vertex_pts_.push_back(cv::Point2f(top_right_x, top_right_y));

    double bottom_left_x = 0.0, bottom_left_y = 0.0;
    calc_cross_point(a_[2], a_[3], b_[2], b_[3], c_[2], c_[3], bottom_left_x, bottom_left_y);
    vertex_pts_.push_back(cv::Point2f(bottom_left_x, bottom_left_y));

    double bottom_right_x = 0.0, bottom_right_y = 0.0;
    calc_cross_point(a_[3], a_[0], b_[3], b_[0], c_[3], c_[0], bottom_right_x, bottom_right_y);
    vertex_pts_.push_back(cv::Point2f(bottom_right_x, bottom_right_y));

    return cv::minAreaRect(vertex_pts_);
}

std::vector<cv::Point2f> LShapedFIT::getRectVertex()
{
    return vertex_pts_;
}
