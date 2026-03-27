#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <array>

std::optional<std::array<cv::Point2f, 4>> localizeQR(const cv::Mat&);
