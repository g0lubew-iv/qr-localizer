#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <optional>
#include <vector>
#include <array>
#include <cmath>

#include "qr_localize.hpp"

namespace {


struct FinderCandidate {
    cv::RotatedRect box;
    cv::Point2f center;
    float sizePx;
    float area;
};


static float norm2(const cv::Point2f& p) {
    return p.x * p.x + p.y * p.y;
}


static float norm(const cv::Point2f& p) {
    return std::sqrt(norm2(p));
}


static cv::Point2f norma(const cv::Point2f& v) {
    float n = norm(v);
    if (n < 1e-6f) {
        return cv::Point2f(0, 0);
    }
    return v * (1.0f / n);
}


static std::array<cv::Point2f, 4> orderQuad(const std::array<cv::Point2f, 4>& pts) {
    std::array<cv::Point2f, 4> out;
    std::vector<cv::Point2f>   p(pts.begin(), pts.end());

    auto sum  = [](const cv::Point2f& a){ return a.x + a.y; };
    auto diff = [](const cv::Point2f& a){ return a.x - a.y; };

    out[0] = *std::min_element(p.begin(), p.end(), [&](auto& a, auto& b){ return sum(a) < sum(b); });   // TL
    out[2] = *std::max_element(p.begin(), p.end(), [&](auto& a, auto& b){ return sum(a) < sum(b); });   // BR
    out[1] = *std::max_element(p.begin(), p.end(), [&](auto& a, auto& b){ return diff(a) < diff(b); }); // TR
    out[3] = *std::min_element(p.begin(), p.end(), [&](auto& a, auto& b){ return diff(a) < diff(b); }); // BL

    return out;
}


static bool isRightAngles(const std::vector<cv::Point>& polygon, float maxCos = 0.35) {
    if (polygon.size() != 4) {
        return false;
    }

    std::vector<double> cosines;

    for (int i = 0; i < 4; i++) {
        cv::Point2f p0 = polygon[i];
        cv::Point2f p1 = polygon[(i + 1) % 4];
        cv::Point2f p2 = polygon[(i + 2) % 4];

        cv::Point2f v1 = p0 - p1;
        cv::Point2f v2 = p2 - p1;

        double denom = std::sqrt(v1.dot(v1) * v2.dot(v2)) + 1e-12; // must be non-zero
        cosines.push_back(std::abs((v1.dot(v2)) / denom));
    }

    double maxC = *std::max_element(cosines.begin(), cosines.end());

    return (maxC < maxCos);
}


static bool isLikeSquare(const std::vector<cv::Point>& polygon, float minRatio = 1.5) {
    cv::RotatedRect rr = cv::minAreaRect(polygon);

    float w = rr.size.width, h = rr.size.height;
    float ratio = std::max(w, h) / std::min(w, h);

    return (ratio <= minRatio);
}


static std::vector<FinderCandidate> findCandidate(const cv::Mat& bwImg, float minArea = 30., float minSize = 2., float approxEpsilon = 0.05) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(bwImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<FinderCandidate> fcs;

    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < minArea) {
            continue;
        }

        int child = hierarchy[i][2];
        if (child < 0) {
            continue;
        }

        int grandChild = hierarchy[child][2];
        if (grandChild < 0) {
            continue;
        }

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, approxEpsilon * cv::arcLength(contours[i], true), true);
        if (approx.size() != 4) {
            continue;
        }
        if (!cv::isContourConvex(approx)) {
            continue;
        }
        if (!isRightAngles(approx)) {
            continue;
        }
        if (!isLikeSquare(contours[i])) {
            continue;
        }


        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        float w = rr.size.width, h = rr.size.height;
        float ratio = std::max(w, h) / std::min(w, h);

        FinderCandidate fc;
        fc.box    = rr;
        fc.center = fc.box.center;
        fc.sizePx = std::min(w, h);
        fc.area   = (float) std::abs(area);

        fcs.push_back(fc);
    }

    return fcs;
}


static std::vector<FinderCandidate> findFinderCandidates(const cv::Mat& grayImg, int kernelSize = 5, float minDist = 3.) {
    cv::Mat blur, bin;
    cv::GaussianBlur(grayImg, blur, cv::Size(kernelSize, kernelSize), 0);
    cv::threshold(blur, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<FinderCandidate> a = findCandidate(bin);
    cv::Mat inv; cv::bitwise_not(bin, inv);
    std::vector<FinderCandidate> b = findCandidate(inv);
    a.insert(a.end(), b.begin(), b.end());

    std::vector<FinderCandidate> unique;
    for (auto& c : a) {
        bool isDuplicate = false;
        for (auto& u : unique) {
            if (norm(c.center - u.center) < minDist) {
                if (c.area > u.area) {
                    u = c;
                }
                isDuplicate = true;
                break;
            }
        }
        if (!isDuplicate) {
            unique.push_back(c);
        }
    }

    std::sort(unique.begin(), unique.end(), [](const auto& x, const auto& y){ return x.area > y.area; });
    return unique;
}


struct FinderPatterns {
    cv::Point2f tl, tr, bl;
    float tlSize, trSize, blSize;
};


static std::optional<FinderPatterns> pickBestFinders(const std::vector<FinderCandidate>& fcs, float lambda = 0.25) {
    if (fcs.size() < 3) {
        return std::nullopt;
    }

    auto cosAt = [](const cv::Point2f& p, const cv::Point2f& q, const cv::Point2f& r) {
        cv::Point2f v1 = q - p, v2 = r - p;
        double denom = std::sqrt(v1.dot(v1) * v2.dot(v2)) + 1e-12; // non-zero
        return std::abs(v1.dot(v2) / denom);
    };

    double bestScore = 1e18;
    int bi = -1, bj = -1, bk = -1;

    for (int i = 0; i < (int) fcs.size(); i++) {
        for (int j = i + 1; j < (int) fcs.size(); j++) {
            for (int k = j + 1; k < (int) fcs.size(); k++) {

                cv::Point2f A = fcs[i].center, B = fcs[j].center, C = fcs[k].center;
                double dab = norm(A-B), dbc = norm(B-C), dac = norm(A-C);
                double dmin = std::min({dab, dbc, dac});
                double dmax = std::max({dab, dbc, dac});

                if (dmin < 10) {
                    continue;
                }
                if (dmax / (dmin + 1e-9) > 6.0) {
                    continue;
                }

                double ca = cosAt(A, B, C);
                double cb = cosAt(B, A, C);
                double cc = cosAt(C, A, B);
                double bestAngle = std::min({ca, cb, cc});

                cv::Point2f P = A, Q = B, R = C;
                if (bestAngle == cb) {
                    P = B; Q = A; R = C;
                } else if (bestAngle == cc) {
                    P = C; Q = A; R = B;
                }

                double d1 = norm(P - Q), d2 = norm(P - R);
                double ratio = std::max(d1, d2) / (std::min(d1, d2) + 1e-9);

                double score = bestAngle + lambda * (ratio - 1.0);
                if (score < bestScore) {
                    bestScore = score;
                    bi = i; bj = j; bk = k;
                }
            }
        }
    }

    if (bi < 0) {
        return std::nullopt;
    }

    cv::Point2f A = fcs[bi].center, B = fcs[bj].center, C = fcs[bk].center;

    double ca = cosAt(A, B, C);
    double cb = cosAt(B, A, C);
    double cc = cosAt(C, A, B);

    cv::Point2f tl, p1, p2;
    float tlSize, p1Size, p2Size;

    if (ca <= cb && ca <= cc) {
        tl = A; p1 = B; p2 = C;
        tlSize = fcs[bi].sizePx; p1Size = fcs[bj].sizePx; p2Size = fcs[bk].sizePx;
    } else if (cb <= ca && cb <= cc) {
        tl = B; p1 = A; p2 = C;
        tlSize = fcs[bj].sizePx; p1Size = fcs[bi].sizePx; p2Size = fcs[bk].sizePx;
    } else {
        tl = C; p1 = A; p2 = B;
        tlSize = fcs[bk].sizePx; p1Size = fcs[bi].sizePx; p2Size = fcs[bj].sizePx;
    }

    cv::Point2f v1 = p1 - tl;
    cv::Point2f v2 = p2 - tl;
    float cross = v1.x * v2.y - v1.y * v2.x; // cross product

    cv::Point2f tr, bl;
    float trSize, blSize;
    if (cross > 0) {
        tr = p1; bl = p2; trSize = p1Size; blSize = p2Size;
    } else {
        tr = p2; bl = p1; trSize = p2Size; blSize = p1Size;
    }

    return FinderPatterns{tl, tr, bl, tlSize, trSize, blSize};
}

} // namespace


std::optional<std::array<cv::Point2f, 4>> localizeQR(const cv::Mat& bgr) {
    if (bgr.empty()) {
        return std::nullopt;
    }

    cv::Mat gray;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = bgr.clone();
    }

    const int targetMax = 1400; double scale = 1.0;
    int maxSide = std::max(gray.cols, gray.rows);
    if (maxSide > targetMax) {
        scale = (double) targetMax / (double) maxSide;
    }

    cv::Mat graySmall;
    if (scale < 1.0) {
        cv::resize(gray, graySmall, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        graySmall = gray;
    }

    auto candidates = findFinderCandidates(graySmall);
    if (candidates.size() < 1) {
        return std::nullopt;
    }

    auto ftOpt = pickBestFinders(candidates);
    if (!ftOpt) {
        return std::nullopt;
    }
    auto ft = *ftOpt;

    constexpr float numberModules           = 21.0f; // QR version 1 (21 modules)
    constexpr float dCentersModules         = numberModules - 7.0f;
    constexpr float fromCenterToEdgeModules = 3.5f;

    cv::Point2f tl = ft.tl * (float)(1.0 / scale);
    cv::Point2f tr = ft.tr * (float)(1.0 / scale);
    cv::Point2f bl = ft.bl * (float)(1.0 / scale);

    cv::Point2f uRaw = tr - tl;
    cv::Point2f vRaw = bl - tl;

    float dx = cv::norm(uRaw);
    float dy = cv::norm(vRaw);
    if (dx < 1e-6f || dy < 1e-6f) {
        return std::nullopt;
    }

    cv::Point2f u = uRaw * (1.0f / dx);
    cv::Point2f v = vRaw * (1.0f / dy);

    /* optional: make v orthogonal to u
    v = v - u * (u.dot(v));
    float vn = cv::norm(v);
    if (vn < 1e-6f) {
        return std::nullopt;
    }
    v = v * (1.0f / vn); */

    float mx = dx / dCentersModules;
    float my = dy / dCentersModules;
    float m  = 0.5f * (mx + my);

    float off = fromCenterToEdgeModules * m;

    cv::Point2f TL = tl - u * off - v * off;
    cv::Point2f TR = tr + u * off - v * off;
    cv::Point2f BL = bl - u * off + v * off;
    cv::Point2f BR = TR + (BL - TL);

    return orderQuad({TL, TR, BR, BL});
}
