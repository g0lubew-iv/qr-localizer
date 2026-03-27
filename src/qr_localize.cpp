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


static bool approxIsQuadWithRightAngles(const std::vector<cv::Point>& approx, double maxCos = 0.35) {
    if (approx.size() != 4) {
        return false;
    }

    std::vector<double> cosines; cosines.reserve(4);

    for (int i = 0; i < 4; i++) {
        cv::Point2f p0 = approx[i];
        cv::Point2f p1 = approx[(i + 1) % 4];
        cv::Point2f p2 = approx[(i + 2) % 4];

        cv::Point2f v1 = p0 - p1;
        cv::Point2f v2 = p2 - p1;
        double denom = std::sqrt(v1.dot(v1) * v2.dot(v2)) + 1e-12;
        double c = (v1.dot(v2)) / denom;
        cosines.push_back(std::abs(c));
    }

    double maxC = *std::max_element(cosines.begin(), cosines.end());
    return (maxC < maxCos);
}


static std::vector<FinderCandidate> findFinderCandidatesOne(const cv::Mat& bw) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bw, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<FinderCandidate> out;
    out.reserve(128);

    for (int i = 0; i < (int) contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 30) {
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
        cv::approxPolyDP(contours[i], approx, 0.05 * cv::arcLength(contours[i], true), true);
        if (approx.size() != 4) {
            continue;
        }
        if (!cv::isContourConvex(approx)) {
            continue;
        }
        if (!approxIsQuadWithRightAngles(approx)) {
            continue;
        }

        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        float w = rr.size.width, h = rr.size.height;
        if (w < 2 || h < 2) {
            continue;
        }

        float ratio = std::max(w, h) / std::min(w, h);
        if (ratio > 1.5f) {
            continue;
        }

        FinderCandidate fc;
        fc.box = rr;
        fc.center = rr.center;
        fc.sizePx = std::min(w,h);
        fc.area = (float) std::abs(area);
        out.push_back(fc);
    }

    return out;
}


static std::vector<FinderCandidate> findFinderCandidates(const cv::Mat& graySmall) {
    cv::Mat blur, bin;
    cv::GaussianBlur(graySmall, blur, cv::Size(5,5), 0);
    cv::threshold(blur, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<FinderCandidate> a = findFinderCandidatesOne(bin);
    cv::Mat inv; cv::bitwise_not(bin, inv);
    std::vector<FinderCandidate> b = findFinderCandidatesOne(inv);
    a.insert(a.end(), b.begin(), b.end());

    std::vector<FinderCandidate> unique;
    for (auto& c : a) {
        bool dup = false;
        for (auto& u : unique) {
            if (norm(c.center - u.center) < 3.0f) {
                if (c.area > u.area) {
                    u = c;
                }
                dup = true;
                break;
            }
        }
        if (!dup) {
            unique.push_back(c);
        }
    }

    std::sort(unique.begin(), unique.end(), [](const auto& x, const auto& y){ return x.area > y.area; });

    if (unique.size() > 60) {
        unique.resize(60);
    }

    return unique;
}


struct FindersTLTRBL {
    cv::Point2f tl, tr, bl;
    float tlSize, trSize, blSize;
};


static std::optional<FindersTLTRBL> pickBestFinders(const std::vector<FinderCandidate>& cands) {
    if (cands.size() < 3) {
        return std::nullopt;
    }

    auto angleCosAt = [](const cv::Point2f& p, const cv::Point2f& q, const cv::Point2f& r) {
        cv::Point2f v1 = q - p, v2 = r - p;
        double denom = std::sqrt(v1.dot(v1) * v2.dot(v2)) + 1e-12;
        return std::abs(v1.dot(v2) / denom);
    };

    double bestScore = 1e18;
    int bi = -1, bj = -1, bk = -1;

    for (int i = 0; i < (int) cands.size(); i++) {
        for (int j = i + 1; j < (int) cands.size(); j++) {
            for (int k = j + 1; k < (int) cands.size(); k++) {

                cv::Point2f A = cands[i].center, B = cands[j].center, C = cands[k].center;
                double dab = norm(A-B), dbc = norm(B-C), dac = norm(A-C);
                double dmin = std::min({dab, dbc, dac});
                double dmax = std::max({dab, dbc, dac});

                if (dmin < 10) {
                    continue;
                }
                if (dmax / (dmin + 1e-9) > 6.0) {
                    continue;
                }

                double ca = angleCosAt(A,B,C);
                double cb = angleCosAt(B,A,C);
                double cc = angleCosAt(C,A,B);
                double bestAng = std::min({ca,cb,cc});

                cv::Point2f P = A, Q = B, R = C;
                if (bestAng == cb) {
                    P = B; Q = A; R = C;
                } else if (bestAng == cc) {
                    P = C; Q = A; R = B;
                }

                double d1 = norm(P - Q), d2 = norm(P - R);
                double ratio = std::max(d1, d2) / (std::min(d1, d2) + 1e-9);

                double score = bestAng + 0.25 * (ratio - 1.0);
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

    cv::Point2f A = cands[bi].center, B = cands[bj].center, C = cands[bk].center;

    double ca = angleCosAt(A,B,C);
    double cb = angleCosAt(B,A,C);
    double cc = angleCosAt(C,A,B);

    cv::Point2f tl, p1, p2;
    float tlSize, p1Size, p2Size;

    if (ca <= cb && ca <= cc) {
        tl=A; p1=B; p2=C; tlSize=cands[bi].sizePx; p1Size=cands[bj].sizePx; p2Size=cands[bk].sizePx;
    } else if (cb <= ca && cb <= cc) {
        tl=B; p1=A; p2=C; tlSize=cands[bj].sizePx; p1Size=cands[bi].sizePx; p2Size=cands[bk].sizePx;
    } else {
        tl=C; p1=A; p2=B; tlSize=cands[bk].sizePx; p1Size=cands[bi].sizePx; p2Size=cands[bj].sizePx;
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

    return FindersTLTRBL{tl, tr, bl, tlSize, trSize, blSize};
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

    auto candsSmall = findFinderCandidates(graySmall);
    if (candsSmall.size() < 1) {
        return std::nullopt;
    }

    auto ftOpt = pickBestFinders(candsSmall);
    if (!ftOpt) {
        return std::nullopt;
    }
    auto ft = *ftOpt;

    cv::Point2f tl = ft.tl * (float) (1.0 / scale);
    cv::Point2f tr = ft.tr * (float) (1.0 / scale);
    cv::Point2f bl = ft.bl * (float) (1.0 / scale);

    float finderSize = ((ft.tlSize + ft.trSize + ft.blSize) / 3.0f) * (float)(1.0 / scale);

    cv::Point2f u = norma(tr - tl); // right
    cv::Point2f v = norma(bl - tl); // down

    v = v - u * (u.dot(v));
    v = norma(v);

    float S  = 0.5f * (norm(tr - tl) + finderSize + norm(bl - tl) + finderSize);

    cv::Point2f TL = tl - u * (0.5f * finderSize) - v * (0.5f * finderSize);
    cv::Point2f TR = TL + u * S;
    cv::Point2f BL = TL + v * S;
    cv::Point2f BR = TL + u * S + v * S;

    auto quad = orderQuad({TL, TR, BR, BL});

    cv::Rect2f bounds(-20.f, -20.f, (float) bgr.cols + 40.f, (float) bgr.rows + 40.f);
    for (auto& p : quad) {
        if (!bounds.contains(p)) {
            return std::nullopt;
        }
    }

    return quad;
}
