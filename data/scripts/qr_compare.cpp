#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using json = nlohmann::json;

static bool readQuadFromJson(const std::string& jsonPath, std::vector<cv::Point2f>& pts4) {
    std::ifstream ifs(jsonPath);
    if (!ifs) return false;

    json j;
    ifs >> j;

    if (!j.contains("shapes") || !j["shapes"].is_array() || j["shapes"].empty()) return false;

    json shape;
    bool found = false;
    for (auto& sh : j["shapes"]) {
        if (sh.contains("label") && sh["label"].is_string() && sh["label"] == "qr") {
            shape = sh;
            found = true;
            break;
        }
    }
    if (!found) return false;

    if (!shape.contains("points") || !shape["points"].is_array()) return false;
    auto pts = shape["points"];
    if (pts.size() != 4) return false;

    pts4.clear();
    for (int i = 0; i < 4; ++i) {
        if (!pts[i].is_array() || pts[i].size() != 2) return false;
        float x = pts[i][0].get<float>();
        float y = pts[i][1].get<float>();
        pts4.emplace_back(x, y);
    }
    return true;
}

static cv::Mat keepPolygonMakeGreenBg(const cv::Mat& bgr, const std::vector<cv::Point2f>& quad) {
    cv::Mat out(bgr.size(), bgr.type(), cv::Scalar(0, 255, 0)); // зелёный фон

    std::vector<cv::Point> poly;
    poly.reserve(quad.size());
    for (auto& p : quad) poly.emplace_back((int)std::lround(p.x), (int)std::lround(p.y));

    cv::Mat mask(bgr.rows, bgr.cols, CV_8U, cv::Scalar(0));
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));

    bgr.copyTo(out, mask);
    return out;
}

static bool detectQR(const cv::Mat& bgr, std::string& decoded, std::vector<cv::Point2f>& corners) {
    cv::QRCodeDetector det;
    cv::Mat pts;
    decoded = det.detectAndDecode(bgr, pts);
    corners.clear();

    if (!pts.empty() && pts.total() >= 4) {
        pts = pts.reshape(2);
        for (int i = 0; i < std::min(4, pts.rows); ++i) {
            corners.emplace_back(pts.at<float>(i,0), pts.at<float>(i,1));
        }
    }
    return !decoded.empty();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: qr_compare <image_path> <poly.json> [out_masked.png]\n";
        return 1;
    }

    std::string imagePath = argv[1];
    std::string jsonPath  = argv[2];
    std::string outMasked = (argc >= 4) ? argv[3] : "";

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: cannot read image: " << imagePath << "\n";
        return 2;
    }

    std::string decodedFull;
    std::vector<cv::Point2f> cornersFull;
    bool okFull = detectQR(img, decodedFull, cornersFull);

    std::vector<cv::Point2f> quad;
    if (!readQuadFromJson(jsonPath, quad)) {
        std::cerr << "Error: cannot read quad from json or shapes empty: " << jsonPath << "\n";
        return 3;
    }

    cv::Mat masked = keepPolygonMakeGreenBg(img, quad);

    if (!outMasked.empty()) {
        cv::imwrite(outMasked, masked);
    }

    std::string decodedMasked;
    std::vector<cv::Point2f> cornersMasked;
    bool okMasked = detectQR(masked, decodedMasked, cornersMasked);

    std::cout << "FULL: "   << (okFull ? "YES" : "NO") << "\n";
    std::cout << "MASKED: " << (okMasked ? "YES" : "NO") << "\n";

    if (okFull)   std::cout << "FULL_TEXT: "   << decodedFull << "\n";
    if (okMasked) std::cout << "MASKED_TEXT: " << decodedMasked << "\n";

    return 0;
}
