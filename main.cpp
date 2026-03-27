#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include "src/qr_localize.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

static void drawQuad(cv::Mat& img, const std::array<cv::Point2f,4>& q) {
    for (int i = 0; i < 4; ++i) {
        cv::line(img, q[i], q[(i + 1) % 4], cv::Scalar(0,255,0), 3, cv::LINE_AA);
        cv::circle(img, q[i], 5, cv::Scalar(0,0,255), -1, cv::LINE_AA);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: qr_localizer <image_path> --out <out.json> [--show] [--vis out.png]\n";
        return 1;
    }

    std::string imagePath = argv[1];

    bool doShow = false;
    bool doSaveVis = false;
    std::string visPath;

    bool haveOut = false;
    std::string outJsonPath;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--show") {
            doShow = true;
        } else if (a == "--vis" && i + 1 < argc) {
            doSaveVis = true;
            visPath = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            haveOut = true;
            outJsonPath = argv[++i];
        }
    }

    if (!haveOut) {
        std::cerr << "Error: output json is not specified. Use --out <out.json>\n";
        return 1;
    }

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

    json out;
    out["shapes"] = json::array();
    out["imagePath"] = fs::path(imagePath).filename().string();
    out["imageHeight"] = img.empty() ? 0 : img.rows;
    out["imageWidth"]  = img.empty() ? 0 : img.cols;

    if (!img.empty()) {
        auto quadOpt = localizeQR(img);
        if (quadOpt) {
            const auto& q = *quadOpt;

            json shape;
            shape["label"] = "qr";
            shape["points"] = json::array({
                json::array({q[0].x, q[0].y}), // TL
                json::array({q[1].x, q[1].y}), // TR
                json::array({q[2].x, q[2].y}), // BR
                json::array({q[3].x, q[3].y})  // BL
            });
            shape["description"] = "";
            shape["shape_type"] = "polygon";

            out["shapes"].push_back(shape);

            if (doShow || doSaveVis) {
                cv::Mat vis = img.clone();
                drawQuad(vis, q);

                if (doSaveVis) cv::imwrite(visPath, vis);

                if (doShow) {
                    cv::namedWindow("QR localization", cv::WINDOW_NORMAL);
                    cv::imshow("QR localization", vis);
                    cv::waitKey(0);
                }
            }
        }
    }

    try {
        fs::path outPath(outJsonPath);
        if (outPath.has_parent_path()) {
            fs::create_directories(outPath.parent_path());
        }
    } catch (...) {
        //
    }

    std::ofstream ofs(outJsonPath);
    if (!ofs) {
        std::cerr << "Error: cannot open file for writing: " << outJsonPath << "\n";
        return 2;
    }
    ofs << out.dump(2) << "\n";
    ofs.close();

    return 0;
}
