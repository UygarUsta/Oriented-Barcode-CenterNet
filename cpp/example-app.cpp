#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <torch/torch.h>
#include <torch/script.h>
#include <filesystem>
#include <iostream>
#include <tuple>

// Normalize class to preprocess image channels
class Normalize {
public:
    Normalize() {
        mean = {0.485, 0.456, 0.406};
        std = {0.229, 0.224, 0.225};
    }

    // Function to normalize the image
    cv::Mat operator()(const cv::Mat& image) {
        cv::Mat normalizedImage;
        image.convertTo(normalizedImage, CV_32F, 1.0/255.0); // Convert to float32 and scale

        // Normalize each channel
        for (int i = 0; i < 3; ++i) {
            normalizedImage.forEach<cv::Vec3f>([&](cv::Vec3f &pixel, const int * position) -> void {
                pixel[i] = (pixel[i] - mean[i]) / std[i];
            });
        }

        return normalizedImage;
    }

private:
    std::vector<float> mean;
    std::vector<float> std;
};


namespace cv {
    typedef Vec<float, 5> Vec5f;
}


// Assuming you have a constant MODEL_SCALE defined somewhere
const float MODEL_SCALE = 4; // Replace with your actual value


std::pair<cv::Mat, cv::Mat> pred2box(const cv::Mat& hm, const torch::Tensor& offset, const torch::Tensor& regr, const torch::Tensor& cos_sin_hm, float thresh = 0.99) {


    // Get center points where heatmap value is above threshold
    cv::Mat pred;
    cv::compare(hm, thresh, pred, cv::CMP_GT); // pred is a binary mask

    std::vector<cv::Point> pred_center_points;
    cv::findNonZero(pred, pred_center_points);

    // Extract regressions and angles at predicted centers
    std::vector<cv::Vec2f> pred_r;
    std::vector<cv::Vec2f> pred_angles;
    for (auto& center : pred_center_points) {
        float regr_val1 = regr.index({0,center.y, center.x}).item<float>();
        float regr_val2 = regr.index({1,center.y, center.x}).item<float>();

        float cos_sin_val1 = cos_sin_hm.index({0,center.y, center.x}).item<float>();
        float cos_sin_val2 = cos_sin_hm.index({1,center.y, center.x}).item<float>();


        pred_angles.push_back(cv::Vec2f(cos_sin_val1, cos_sin_val2));
        pred_r.push_back(cv::Vec2f(regr_val1, regr_val2));

    }

    // Create bounding boxes
    std::vector<cv::Vec5f> boxes;
    cv::Mat scores;
    hm.convertTo(scores, CV_32F); // Convert scores to float for later use

    for (size_t i = 0; i < pred_center_points.size(); ++i) {
        auto& center = pred_center_points[i];
        const auto& b = pred_r[i];
        const auto& pred_angle = pred_angles[i];

        float offsetx = offset.index({0,center.y, center.x}).item<float>();
        float offsety = offset.index({1,center.y, center.x}).item<float>();

        cv::Vec2f offset_xy = cv::Vec2f(offsetx, offsety);
        float angle = std::atan2(pred_angle[1], pred_angle[0]);


        cv::Vec5f arr = {
            (center.x + offset_xy[0]) * MODEL_SCALE,
            (center.y + offset_xy[1]) * MODEL_SCALE,
            b[0] * MODEL_SCALE,
            b[1] * MODEL_SCALE,
            angle
        };

        boxes.push_back(arr);
    }

    // Convert to cv::Mat for output
    cv::Mat boxes_mat(boxes.size(), 1, CV_32FC(5), boxes.data());
    cv::Mat scores_mat;
    for (const auto& center : pred_center_points) {
        scores_mat.push_back(scores.at<float>(center));
    }


    return {boxes_mat, scores_mat};
}







cv::Mat select(cv::Mat& hm, float threshold) {
    cv::Mat pred;
    cv::compare(hm, threshold, pred, cv::CMP_GT);

    std::vector<cv::Point> pred_centers;
    cv::findNonZero(pred, pred_centers);

    for (size_t i = 0; i < pred_centers.size(); ++i) {
        for (size_t j = i + 1; j < pred_centers.size(); ++j) {
            const cv::Point& ci = pred_centers[i];
            const cv::Point& cj = pred_centers[j];

            float distance = cv::norm(ci - cj);
            if (distance <= 2) {
                float score_i = hm.at<float>(ci);
                float score_j = hm.at<float>(cj);

                if (score_i > score_j) {
                    hm.at<float>(cj) = 0;
                } else {
                    hm.at<float>(ci) = 0;
                }
            }
        }
    }

    return hm;
}





std::tuple<cv::Mat, float, int, int> resize_and_pad(const cv::Mat& image, const cv::Size& target_size = cv::Size(512, 512)) {
    int original_height = image.rows;
    int original_width = image.cols;
    int target_width = target_size.width;
    int target_height = target_size.height;

    // Calculate the scaling factor
    float scale = std::min(static_cast<float>(target_width) / original_width, static_cast<float>(target_height) / original_height);

    // Calculate new dimensions
    int new_width = static_cast<int>(original_width * scale);
    int new_height = static_cast<int>(original_height * scale);

    // Resize the image
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));

    // Pad the image to the target size
    int delta_w = target_width - new_width;
    int delta_h = target_height - new_height;
    int top = delta_h / 2;
    int bottom = delta_h - top;
    int left = delta_w / 2;
    int right = delta_w - left;

    cv::Mat padded_image;
    cv::copyMakeBorder(resized_image, padded_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return std::make_tuple(padded_image, scale, left, top);
}



std::vector<cv::Point> pred4corner(const cv::Mat& hm, float thresh = 0.99) {
    float threshold = 0.2;
    cv::Mat thresholded_heatmap;
    cv::threshold(hm, thresholded_heatmap, threshold, 1, cv::THRESH_BINARY);

    cv::Mat thresholded_heatmap_8u;
    thresholded_heatmap.convertTo(thresholded_heatmap_8u, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholded_heatmap_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> keypoints;
    for (const auto& cnt : contours) {
        cv::Moments M = cv::moments(cnt);
        if (M.m00 != 0) { // Avoid division by zero
            int cx = static_cast<int>(M.m10 / M.m00);
            int cy = static_cast<int>(M.m01 / M.m00);
            keypoints.push_back(cv::Point(cx, cy));
        }
    }

    return keypoints;
}


cv::Mat showbox(const cv::Mat& img, const cv::Mat& hm, const torch::Tensor& offset, const torch::Tensor& regr, const torch::Tensor& cos_sin_hm, float thresh = 0.9) {
    auto [boxes, _] = pred2box(hm, offset, regr, cos_sin_hm, thresh); // Get predicted boxes

    cv::Mat sample = img.clone(); // Create a copy to draw on

    for (int i = 0; i < boxes.rows; ++i) {
        const float* box_data = boxes.ptr<float>(i); // Access box data

        cv::Point center(box_data[0], box_data[1]);

        float cos_angle = std::cos(box_data[4]);
        float sin_angle = std::sin(box_data[4]);

        cv::Mat rot = (cv::Mat_<float>(2, 2) << cos_angle, sin_angle, -sin_angle, cos_angle);

        cv::Mat half_size = (cv::Mat_<float>(2, 1) << box_data[2] / 2, box_data[3] / 2);

        cv::Mat bottom_right = rot * half_size;
        cv::Mat top_right = rot * (cv::Mat_<float>(2, 1) << box_data[2] / 2, -box_data[3] / 2);
        cv::Mat top_left = rot * (cv::Mat_<float>(2, 1) << -box_data[2] / 2, -box_data[3] / 2);
        cv::Mat bottom_left = rot * (cv::Mat_<float>(2, 1) << -box_data[2] / 2, box_data[3] / 2);

        int thickness = 3;
        cv::line(sample, center + cv::Point(bottom_right), center + cv::Point(top_right), cv::Scalar(0, 220, 0), thickness);
        cv::line(sample, center + cv::Point(bottom_right), center + cv::Point(bottom_left), cv::Scalar(220, 220, 0), thickness);
        cv::line(sample, center + cv::Point(top_left), center + cv::Point(bottom_left), cv::Scalar(220, 220, 0), thickness);
        cv::line(sample, center + cv::Point(top_left), center + cv::Point(top_right), cv::Scalar(220, 220, 0), thickness);
    }

    return sample;
}





namespace fs = std::filesystem;

int main() {
    // Load your PyTorch model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("C:/Users/John/Desktop/CPP_Centernet/hardnet_angle_4c_centernet_jit.pth"); // Replace with the actual path
        model.to(torch::kCUDA); // Move model to GPU if available
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    bool image = false;


    std::string test_folder = "C:/Users/John/Desktop/rotated_barcode/roboflow_barcode/test/images/";
    float threshold = 0.2;
    bool half = false; // Set to true if you want to use half precision

    if (image) {
    // Iterate through image files in the test folder
    for (const auto& entry : fs::directory_iterator(test_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            cv::Mat image = cv::imread(entry.path().string());

            // Preprocess the image
            auto [resized_image, scale, left, top] = resize_and_pad(image);
            cv::Mat img = resized_image.clone();
            cv::Mat imgshow = img.clone();
            img = Normalize()(img);

            torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kFloat).permute({0, 3, 1, 2});

            if (half) {
                tensor = tensor.to(torch::kCUDA).to(torch::kHalf); //.unsqueeze(0);
            } else {
                tensor = tensor.to(torch::kCUDA); //.unsqueeze(0);
            }

            // Perform inference
            torch::NoGradGuard no_grad;

            auto outputs = model.forward({tensor}).toTuple();
            torch::Tensor hm = outputs->elements()[0].toTensor().to(torch::kCPU).squeeze(0);
            torch::Tensor offset_tensor = outputs->elements()[1].toTensor().to(torch::kCPU).squeeze(0);
            torch::Tensor wh_tensor = outputs->elements()[2].toTensor().to(torch::kCPU).squeeze(0);
            torch::Tensor angle_tensor = outputs->elements()[3].toTensor().to(torch::kCPU).squeeze(0);


            // Postprocess the outputs
            hm = torch::sigmoid(hm);
            cv::Mat hm_mat(hm.size(1), hm.size(2), CV_32F, hm[0].data_ptr<float>());
            cv::Mat hm_mat_corner(hm.size(1), hm.size(2), CV_32F, hm[1].data_ptr<float>());


            cv::imshow("heatmap",hm_mat);
            cv::imshow("heatmap corner",hm_mat_corner);


            hm_mat = select(hm_mat, threshold);



            // Visualize the results
            cv::Mat sample = showbox(imgshow, hm_mat, offset_tensor, wh_tensor, angle_tensor, threshold);

            cv::resize(hm_mat_corner,hm_mat_corner,cv::Size(512,512));
            auto kpoints = pred4corner(hm_mat_corner,threshold);

            for (auto element: kpoints) {

                cv::circle(sample,cv::Point(element.x,element.y),5,cv::Scalar(255,255,255),-1);
            }

            cv::imshow("output", sample);

            // Wait for a key press
            char ch = cv::waitKey(0);
            if (ch == 'q') {
                cv::destroyAllWindows();
                break;
            }
        }





        }
    }



    else {

        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FOURCC,1196444237);
        cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 3); //auto
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); //manual
        //cap.set(cv::CAP_PROP_EXPOSURE,25);
        cap.set(cv::CAP_PROP_EXPOSURE,-5);
        cap.set(cv::CAP_PROP_FPS,30);

        if (!cap.isOpened()) {
            std::cerr << "Error opening webcam" << std::endl;
            return -1;
        }

        while (true) {
            cv::Mat frame;
            cap >> frame;

            cv::Mat image = frame.clone();

            // Preprocess the image
            auto [resized_image, scale, left, top] = resize_and_pad(image);
            cv::Mat img = resized_image.clone();
            cv::Mat imgshow = img.clone();
            img = Normalize()(img);

            torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kFloat).permute({0, 3, 1, 2});

            if (half) {
                tensor = tensor.to(torch::kCUDA).to(torch::kHalf); //.unsqueeze(0);
            } else {
                tensor = tensor.to(torch::kCUDA); //.unsqueeze(0);
            }

            // Perform inference
            torch::NoGradGuard no_grad;

            auto start_time = std::chrono::high_resolution_clock::now();
            auto outputs = model.forward({tensor}).toTuple();
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;
            double fps = 1.0 / elapsed_seconds.count();



            torch::Tensor hm = outputs->elements()[0].toTensor().to(torch::kCPU).squeeze(0);
            torch::Tensor offset_tensor = outputs->elements()[1].toTensor().to(torch::kCPU).squeeze(0);
            torch::Tensor wh_tensor = outputs->elements()[2].toTensor().to(torch::kCPU).squeeze(0);
            torch::Tensor angle_tensor = outputs->elements()[3].toTensor().to(torch::kCPU).squeeze(0);


            // Postprocess the outputs
            hm = torch::sigmoid(hm);
            cv::Mat hm_mat(hm.size(1), hm.size(2), CV_32F, hm[0].data_ptr<float>());
            cv::Mat hm_mat_corner(hm.size(1), hm.size(2), CV_32F, hm[1].data_ptr<float>());


            //cv::imshow("heatmap",hm_mat);
            //cv::imshow("heatmap corner",hm_mat_corner);


            hm_mat = select(hm_mat, threshold);



            // Visualize the results
            cv::Mat sample = showbox(imgshow, hm_mat, offset_tensor, wh_tensor, angle_tensor, threshold);
            cv::putText(sample, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            cv::resize(hm_mat_corner,hm_mat_corner,cv::Size(512,512));
            auto kpoints = pred4corner(hm_mat_corner,threshold);

            for (auto element: kpoints) {

                cv::circle(sample,cv::Point(element.x,element.y),5,cv::Scalar(255,255,255),-1);
            }

            cv::imshow("output", sample);

            // Wait for a key press
            char ch = cv::waitKey(1);
            if (ch == 'q') {
                cv::destroyAllWindows();
                break;
            }





        }



    }

    return 0;
}
