#include <opencv4/opencv2/core/cvdef.h>
#include <opencv4/opencv2/core/version.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;


bool validateDimensions(int oldWidth, int oldHeight, int newWidth, int newHeight) {
    return newWidth > 0 && newHeight > 0 && (newWidth <= oldWidth && newHeight <= oldHeight);
}

void normalizeEnergyMatrix(const Mat& energyMatrix, Mat& normalizedMatrix) {
    double minVal, maxVal;

    minMaxLoc(energyMatrix, &minVal, &maxVal);

    normalizedMatrix = Mat::zeros(energyMatrix.size(), CV_8U); 
    for (int i = 0; i < energyMatrix.rows; ++i) {
        for (int j = 0; j < energyMatrix.cols; ++j) {
            normalizedMatrix.at<uchar>(i, j) = static_cast<uchar>(255.0 * (energyMatrix.at<double>(i, j) - minVal) / (maxVal - minVal));
        }
    }
}
void PixelEnergy(Mat& inImage, Mat& energyMatrix) {
    int rows = inImage.rows;
    int cols = inImage.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            Vec3b pixelRight, pixelLeft, pixelDown, pixelUp;

            if (i != rows - 1)
                pixelRight = inImage.at<Vec3b>(i + 1, j);
            else
                pixelRight = inImage.at<Vec3b>(0, j);

            if (i != 0)
                pixelLeft = inImage.at<Vec3b>(i - 1, j);
            else
                pixelLeft = inImage.at<Vec3b>(rows - 1, j);

            if (j != cols - 1)
                pixelDown = inImage.at<Vec3b>(i, j + 1);
            else
                pixelDown = inImage.at<Vec3b>(i, 0);

            if (j != 0)
                pixelUp = inImage.at<Vec3b>(i, j - 1);
            else
                pixelUp = inImage.at<Vec3b>(i, cols - 1);

            double xGradient = 0.0;
            double yGradient = 0.0;

            for (int c = 0; c < 3; ++c) { 
                double diffRightLeft = static_cast<double>(pixelRight[c]) - static_cast<double>(pixelLeft[c]);
                xGradient += diffRightLeft * diffRightLeft;
            }

            for (int c = 0; c < 3; ++c) { 
                double diffDownUp = static_cast<double>(pixelDown[c]) - static_cast<double>(pixelUp[c]);
                yGradient += diffDownUp * diffDownUp;
            }

            double energy = sqrt(xGradient + yGradient);

            energyMatrix.at<double>(i, j) = energy;
        }
    }
}

//---------------------------------------vertical seam------------------------------------
void removeVerticalSeam(Mat& img, const int* seam) {
    int rows = img.rows;
    int cols = img.cols;
    
    for (int x = 0; x < rows; ++x) {
        int seamX = seam[x];
        if (seamX < 0 || seamX >= cols) {
            cerr << "Invalid seam index at row " << x << ": " << seamX << endl;
            return;
        }
        for (int y = seamX; y < cols - 1; ++y) {
            img.at<Vec3b>(x, y) = img.at<Vec3b>(x, y + 1);
        }
}
    
    img = img(Rect(0, 0, cols - 1, rows));
}



void highlightVerticalSeam(Mat& img, int* seam) {
    int rows = img.rows;

    for (int i = 0; i < rows; ++i) {
        img.at<Vec3b>(i, seam[i]) = Vec3b(0, 0, 255); 
    }
}
int findMinSeamIndexv(const Mat& cumEnergy, int row) {
    int minIdx = 0;
    double minValue = cumEnergy.at<double>(row, 0);
    for (int y = 1; y < cumEnergy.cols; ++y) {
        if (cumEnergy.at<double>(row, y) < minValue) {
            minValue = cumEnergy.at<double>(row, y);
            minIdx = y;
        }
    }
    return minIdx;
}
void backtrackSeamv(int* seam, const Mat& cumEnergy, int rows, int cols) {
    for (int x = rows - 2; x >= 0; --x) {
        int prevSeamIdx = seam[x + 1];
        int minIdx = prevSeamIdx;

        if (prevSeamIdx > 0 && cumEnergy.at<double>(x, prevSeamIdx - 1) < cumEnergy.at<double>(x, minIdx)) {
            minIdx = prevSeamIdx - 1;
        }
        if (prevSeamIdx < cols - 1 && cumEnergy.at<double>(x, prevSeamIdx + 1) < cumEnergy.at<double>(x, minIdx)) {
            minIdx = prevSeamIdx + 1;
        }

        seam[x] = minIdx;
    }
}
void findingVerticalSeam(Mat& img, Mat& outImg) {
    int rows = img.rows;
    int cols = img.cols;
    cv::Mat energy(rows, cols, CV_64F);
    
    PixelEnergy(img,energy);
    cv::Mat cumEnergy(rows, cols, CV_64F);
    for (int y = 0; y < cols; ++y) {
        cumEnergy.at<double>(0, y) = energy.at<double>(0, y);
    }

    for (int x = 1; x < rows; ++x) {
        for (int y = 0; y < cols; ++y) {
            double minPrev = cumEnergy.at<double>(x - 1, y);
            if (y > 0) minPrev = min(minPrev, cumEnergy.at<double>(x - 1, y - 1));
            if (y < cols - 1) minPrev = min(minPrev, cumEnergy.at<double>(x - 1, y + 1));
            cumEnergy.at<double>(x, y) = energy.at<double>(x, y) + minPrev;
        }
    }
    int* seam = new int[rows];
    seam[rows - 1] = findMinSeamIndexv(cumEnergy, rows - 1);

    backtrackSeamv(seam, cumEnergy, rows, cols);

    outImg = img.clone();

    highlightVerticalSeam(outImg, seam);

    delete[] seam;
    
}
//-----------------------------------horizontal seam--------------------------------------

void removeHorizontalSeam(Mat& img, const int* seam) {
int rows = img.rows;
    int cols = img.cols;
    
    Mat newImg(rows - 1, cols, img.type());

    for (int y = 0; y < cols; ++y) {
        int seamY = seam[y];
        if (seamY < 0 || seamY >= rows) {
            cerr << "Invalid seam index at col " << y << ": " << seamY << endl;
            return;
        }

        for (int x = 0; x < seamY; ++x) {
            newImg.at<Vec3b>(x, y) = img.at<Vec3b>(x, y);
        }
        for (int x = seamY; x < rows - 1; ++x) {
            newImg.at<Vec3b>(x, y) = img.at<Vec3b>(x + 1, y);
        }
    }

    img = newImg;
}

void highlightHorizontalSeam(Mat& img, int* seam) {
    int cols = img.cols;

    for (int x = 0; x < cols; ++x) {
        int seamRow = seam[x];
        if (seamRow >= 0 && seamRow < img.rows) {
            img.at<Vec3b>(seamRow, x) = Vec3b(0, 0, 255); 
        }
    }
}

int findMinSeamIndexh(const Mat& cumEnergy, int col) {
    int minIdx = 0;
    double minValue = cumEnergy.at<double>(0, col);
    for (int x = 1; x < cumEnergy.rows; ++x) {
        if (cumEnergy.at<double>(x, col) < minValue) {
            minValue = cumEnergy.at<double>(x, col);
            minIdx = x;
        }
    }
    return minIdx;
}

void backtrackSeamh(int* seam, const Mat& cumEnergy, int rows, int cols) {
    for (int x = cols - 2; x >= 0; --x) {
        int prevSeamIdx = seam[x + 1];
        int minIdx = prevSeamIdx;

        if (prevSeamIdx > 0 && cumEnergy.at<double>(prevSeamIdx - 1, x) < cumEnergy.at<double>(minIdx, x)) {
            minIdx = prevSeamIdx - 1;
        }
        if (prevSeamIdx < rows - 1 && cumEnergy.at<double>(prevSeamIdx + 1, x) < cumEnergy.at<double>(minIdx, x)) {
            minIdx = prevSeamIdx + 1;
        }

        seam[x] = minIdx;
    }
}
void findingHorizontalSeam(Mat& img, Mat& outImg) {
    int rows = img.rows;
    int cols = img.cols;

    cv::Mat energy(rows, cols, CV_64F);
    cv::Mat cumEnergy(rows, cols, CV_64F);

    PixelEnergy(img, energy); 

    for (int x = 0; x < cols; ++x) {
        cumEnergy.at<double>(0, x) = energy.at<double>(0, x);
    }

    for (int y = 1; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double minPrev = cumEnergy.at<double>(y - 1, x);
            if (x > 0) minPrev = min(minPrev, cumEnergy.at<double>(y - 1, x - 1));
            if (x < cols - 1) minPrev = min(minPrev, cumEnergy.at<double>(y - 1, x + 1));
            cumEnergy.at<double>(y, x) = energy.at<double>(y, x) + minPrev;
        }
    }

    int* seam = new int[cols];
    seam[cols - 1] = findMinSeamIndexh(cumEnergy, cols - 1);

    backtrackSeamh(seam, cumEnergy, rows, cols);

    outImg = img.clone();

    highlightHorizontalSeam(outImg, seam);

    delete[] seam;
}

void seamCarving(Mat& img, Mat& outImage, int newWidth, int newHeight) {
    int cols = img.cols;
    int rows = img.rows;
    while (cols > newWidth) {
        Mat energy(rows, cols, CV_64F);
        Mat cumEnergy(rows, cols, CV_64F);

        PixelEnergy(img, energy);

        for (int y = 0; y < cols; ++y) {
            cumEnergy.at<double>(0, y) = energy.at<double>(0, y);
        }

        for (int x = 1; x < rows; ++x) {
            for (int y = 0; y < cols; ++y) {
                double minPrev = cumEnergy.at<double>(x - 1, y);
                if (y > 0) minPrev = min(minPrev, cumEnergy.at<double>(x - 1, y - 1));
                if (y < cols - 1) minPrev = min(minPrev, cumEnergy.at<double>(x - 1, y + 1));
                cumEnergy.at<double>(x, y) = energy.at<double>(x, y) + minPrev;
            }
        }

        int* seam = new int[rows];
        seam[rows - 1] = findMinSeamIndexv(cumEnergy, rows - 1);

        backtrackSeamv(seam, cumEnergy, rows, cols);

        Mat seamImage = img.clone();
        highlightVerticalSeam(seamImage, seam);
        imshow("Vertical Seam", seamImage);
        waitKey(40); 

        removeVerticalSeam(img, seam);

        cols = img.cols;

        delete[] seam;
    }
    while (rows > newHeight) {
        Mat energy(rows, cols, CV_64F);
Mat cumEnergy(rows, cols, CV_64F);

        PixelEnergy(img, energy);

        for (int x = 0; x < rows; ++x) {
            cumEnergy.at<double>(x, 0) = energy.at<double>(x, 0);
        }

        for (int y = 1; y < cols; ++y) {
            for (int x = 0; x < rows; ++x) {
                double minPrev = cumEnergy.at<double>(x, y - 1);
                if (x > 0) minPrev = min(minPrev, cumEnergy.at<double>(x - 1, y - 1));
                if (x < rows - 1) minPrev = min(minPrev, cumEnergy.at<double>(x + 1, y - 1));
                cumEnergy.at<double>(x, y) = energy.at<double>(x, y) + minPrev;
            }
        }

        int* seam = new int[cols];
        seam[cols - 1] = findMinSeamIndexh(cumEnergy, cols - 1);

        backtrackSeamh(seam, cumEnergy, rows, cols);

        Mat seamImage = img.clone();
        highlightHorizontalSeam(seamImage, seam);
        imshow("Horizontal Seam", seamImage);
        waitKey(40);

        removeHorizontalSeam(img, seam);

        rows = img.rows;

        delete[] seam;
    }
    
    imshow("Final Image", img);
    imwrite("output_image_final.jpg", img);
    waitKey(0);
}



int main(int argc, char** argv) {

    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <input_image> " << endl;
        return -1;
    }

    Mat inputImage = imread(argv[1], IMREAD_COLOR);
    if (inputImage.empty()) {
        cout << "Could not load input image!" << endl;
        return -1;
    }

    if (inputImage.channels() != 3) {
        cout << "Image does not have 3 channels!" << endl;
        return -1;
    }
    
    int Width = inputImage.cols;
    int Height = inputImage.rows;
    int newWidth;
    int newHeight;


    cout << "Image loaded successfully!" << endl;
    cout << "Width: " << Width << ", Height: " << Height << endl;

    cout<<"Enter new width and height: ";
    cin>>newWidth>>newHeight;



    if (!validateDimensions(inputImage.cols, inputImage.rows, newWidth, newHeight)) {
        cout << "Invalid dimensions!" << endl;
        return -1;
    }

    Mat outputImage ;
    seamCarving(inputImage,outputImage,newWidth,newHeight);

    return 0;
}
