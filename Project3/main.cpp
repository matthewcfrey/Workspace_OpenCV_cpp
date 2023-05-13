#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using std::vector;
using namespace cv;

std::string output_path = "data\\output\\";
std::string input_path = "data\\input\\";

auto detector = SiftFeatureDetector::create();
auto extractor = SiftDescriptorExtractor::create();

void show_image(Mat img, std::string window_name = "New Window", int xplace = 0, int yplace = 0) {
    //create window with name, show window, place window
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, img);
    moveWindow(window_name, xplace, yplace);
}

void save_image(Mat img, std::string end_file_name) {
    std::string img_path = output_path + end_file_name;
    imwrite(img_path, img);
}

//creating dft made of complex planes
void to_dft(Mat& src, Mat& dst) {
    //need optimal size for dft efficiency 
    Mat padded;
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols); // on the border add zero values
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
    //dft has a real plane and a complex plane
    //creating matrix to hold both planes
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    merge(planes, 2, dst);
    dft(dst, dst);
}

//must shift quadrants in frequency domain
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

//creates gaussain lowpass filter
Mat gaussian_lpf(Mat& scr, float D0)
{
    Mat H(scr.size(), CV_32F, Scalar(1));
    float D = 0;
    for (int u = 0; u < H.rows; u++) {
        for (int v = 0; v < H.cols; v++)
        {
            D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
            H.at<float>(u, v) = exp(-D * D / (2 * D0 * D0));
        }
    }
    return H;
}

//creates gaussain highpass filter
Mat gaussian_hpf(Mat& scr, float D0)
{
    Mat H(scr.size(), CV_32F, Scalar(1));
    Mat HReverse(scr.size(), CV_32F, Scalar(1));
    float D = 0;
    for (int u = 0; u < H.rows; u++) {
        for (int v = 0; v < H.cols; v++)
        {
            D = sqrt((u - scr.rows / 2) * (u - scr.rows / 2) + (v - scr.cols / 2) * (v - scr.cols / 2));
            H.at<float>(u, v) = exp(-D * D / (2 * D0 * D0));
        }
    }
    return HReverse - H;
}

void show_mag(Mat img, std::string wname) {

    Mat shiftimg;
    fftshift(img, shiftimg);

    Mat planes[] = { Mat_<float>(shiftimg), Mat::zeros(shiftimg.size(), CV_32F) };
    split(shiftimg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];
    magI += Scalar::all(1);
    log(magI, magI);

    fftshift(magI, magI);

    normalize(magI, magI, 0, 1, NORM_MINMAX);

    show_image(magI, wname);

    magI.convertTo(magI, CV_8UC1, 255.0);
    save_image(magI, wname + ".jpg");
}

//function to multiply a filter by frequency domain image
void HtimesF(Mat& scr, Mat& dst, Mat& H)
{
    fftshift(H, H);
    //multiply both planes
    Mat planesH[] = { Mat_<float>(H.clone()), Mat_<float>(H.clone()) };

    Mat planes_dft[] = { scr, Mat::zeros(scr.size(), CV_32F) };
    split(scr, planes_dft);

    Mat planes_out[] = { Mat::zeros(scr.size(), CV_32F), Mat::zeros(scr.size(), CV_32F) };
    planes_out[0] = planesH[0].mul(planes_dft[0]);
    planes_out[1] = planesH[1].mul(planes_dft[1]);

    merge(planes_out, 2, dst);

}

void show_dft_mag(Mat img, std::string wname) {

    Mat padded;
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    //need optimal size for dft efficiency 

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat dst;
    merge(planes, 2, dst);
    dft(dst, dst);//
    split(dst, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];
    magI += Scalar::all(1);
    log(magI, magI);

    fftshift(magI, magI);

    normalize(magI, magI, 0, 1, NORM_MINMAX);
    magI.convertTo(magI, CV_8UC1, 255.0);
    show_image(magI, wname);
    save_image(magI, wname + ".jpg");
}

void show_dft_phase(Mat img, std::string wname) {

    //need optimal size for dft efficiency 
    Mat padded;
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat dst;
    merge(planes, 2, dst);
    dft(dst, dst);
    split(dst, planes);
    phase(planes[0], planes[1], planes[1]);
    Mat phaseI = planes[1];
    phaseI += Scalar::all(1);
    log(phaseI, phaseI);

    fftshift(phaseI, phaseI);

    normalize(phaseI, phaseI, 0, 1, NORM_MINMAX);
    phaseI.convertTo(phaseI, CV_8UC1, 255.0);
    show_image(phaseI, wname);
    save_image(phaseI, wname + ".jpg");
}
//invert dft
void inv_dft(Mat& src, Mat& dst) {
    dft(src, dst, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(dst, dst, 0, 1, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC1, 255.0);
}

void do_gaussian_lowpass(Mat& src, Mat& dst, float D0, bool show_H_mag = false, std::string wname = "filter") {
    Mat DFT_image;
    to_dft(src, DFT_image);

    //create gaussian lpf
    Mat H;
    H = gaussian_lpf(DFT_image, D0);

    if (show_H_mag) {
        show_mag(H, wname);
    }

    //multiply lpf times frequency image
    Mat complexIH;
    HtimesF(DFT_image, complexIH, H);
    //invert
    inv_dft(complexIH, dst);
}

void do_gaussian_highpass(Mat& src, Mat& dst, float D0, bool show_H_mag = false, std::string wname = "filter") {
    Mat DFT_image;
    to_dft(src, DFT_image);

    //create gaussian hpf
    Mat H;
    H = gaussian_hpf(DFT_image, D0);

    if (show_H_mag) {
        show_mag(H, wname);
    }

    //multiply lpf times frequency image
    Mat complexIH;
    HtimesF(DFT_image, complexIH, H);
    //invert
    inv_dft(complexIH, dst);
}

void do_ghp_emph(Mat& src, Mat& dst, float D0, bool show_H_mag = false, std::string wname = "filter") {
    Mat DFT_image;
    to_dft(src, DFT_image);

    //create gaussian hpf
    Mat H;
    H = gaussian_hpf(DFT_image, D0);

    float k1 = 1;
    float k2 = 2;

    H = k1 + k2 * H;

    if (show_H_mag) {
        show_mag(H, wname);
    }

    //multiply lpf times frequency image
    Mat complexIH;
    HtimesF(DFT_image, complexIH, H);
    //invert
    inv_dft(complexIH, dst);
}

void do_sift(Mat src, std::string fn) {
    
    vector<KeyPoint>kps;
    detector->detect(src, kps);

    Mat iwk;
    drawKeypoints(src, kps, iwk);
    
    Mat descriptors;
    extractor->compute(src, kps, descriptors);

    show_image(iwk, fn);
    show_image(descriptors, "sift vector");
    save_image(iwk, fn);
}

void do_canny(Mat src, std::string fn, int lowThreshold = 0) {
    int ratio = 3;
    int kernel_size = 3;

    Mat edges, dst;
    blur(src, edges, Size(3, 3));
    Canny(edges, edges, lowThreshold, lowThreshold*ratio, kernel_size);

    dst = Scalar::all(0);

    src.copyTo(dst, edges);
    show_image(dst, fn);
    save_image(dst, fn);
}

void do_fr_feature(Mat src, std::string fn, int low=5 , int high=10, int tlow=115, int thigh=255) {
    //here first we blur,
    //then we take the highpass to see the edges
    //then we equalize the image to exentuate the features
    //then we threshold so we can extract features
    Mat lp;
    do_gaussian_lowpass(src, lp, low);
    //show_image(lp, "lp");
    Mat lphp;
    do_gaussian_highpass(lp, lphp, high);
    //show_image(lphp, "lphp");
    Mat elphp;
    equalizeHist(lphp, elphp);
    //show_image(elphp, "elphp");
    Mat telphp;
    threshold(elphp, telphp, tlow, thigh, THRESH_BINARY);
    show_image(telphp, fn);
    save_image(telphp, fn + ".jpg");
}

int main()
{
    //loading images to program
    std::string ec_path = input_path + "disease\\eczema.jpg";
    Mat ec = imread(ec_path);
    //convert to one channel and cut off text at bottom
    cvtColor(ec, ec, COLOR_BGR2GRAY);
    ec = ec(Range(0, ec.rows - 100), Range(0, ec.cols));
    show_image(ec, "Original eczema.jpg");
    save_image(ec, "ec.jpg");

    std::string mil_path = input_path + "disease\\milia.jpg";
    Mat m = imread(mil_path);
    //convert to one channel and cut off text at bottom
    cvtColor(m, m, COLOR_BGR2GRAY);
    m = m(Range(0, m.rows - 100), Range(0, m.cols));
    show_image(m, "Original milia.jpg");
    save_image(m, "m.jpg");

    std::string p_path = input_path + "disease\\psoriasis.jpg";
    Mat p = imread(p_path);
    //convert to one channel and cut off text at bottom
    cvtColor(p, p, COLOR_BGR2GRAY);
    p = p(Range(0, p.rows - 100), Range(0, p.cols));
    show_image(p, "Original psoriasis.jpg");
    save_image(p, "p.jpg");

    std::string s_path = input_path + "disease\\sacne.jpg";
    Mat s = imread(s_path);
    //convert to one channel and cut off text at bottom
    cvtColor(s, s, COLOR_BGR2GRAY);
    s = s(Range(0, s.rows - 100), Range(0, s.cols));
    show_image(s, "Original steroid acne.jpg");
    save_image(s, "s.jpg");

    std::string t1_path = input_path + "tumor\\t1.jpg";
    Mat t1 = imread(t1_path);
    //convert to one channel and cut off text at bottom
    cvtColor(t1, t1, COLOR_BGR2GRAY);
    t1 = t1(Range(0, t1.rows - 100), Range(0, t1.cols));
    show_image(t1, "Original t1.jpg");
    save_image(t1, "t1.jpg");

    std::string t2_path = input_path + "tumor\\t2.jpg";
    Mat t2 = imread(t2_path);
    //convert to one channel and cut off text at bottom
    cvtColor(t2, t2, COLOR_BGR2GRAY);
    t2 = t2(Range(0, t2.rows - 100), Range(0, t2.cols));
    show_image(t2, "Original t2.jpg");
    save_image(t2, "t2.jpg");

    std::string t3_path = input_path + "tumor\\t3.jpg";
    Mat t3 = imread(t3_path);
    //convert to one channel and cut off text at bottom
    cvtColor(t3, t3, COLOR_BGR2GRAY);
    t3 = t3(Range(0, t3.rows - 100), Range(0, t3.cols));
    show_image(t3, "Original t3.jpg");
    save_image(t3, "t3.jpg");

    std::string t4_path = input_path + "tumor\\t4.jpg";
    Mat t4 = imread(t4_path);
    //convert to one channel and cut off text at bottom
    cvtColor(t4, t4, COLOR_BGR2GRAY);
    t4 = t4(Range(0, t4.rows - 100), Range(0, t4.cols));
    show_image(t4, "Original t4.jpg");
    save_image(t4, "t4.jpg");

/******************************************************************* Part 1.1 *****************************************/
    show_dft_mag(ec, "magec");
    show_dft_phase(ec, "phec");

    show_dft_mag(m, "magmil");
    show_dft_phase(m, "phmil");

    show_dft_mag(p, "magpso");
    show_dft_phase(p, "phpso");

    show_dft_mag(s, "magsa");
    show_dft_phase(s, "phsa");

    show_dft_mag(t1, "magt1");
    show_dft_phase(t1, "pht1");

    show_dft_mag(t2, "magt2");
    show_dft_phase(t2, "pht2");

    show_dft_mag(t3, "magt3");
    show_dft_phase(t3, "pht3");

    show_dft_mag(t4, "magt4");
    show_dft_phase(t4, "pht4");

/******************************************************************* Part 1.2 *****************************************/
    Mat eclpf;
    do_gaussian_lowpass(ec, eclpf, 20, true, "lowpass filter");
    show_image(eclpf, "Eczema LPF");
    save_image(eclpf, "eclpf.jpg");

    Mat echpf;
    do_gaussian_highpass(ec, echpf, 20, true, "highpass filter");
    show_image(echpf, "Eczema HPF");
    save_image(echpf, "echpf.jpg");

    Mat mlpf;
    do_gaussian_lowpass(m, mlpf, 20);
    show_image(mlpf, "Milia LPF");
    save_image(mlpf, "mlpf.jpg");

    Mat mhpf;
    do_gaussian_highpass(m, mhpf, 20);
    show_image(mhpf, "Milia HPF");
    save_image(mhpf, "mhpf.jpg");


    Mat plpf;
    do_gaussian_lowpass(p, plpf, 20);
    show_image(plpf, "pso LPF");
    save_image(plpf, "plpf.jpg");

    Mat phpf;
    do_gaussian_highpass(p, phpf, 20);
    show_image(phpf, "Psoriasis HPF");
    save_image(phpf, "phpf.jpg");

    Mat slpf;
    do_gaussian_lowpass(s, slpf, 20, true);
    show_image(slpf, "S Acne LPF");
    save_image(slpf, "slpf.jpg");

    Mat shpf;
    do_gaussian_highpass(s, shpf, 20, true);
    show_image(shpf, "S Acne HPF");
    save_image(shpf, "shpf.jpg");

    Mat t1lpf;
    do_gaussian_lowpass(t1, t1lpf, 20, true);
    show_image(t1lpf, "t1 LPF");
    save_image(t1lpf, "t1lpf.jpg");

    Mat t1hpf;
    do_gaussian_highpass(t1, t1hpf, 20, true);
    show_image(t1hpf, "t1 HPF");
    save_image(t1hpf, "t1hpf.jpg");

    Mat t2lpf;
    do_gaussian_lowpass(t2, t2lpf, 20, true);
    show_image(t2lpf, "t2 LPF");
    save_image(t2lpf, "t2lpf.jpg");

    Mat t2hpf;
    do_gaussian_highpass(t2, t2hpf, 20);
    show_image(t2hpf, "t2 HPF");
    save_image(t2hpf, "t2hpf.jpg");

    Mat t3lpf;
    do_gaussian_lowpass(t3, t3lpf, 20);
    show_image(t3lpf, "t3 LPF");
    save_image(t3lpf, "t3lpf.jpg");

    Mat t3hpf;
    do_gaussian_highpass(t3, t3hpf, 20);
    show_image(t3hpf, "t3 HPF");
    save_image(t3hpf, "t3hpf.jpg");

    Mat t4lpf;
    do_gaussian_lowpass(t4, t4lpf, 20);
    show_image(t4lpf, "t4 LPF");
    save_image(t4lpf, "t4lpf.jpg");

    Mat t4hpf;
    do_gaussian_highpass(t4, t4hpf, 20);
    show_image(t4hpf, "t4 HPF");
    save_image(t4hpf, "t4hpf.jpg");

/******************************************************************* Part 1.3 *****************************************/
    do_fr_feature(ec, "Eczema Feature Extraction", 10, 5);

    do_fr_feature(m, "Milia Feature Extraction", 100, 5, 20);

    do_fr_feature(p, "Psoriasis Feature Extraction", 100, 100, 15);

    do_fr_feature(s, "Steroid Acne Feature Extraction", 100, 5, 80);

    do_fr_feature(t1, "T1 Feature Extraction", 10, 5, 25);

    do_fr_feature(t2, "T2 Feature Extraction", 10, 5, 25);

    do_fr_feature(t3, "T3 Feature Extraction", 100, 5, 25);

    do_fr_feature(t4, "T4 Feature Extraction", 5, 5, 70);


/******************************************************************* Part 2 *****************************************/

    do_sift(ec, "EczemaSIFT.jpg");

    do_canny(ec, "EczemaCANNY.jpg", 45);

    do_sift(m, "mSIFT.jpg");

    do_canny(m, "mCANNY.jpg", 50);

    do_sift(p, "pSIFT.jpg");

    do_canny(p, "pCANNY.jpg", 30);

    do_sift(s, "sSIFT.jpg");

    do_canny(s, "sCANNY.jpg", 45);

    do_sift(t1, "t1SIFT.jpg");

    do_canny(t1, "t1CANNY.jpg", 30);

    do_sift(t2, "t2SIFT.jpg");

    do_canny(t2, "t2CANNY.jpg", 50);

    do_sift(t3, "t3SIFT.jpg");

    do_canny(t3, "t3CANNY.jpg", 60);

    do_sift(t4, "t4SIFT.jpg");

    do_canny(t4, "t4CANNY.jpg", 45);

    //clean up and shut down program on key press
    waitKey(0);
    destroyAllWindows();

    return 0;
}