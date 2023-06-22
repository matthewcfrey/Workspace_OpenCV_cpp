#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/photo.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdlib.h>     
#include <time.h>
using namespace cv;
using namespace std;

string output_path = "data\\output\\";
string input_path = "data\\input\\";
string trained_classifier_location = "...Documents/libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
string eye_classifier_location = "...Documents/libraries/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
string mouth_classifier_location = "...Documents/libraries/opencv/sources/data/haarcascades/haarcascade_smile.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier smile_cascade;
int num_colors = 0;

/*
- dlib for face location detection
*/

void show_image(Mat img, string window_name = "New Window", int xplace = 0, int yplace = 0) {
    //create window with name, show window, place window
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, img);
    moveWindow(window_name, xplace, yplace);
}

void save_image(Mat img, string end_file_name) {
    string img_path = output_path + end_file_name;
    imwrite(img_path, img);
}

//get half sized matrix
Mat half_mat(Mat big) {
    int rows = big.rows;
    int cols = big.cols;

    if (rows % 2 == 1) { rows++; }
    if (cols % 2 == 1) { cols++; }

    Mat small(rows/2, cols/2, big.type());

    return small;
}

void halve_color_image(Mat big, Mat small) {
    for (int i = 0; i < big.rows; i += 2) {
        for (int j = 0; j < big.cols; j += 2) {
            small.at<Vec3b>(i / 2, j / 2) = big.at<Vec3b>(i, j);
        }
    }
}

//get face matrix
void face_grab(Mat& iwf, Mat& face) {
    face_cascade.load(trained_classifier_location);
    vector<Rect>faces;
    vector<Rect>boundary;
    face_cascade.detectMultiScale(iwf, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(20, 20)); //detectMultiScale(source matrix, vector, searchScaleFactor, minNeighbours, flags, minfeatureSize)
    face = iwf(faces[0]);
}

//find and show face
void face_detect(Mat iwf, string wname = "detect") {
    face_cascade.load(trained_classifier_location);
    vector<Rect>faces;
    vector<Rect>boundary;
    face_cascade.detectMultiScale(iwf, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(20, 20)); //detectMultiScale(source matrix, vector, searchScaleFactor, minNeighbours, flags, minfeatureSize)
    //drawing rectangle around face
    for (size_t i = 0; i < faces.size(); i++) { 
        Mat faceROI = iwf(faces[i]);
        int x = faces[i].x;
        int y = faces[i].y;
        int h = y + faces[i].height;
        int w = x + faces[i].width;
        rectangle(iwf, Point(x, y), Point(w, h), Scalar(255, 0, 255), 2, 8, 0);
    }
    show_image(iwf, wname);//Showing the detected face//
}

//find and show eyes
vector<Point2f> eye_detect(Mat fwe) {
    Mat nfwe = fwe.clone();
    vector<Point2f> eyev;
    face_cascade.load(trained_classifier_location);
    eyes_cascade.load(eye_classifier_location);
    Mat frame_gray;
    cvtColor(fwe, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    Mat faceROI = frame_gray(faces[0]);
    //-- In each face, detect eyes
    std::vector<Rect> eyes;
    eyes_cascade.detectMultiScale(faceROI, eyes);
    for (size_t j = 0; j < eyes.size(); j++)
    {
        Point eye_center(faces[0].x + eyes[j].x + eyes[j].width / 2, faces[0].y + eyes[j].y + eyes[j].height / 2);
        eyev.push_back(eye_center);
        int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
        circle(nfwe, eye_center, radius, Scalar(255, 0, 0), 4);
    }

    imshow("Capture - Face detection", nfwe);
    return eyev;
}

Mat get_bad_image(Mat& good) {
    Mat p1h = half_mat(good);
    Mat p1hh = half_mat(p1h);
    Mat p13h = half_mat(p1hh);
    Mat p14h = half_mat(p13h);
    Mat p15h = half_mat(p14h);
    Mat p16h = half_mat(p15h);

    halve_color_image(good, p1h);
    halve_color_image(p1h, p1hh);
    halve_color_image(p1hh, p13h);
    halve_color_image(p13h, p14h);

    Mat bad(good.size(), good.type());

    resize(p14h, bad, bad.size(), INTER_CUBIC);

    return bad;
}

Point face_center(Mat face, bool show = false) {
    Point center(face.cols / 2, face.rows / 2);
    if (show) {
        Mat nface = face.clone();
        circle(nface, center, 30, Scalar(255, 0, 0), 4);
        show_image(nface, "Center circled");
    }
    return center;
}

void heat_map(Mat face, Mat heat) {
    Mat im_gray;
    cvtColor(face, im_gray, COLOR_BGR2GRAY);
    applyColorMap(im_gray, heat, COLORMAP_JET);
}

void draw_circle_points(Mat& img, vector<Point2f> mc, int rad = 50, string wname = "circle points") {
    Mat nimg = img.clone();
    for (int i = 0; i < mc.size(); i++) {
        circle(nimg, mc[i], 5, Scalar(255, 0, 0), 4);
        circle(nimg, mc[i], rad, Scalar(255, 0, 0), 4);
    }
    show_image(nimg, wname);
}

void draw_circle_point(Mat& img, Point2f mc, int rad = 50, string wname = "circle point") {
    Mat nimg = img.clone();

    circle(nimg, mc, 5, Scalar(255, 0, 0), 4);
    circle(nimg, mc, rad, Scalar(255, 0, 0), 4);
    show_image(nimg, wname);
}

//removing small blobs/noise that isnt desired feature
void remove_small_blobs(vector<vector<Point>> &c, int area = 2000) {
    for (auto it = c.begin(); it != c.end();) {
        if (contourArea(*it) < area) {
            it = c.erase(it);
        }
        else {
            it++;
        }
    }
}

void remove_similar_points(vector<Point2f>& mc) {
    for (auto it = mc.begin(); it != mc.end();it++) {
        (*it).x = floor((*it).x);
        (*it).y = floor((*it).y);
    }

    auto it = std::unique(mc.begin(), mc.end());
    mc.resize(std::distance(mc.begin(), it));
}


void blob_center_points(vector<Moments>& mu, vector<Point2f>& mc, vector<vector<Point>>& contours) {

    //first need moments
    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }

    // center point of figures
    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
    //we get similar contours from our canney process - creating similar centers
    remove_similar_points(mc);

    //sort based on x value
    sort(mc.begin(), mc.end(), [](const Point2f a, const Point2f b) {
        return a.x < b.x;
    });
}

void draw_contours(Mat img, vector<vector<Point>> contours, vector<Vec4i> hierarchy, string wname = "contours") {

    // draw contours
    Mat drawing(img.size(), CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(167, 151, 0); // B G R values
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }

    show_image(drawing, wname);
    save_image(drawing, "contours.jpg");
}

void draw_contour(Mat img, vector<vector<Point>> contours, vector<Vec4i> hierarchy, string wname = "contours") {

    // draw contours
    Mat drawing(img.size(), CV_8UC3, Scalar(255, 255, 255));

    Scalar color = Scalar(167, 151, 0); // B G R values
    drawContours(drawing, contours, 0, color, 2, 8, hierarchy, 0, Point());
    show_image(drawing, wname);
    save_image(drawing, wname + ".jpg");
}

void draw_points(Mat img, vector<Point> ps, string wname = "drawn points") {
    for (auto it = ps.begin(); it != ps.end(); it++) {
        img.at<Vec3b>((*it).y, (*it).x) = Vec3b(255, 0, 0);
    }
    show_image(img, wname);
    save_image(img, wname + ".jpg");
}

void blob_centers_draw(Mat img) {//binary image input
    Mat canny_output;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // detect edges using canny
    Canny(img, canny_output, 50, 150, 3);

    // find contours
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    remove_small_blobs(contours);

    // get the moments and contour centers
    vector<Moments> mu(contours.size());
    vector<Point2f> mc(contours.size());
    blob_center_points(mu, mc, contours);

    // draw contours
    Mat drawing(canny_output.size(), CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(167, 151, 0); // B G R values
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(drawing, mc[i], 4, color, -1, 8, 0);
    }

    show_image(drawing, "blob centers");

}

vector<Point2f> get_eye_centers(Mat& face, vector<int> eadj = {}, int thresh = 150) {

    //create heatmaps, split and use blue channel for features
    Mat fheat(face.size(), face.type());
    heat_map(face, fheat);
    save_image(fheat, "fheat.jpg");

    Mat bgr[3];

    split(fheat, bgr);
    Mat blue = bgr[0];
    
    //threshold for dark blue, find contours and centers
    threshold(blue, blue, thresh, 255, THRESH_BINARY);
    save_image(blue, "bluethresh.jpg");
    Mat canny_output;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // detect edges using canny
    Canny(blue, canny_output, 50, 150, 3);
    save_image(canny_output, "cblue.jpg");
    // find contours, remove small blobs (keep eyes), get center points of blobs
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    Mat frame_gray(blue.size(), blue.type(), Vec3b(0, 0, 0));
    draw_contours(frame_gray, contours, hierarchy);
    remove_small_blobs(contours, 2000);
   
    vector<Moments> mu(contours.size());
    vector<Point2f> mc(contours.size());
    blob_center_points(mu, mc, contours);
    cout << "Eye detection points left: " << mc.size()<< endl;
    if (eadj.size() == 4) {
        mc[0].x += eadj[0];
        mc[0].y += eadj[1];
        mc[1].x += eadj[2];
        mc[1].y += eadj[3];
    }
    //draw_circle_points(face, mc, 40);
    return mc;
}

void black_background(Mat& b, int thresh = 150) {
    Mat img = b.clone();
    Mat frame_gray; 
    cvtColor(img, frame_gray, COLOR_BGR2GRAY);
    cvtColor(img, img, COLOR_BGR2GRAY);
    threshold(frame_gray, frame_gray, thresh, 255, THRESH_BINARY_INV);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(frame_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    remove_small_blobs(contours, 500);

    fillPoly(img, contours[0], Scalar(0, 0, 0));
    threshold(img, img, thresh, 255, THRESH_BINARY);
    for (int r = 0; r < b.rows; r++) {
        for (int c = 0; c < b.cols; c++) {
            if (img.at<uchar>(r, c) == 255) {
                b.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
            }
        }
    }

}

vector<Point> outer_contours(Mat src, int thresh = 10) {
    Mat img = src.clone();
    Mat frame_gray;
    cvtColor(img, frame_gray, COLOR_BGR2GRAY);
    cvtColor(img, img, COLOR_BGR2GRAY);
    threshold(frame_gray, frame_gray, thresh, 255, THRESH_BINARY);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours(frame_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //remove_small_blobs(contours, 500);
    draw_contour(frame_gray, contours, hierarchy, "outerc");

    /*Mat chin = src.clone();
    vector<Point2f> lowc;
    for (int i = 0; i < contours[0].size(); i++) {
        if (contours[0][i].y > chin.rows / 2) {
            lowc.push_back(contours[0][i]);
        }
    }*/
    return contours[0];
}

Point detect_mouth(Mat face, vector<int> adj = {}, bool show = false) {
    Mat image = face.clone();
    Mat image_gray;
    smile_cascade.load(mouth_classifier_location);
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    equalizeHist(image_gray, image_gray);

    std::vector<Rect> smile;
    smile_cascade.detectMultiScale(image_gray, smile, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(80, 80));
    if (show) {
        for (int j = 0; j < smile.size(); j++) {
            Point smile_center(smile[j].x + smile[j].width / 2, smile[j].y + smile[j].height / 2);
            int radius = cvRound((smile[j].width + smile[j].height) * 0.25);
            circle(image, smile_center, radius, Scalar(0, 255, 0), 4, 8, 0);

        }
        imshow("Detected Face", image);
        save_image(image, "mouth.jpg");
    }
    
    Point smile_center(smile[0].x + smile[0].width / 2, smile[0].y + smile[0].height / 2);
    if (adj.size() == 2) {
        smile_center.x += adj[0];
        smile_center.y += adj[1];
    }
    return smile_center;
}

//contours of the bottom of the face
vector<Point> bottom_50_contours(int height, vector<Point> contour) {
    vector<Point> b50;
    for (auto it = contour.begin(); it != contour.end(); it++) {
        if ((*it).y > height / 2) {
            b50.push_back(*it);
        }
    }
    return b50;
}

vector<Point> connect_points(Mat img, vector<Point> ps) {
    vector<Point> con;
    for (auto it = ps.begin(); it+1 != ps.end(); it++) {
        LineIterator lit(img, *it, *(it + 1));
        for (int i = 0; i < lit.count; i++, lit++){
            con.push_back(lit.pos());
        }
    }
    return con;
}

//next point in a line
Point2f next_point(int row, Point2f fp, Point2f sp) {
    double s1 = (sp.y - fp.y) / (sp.x - fp.x);
    Point2f np1((row - sp.y) / s1 + sp.x, row);
    return np1;
}

//using face lines to smooth jagged jaw
vector<Point> face_lines(Mat img, vector<Point> ps, float f = .65, float s = .85) {
    int r1 = img.rows * f;
    int r2 = img.rows * s;
    vector<Point> fp;
    for (auto it = ps.begin(); it != ps.end(); it++) {
        if ((*it).y == r1) {
            fp.push_back(*it);
        }
    }
    for (auto it = ps.begin(); it != ps.end(); it++) {
        if ((*it).y == r2) {
            fp.push_back(*it);
        }
    }
    
    fp.push_back(next_point(img.rows, fp[0], fp[2]));
    fp.push_back(next_point(img.rows, fp[1], fp[3]));

    sort(fp.begin(), fp.end(), [](const Point a, const Point b) {
        return a.x < b.x;
    });

    vector<Point> adjp = connect_points(img, fp);
    return adjp;
}

vector<Vec3b> get_colors2(Mat img, Point center) {
    num_colors = 0;
    vector<Vec3b> dst;
    for (int r = center.x - 20; r < center.x + 20; r++) {
        for (int c = center.y - 20; c < center.y + 20; c++) {
            num_colors++;
            dst.push_back(img.at<Vec3b>(r, c));
        }
    }
    return dst;
}

//moving lines toward middle of face
vector<Point> move_in(vector<Point> ps, int halfcols, int pix = 5) {
    vector<Point> nps;
    for (int i = 0; i < ps.size(); i++) {
        if (ps[i].x < halfcols) {
            nps.push_back(Point(ps[i].x + pix, ps[i].y));
        }
        else {
            nps.push_back(Point(ps[i].x - pix, ps[i].y));
        }
    }

    return nps;
}

//getting darker colors for shading
vector<Vec3b> shade_colors(vector<Vec3b> src, int a = 10) {
    vector<Vec3b> dst;
    for (int i = 0; i < src.size(); i++) {
        dst.push_back(Vec3b(src[i][0] - a, src[i][1] - a, src[i][2] - a));
    }
    return dst;
}

//get the lighter part of the color vector
vector<Vec3b> lighter_half(vector<Vec3b> colors) {
    vector<Vec3b> lhalf;
    sort(colors.begin(), colors.end(), [](const Vec3b a, const Vec3b b) {
        return a[0] + a[1] + a[2] < b[0] + b[1] + b[2];
        });

    for (int i = colors.size() / 2; i < colors.size(); i++) {
        lhalf.push_back(colors[i]);
    }
    return lhalf;
}

//trying to create non jagged place to put face features
void smooth_canvas(Mat& dst, vector<Point> ps, vector<Vec3b> colors, int passes = 1) {
    
    //Mat filler(dst.size(), dst.type());
    Mat filler = dst.clone();
    int halfcols = dst.cols / 2;

    vector<Vec3b> lcolors = lighter_half(colors);

    vector<Point> in1 = move_in(ps, halfcols, 18);
    vector<Point> in2 = move_in(in1, halfcols, 12);
    vector<Point> in3 = move_in(in2, halfcols, 12);
    vector<Point> in4 = move_in(in3, halfcols, 12);
    vector<Vec3b> scolors1 = shade_colors(lcolors, 40);
    vector<Vec3b> scolors2 = shade_colors(lcolors, 30);
    vector<Vec3b> scolors3 = shade_colors(lcolors, 20);
    vector<Vec3b> scolors4 = shade_colors(lcolors, 10);

    for (int i = 0; i < in1.size(); i++) {
        circle(filler, in1[i], 5, scolors1[rand() % lcolors.size()], FILLED, 1);
    }

    for (int i = 0; i < in2.size(); i++) {
        circle(filler, in2[i], 5, scolors2[rand() % lcolors.size()], FILLED, 1);
    }

    for (int i = 0; i < in3.size(); i++) {
        circle(filler, in3[i], 5, scolors3[rand() % lcolors.size()], FILLED, 1);
    }
    for (int i = 0; i < in4.size(); i++) {
        circle(filler, in4[i], 5, scolors4[rand() % lcolors.size()], FILLED, 1);
    }

    for (int i = 0; i < 100; i++) {
        GaussianBlur(filler, filler, Size(3, 3), 0);
    }
    dst = filler;

}


void lefth(Mat& src, Mat& dst) {
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (c<src.cols/2) {
                dst.at<Vec3b>(r, c) = src.at<Vec3b>(r, c);
            }
        }
    }
}

void shift_part(Mat& src, Mat& dst, Point2f p1, Point2f p2) {
    int xc = p2.x - p1.x;
    int yc = p2.y - p1.y;
    cout << p1 << " " << p2 << " " << xc << " " << yc << endl;
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (src.at<Vec3b>(r, c) != Vec3b(0, 0, 0)) {
                dst.at<Vec3b>(r+yc, c+xc) = src.at<Vec3b>(r, c);
            }
        }
    }
}

void get_good_eye(Mat src, Mat dst, vector<Point2f> dstcents) {
    vector<Point2f> eyev = eye_detect(src);
    //draw_circle_points(src, eyev);
    Mat blueeyes(src.size(), src.type(), Vec3b(0, 0, 0));
    circle(blueeyes, eyev[0], 50, Scalar(255, 0, 0), FILLED);
    circle(blueeyes, eyev[1], 50, Scalar(255, 0, 0), FILLED);

    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (blueeyes.at<Vec3b>(r, c) == Vec3b(255, 0, 0)) {
                blueeyes.at<Vec3b>(r, c) = src.at<Vec3b>(r, c);
            }
        }
    }
    save_image(blueeyes, "both.jpg");
    //shift eyes to destination location
    Mat eye1(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    Mat eye2(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    lefth(blueeyes, eye1);
    eye2 = blueeyes - eye1;
    Mat move1(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    Mat move2(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    Mat moved(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    sort(eyev.begin(), eyev.end(), [](const Point2f a, const Point2f b) {
        return a.x < b.x;
        });
    sort(dstcents.begin(), dstcents.end(), [](const Point2f a, const Point2f b) {
        return a.x < b.x;
        });
    shift_part(eye1, move1, eyev[0], dstcents[0]);
    shift_part(eye2, move2, eyev[1], dstcents[1]);

    Mat white1(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (move1.at<Vec3b>(r, c) != Vec3b(0, 0, 0)) {
                white1.at<Vec3b>(r, c) = Vec3b(255, 255, 255);
            }
        }
    }
    Mat white2(blueeyes.size(), blueeyes.type(), Vec3b(0, 0, 0));
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (move2.at<Vec3b>(r, c) != Vec3b(0, 0, 0)) {
                white2.at<Vec3b>(r, c) = Vec3b(255, 255, 255);
            }
        }
    }

    seamlessClone(move1, dst, white1, dstcents[0], dst, NORMAL_CLONE);
    seamlessClone(move2, dst, white2, dstcents[1], dst, NORMAL_CLONE);
}

void get_good_mouth(Mat src, Mat dst, Point2f mcent) {
    Point2f srccent = detect_mouth(src);
    Mat bluemouth(src.size(), src.type(), Vec3b(0, 0, 0));
    circle(bluemouth, srccent, 50, Scalar(255, 0, 0), FILLED);
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (bluemouth.at<Vec3b>(r, c) == Vec3b(255, 0, 0)) {
                bluemouth.at<Vec3b>(r, c) = src.at<Vec3b>(r, c);
            }
        }
    }
    show_image(bluemouth, "bluemouth");
    Mat movem(bluemouth.size(), bluemouth.type(), Vec3b(0, 0, 0));
    shift_part(bluemouth, movem, srccent, mcent);
    show_image(movem, "movem");
    save_image(movem, "movem.jpg");

    Mat white(bluemouth.size(), bluemouth.type(), Vec3b(0, 0, 0));
    for (int r = 0; r < dst.rows; r++) {
        for (int c = 0; c < dst.cols; c++) {
            if (movem.at<Vec3b>(r, c) != Vec3b(0, 0, 0)) {
                white.at<Vec3b>(r, c) = Vec3b(255, 255, 255);
            }
        }
    }
    seamlessClone(movem, dst, white, mcent, dst, NORMAL_CLONE);
}

void print_mse(Mat img1, Mat img2) {
    Mat g1(img1.size(), img1.type());
    Mat g2(img1.size(), img1.type());
    cvtColor(img1, g1, COLOR_BGR2GRAY);
    cvtColor(img2, g2, COLOR_BGR2GRAY);
    Mat diff = g1 - g2;
    int sum = 0;
    for (int r = 0; r < diff.rows; r++) {
        for (int c = 0; c < diff.cols; c++) {
            sum += diff.at<uchar>(r, c) * diff.at<uchar>(r, c);
        }
    }
    double diffsq = static_cast<double>(sum) / (diff.rows * diff.cols);
    cout << "MSE: " << diffsq << endl;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);        // I2^2
    Mat I1_2 = I1.mul(I1);        // I1^2
    Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /***********************PRELIMINARY COMPUTING ******************************/

    Mat mu1, mu2;   //
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

int main()
{
    srand(time(NULL));
/****************************************** preprocess first image ***********************************************************/
    //loading images to program
    string p1_path = input_path + "person1.jpg";
    Mat p1 = imread(p1_path);

    //show_image(p1, "person1.jpg");
    //getting just good face
    Mat orface;
    face_grab(p1, orface);
    //show_image(orface, "orface");

    Mat orfacebb = orface.clone();
    black_background(orfacebb, 220);

    show_image(orfacebb, "black background");
    save_image(orfacebb, "orbb.jpg");

    //creating low quality image
    Mat bp1 = get_bad_image(orfacebb);
    show_image(bp1, "bp1");
    save_image(bp1, "bp.jpg");

/****************************************** get facial reference points *************************************************************************/

    //starting with eye locations, mouth location, and outer contours
    Point fcent = face_center(bp1);
    
    vector<Vec3b> face_colors = get_colors2(bp1, Point(300, 260));
    //define 4 adjustments for the eyes: first horizontal, first vertical, second horizontal, second vertical. in pixels (+/-)
    vector<int> eye_adjustments;
    eye_adjustments.push_back(10);
    eye_adjustments.push_back(16);
    eye_adjustments.push_back(10);
    eye_adjustments.push_back(14);
    vector<Point2f> ecents = get_eye_centers(bp1, eye_adjustments, 180);

    vector<Point> outc = outer_contours(bp1);

    vector<int> mouth_adjustments;
    mouth_adjustments.push_back(-6);
    mouth_adjustments.push_back(-5);
    Point2f mcent = detect_mouth(bp1, mouth_adjustments, true);
    //draw_circle_point(bp1, mcent);



/****************************************** work on eyes **********************************************************************************
    Mat bweyes = bpat.clone();
    get_good_eye(orfacebb, bweyes, ecents);
    show_image(bweyes, "bweyes");

/****************************************** work on mouth **********************************************************************************
    Mat bwmouth = bweyes.clone();
    get_good_mouth(orfacebb, bwmouth, mcent);
    show_image(bwmouth, "bwmouth");
    save_image(bwmouth, "bwmouth.jpg");

    print_mse(orfacebb, bwmouth);
    cout << "PSTN: " << getPSNR(orfacebb, bwmouth) << endl;
    cout << "SSIM: " << getMSSIM(orfacebb, bwmouth) << endl;

/****************************************** cut nose **********************************************************************************/
/****************************************** clean up and shut down program on key press ***********************************************************/
    waitKey(0);
    destroyAllWindows();

    return 0;
}