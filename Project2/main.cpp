#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using std::vector;
using namespace cv;

std::string output_folder_path = "Data\\output\\";

/*********************************************************************************************************************************************
*                                                            FUNCTONS                                                                        *
**********************************************************************************************************************************************/
void show_image(Mat img, std::string window_name = "New Window", int xplace = 0, int yplace = 0) {
    //create window with name, show window, place window
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, img);
    moveWindow(window_name, xplace, yplace);
}

void save_image(Mat img, std::string end_file_name) {
    std::string img_path = output_folder_path + end_file_name;
    imwrite(img_path, img);
}

void draw_histogram(vector<Mat> bgr, int histSize, int hist_h, int bin_w, Mat histImage) {
    //line draws a line between two points. Once for each channel. Parameters:
    /*
    img
    pt1
    pt2
    color
    thickness
    lineType
    shift
    */
    for (int i = 1; i < histSize; i++)
    {
        //subtracting bgr values from height because measuring from top of image
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(bgr[0].at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(bgr[0].at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(bgr[1].at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(bgr[1].at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(bgr[2].at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(bgr[2].at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }
}

Mat draw_color_histogram(Mat img) {
    //following the histogram documentation from opencv.org
//parameters for histogram calculation, for reference:
/*
- The source array(s)
- The number of source arrays (in this case we are using 1. We can enter here also a list of arrays )
- The channel (dim) to be measured.
- A mask to be used on the source array ( zeros indicating pixels to be ignored ). If not defined it is not used
- destination Mat
- The histogram dimensionality.
- histSize
- histRange
- uniform: The bin sizes are the same
- accumulate: the histogram is cleared at the beginning.
*/
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true, acc = false;

    Mat hist0, hist1, hist2;

    //3 image color channels
    vector<Mat> bgr;
    split(img, bgr);

    //calc hist returns a matrix
    calcHist(&bgr[0], 1, 0, Mat(), hist0, 1, &histSize, &histRange, uniform, acc);
    calcHist(&bgr[1], 1, 0, Mat(), hist1, 1, &histSize, &histRange, uniform, acc);
    calcHist(&bgr[2], 1, 0, Mat(), hist2, 1, &histSize, &histRange, uniform, acc);
    //std::cout << "Blue Channel Hist" << std::endl << hist0 << std::endl;
    //std::cout << "Green Channel Hist" << std::endl << hist1 << std::endl;
    //std::cout << "Red Channel Hist" << std::endl << hist2 << std::endl;

    //creating matrix to display histogram on
    //histogram width and height
    int hist_w = 512, hist_h = 400;
    //width of histogram bins
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    //normalize the histogram values to the specified ranges
    normalize(hist0, hist0, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(hist1, hist1, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(hist2, hist2, 0, histImage.rows, NORM_MINMAX, -1, Mat());


    vector<Mat> bgr_hist = { hist0, hist1, hist2 };

    //drawHistogram function
    //takes vector of normailzed histogram channels bgr, histogram size, histogram height, bin width, destination histogram image
    draw_histogram(bgr_hist, histSize, hist_h, bin_w, histImage);
    return histImage;
}

Mat equalize_color_image(Mat img) {
    vector<Mat> bgr;
    split(img, bgr);

    equalizeHist(bgr[0], bgr[0]);
    equalizeHist(bgr[1], bgr[1]);
    equalizeHist(bgr[2], bgr[2]);

    Mat eq(bgr[0]);
    merge(bgr, eq);
    return eq;
}

vector<int> channel_color_steps(Mat channel, int num_steps) {
    vector<int> step_colors;
    if (num_steps < 0 || num_steps > 256) { return step_colors; }

    int bins_per_step = ceil(256. / num_steps);

    //first get the most common color at each step
    int hi_num;
    int hi_color;
    for (int i = 0; i < num_steps; i++) {
        hi_num = 0;
        hi_color = 0;
        for (int j = i * bins_per_step; (j < i * bins_per_step + bins_per_step) && j<256; j++) {
            if (channel.at<float>(0, j) > hi_num) {
                hi_num = channel.at<float>(0, j);
                hi_color = j;
            }
        }
        step_colors.push_back(hi_color);
    }
    //next create a vector mapping the desired color to each index
    vector<int> color_vec;
    for (int i = 0; i < num_steps; i++) {
        for (int j = i * bins_per_step; (j < i * bins_per_step + bins_per_step) && j < 256; j++) {
            color_vec.push_back(step_colors[i]);
        }
    }
    return color_vec;
}

void three_hist_channels(Mat img, vector<Mat>& vec) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true, acc = false;

    Mat hist0, hist1, hist2;

    //3 image color channels
    vector<Mat> bgr;
    split(img, bgr);

    //calc hist returns a matrix
    calcHist(&bgr[0], 1, 0, Mat(), hist0, 1, &histSize, &histRange, uniform, acc);
    calcHist(&bgr[1], 1, 0, Mat(), hist1, 1, &histSize, &histRange, uniform, acc);
    calcHist(&bgr[2], 1, 0, Mat(), hist2, 1, &histSize, &histRange, uniform, acc);

    vec.push_back(hist0);
    vec.push_back(hist1);
    vec.push_back(hist2);
}

//modifies an image to have only the most common colors at each step
void step_image(Mat img, int num_steps) {

    //getting three histogram channels for  image
    vector<Mat> hist_channels;
    three_hist_channels(img, hist_channels);

    //calculating most common color at each step for each channel
    //steps are number of bits divided by a chosen number
    vector<int> b_steps = channel_color_steps(hist_channels[0], num_steps);
    vector<int> g_steps = channel_color_steps(hist_channels[1], num_steps);
    vector<int> r_steps = channel_color_steps(hist_channels[2], num_steps);

    MatIterator_<Vec3b> it, end;
    for (it = img.begin<Vec3b>(), end = img.end<Vec3b>(); it != end; ++it) {
        (*it)[0] = b_steps[(*it)[0]];
        (*it)[1] = g_steps[(*it)[1]];
        (*it)[2] = r_steps[(*it)[2]];
    }
}

void remove_perspective(Mat img1, Mat img2, vector<Point2f> src, vector<Point2f> dst) {
    Mat h = findHomography(src, dst);

    warpPerspective(img1, img2, h, img1.size());
}

void remove_perspective_ransac(Mat img1, Mat img2, vector<Point2f> src, vector<Point2f> dst) {
    Mat h = findHomography(src, dst, RANSAC);

    warpPerspective(img1, img2, h, img1.size());
}

void draw_grid(Mat img, int size) {

    int width = img.size().width;
    int height = img.size().height;

    for (int i = 0; i < height; i += size) {
        line(img, Point(0, i), Point(width, i), cv::Scalar(0, 0,0));
    }
        
    for (int i = 0; i < width; i += size) {
        line(img, Point(i, 0), Point(i, height), cv::Scalar(0,0,0));
    }
        
}

void pers_map_image(Mat img1, Mat img2, Point2f c1, Point2f c2, Point2f c3, Point2f c4) {
    //this function maps a section of an image onto a section of another image
    // uses four points
    //images are of same size
    //distances must be calculated
    double leftx = c3.x - c1.x;
    double lefty = c3.y - c1.y;
    double rightx = c4.x - c2.x;
    double righty = c4.y - c2.y;
    double topx = c2.x - c1.x;
    double topy = c2.y - c1.y;
    double bottomx = c4.x - c3.x;
    double bottomy = c4.y - c3.y;
    //variables to follow the slope of the image edges
    //used to calculate the change in column per row, and the change in row per column respectively
    double deltaL;
    double deltaR;
    double deltaT;
    double deltaB;
    //drawing between the lines
    for (int r = 0; r < img2.rows; r++) {
        for (int c = 0; c < img2.cols; c++) {
            deltaL = (leftx / lefty) * (r - c1.y);
            deltaR = (rightx / righty) * (r - c2.y);
            deltaT = (topy / topx) * (c - c1.x);
            deltaB = (bottomy / bottomx) * (c - c3.x);
            if (c > c1.x + deltaL && c <= c2.x + deltaR) {
                if (r > c1.y + deltaT && r <= c3.y + deltaB) {
                    img2.at<Vec3b>(r, c) = img1.at<Vec3b>(r, c);
                }
            }
        }
    }
}

//didnt end up using this function, effect wasnt any better
void put_non_zero(Mat img1, Mat img2) {
    for (int r = 0; r < img2.rows; r++) {
        for (int c = 0; c < img2.cols; c++) {
            if (img1.at<Vec3b>(r,c) != Vec3b(0,0,0)) {
                img2.at<Vec3b>(r, c) = img1.at<Vec3b>(r, c);
            }
        }
    }
}

int main()
{
/******************************************************* PART 1 ******************************************************/
/******************************************************* 1.1 ******************************************************/
    //loading images to program
    std::string adams_path = "Data\\input\\adams.jpg";
    Mat adams = imread(adams_path);
    std::string monitor1_path = "Data\\input\\monitor1.png";
    Mat monitor1 = imread(monitor1_path);

    show_image(adams, "Original adams.jpg");

    show_image(monitor1, "Original monitor1.png", 0, adams.rows / 2);

    //draw original histograms
    Mat adamsHistImage = draw_color_histogram(adams);
    show_image(adamsHistImage, "Original adams Histogram");
    save_image(adamsHistImage, "OriginalAdamsHist.jpg");

    Mat monitor1HistImage = draw_color_histogram(monitor1);
    show_image(monitor1HistImage, "Original monitor1 Histogram", 0, adams.rows/2);
    save_image(monitor1HistImage, "monitor1HistImage.jpg");

    //histogram equalization
    Mat adams_eq = equalize_color_image(adams);
    show_image(adams_eq, "Equalized adams", adams.cols);
    save_image(adams_eq, "adams_eq.jpg");

    Mat monitor1_eq = equalize_color_image(monitor1);
    show_image(monitor1_eq, "Equalized monitor1", adams.cols, adams.rows/2);
    save_image(monitor1_eq, "monitor1_eq.jpg");

    //draw new histograms
    Mat adams_eqHistImage = draw_color_histogram(adams_eq);
    show_image(adams_eqHistImage, "Equalized adams Histogram", adams.cols);
    save_image(adams_eqHistImage, "adams_eqHistImage.jpg");

    Mat monitor1_eqHistImage = draw_color_histogram(monitor1_eq);
    show_image(monitor1_eqHistImage, "Equalized monitor1 Histogram", adams.cols, adams.rows / 2);
    save_image(monitor1_eqHistImage, "monitor1_eqHistImage.jpg");

/******************************************************* 1.2 ******************************************************/
    //here we are writing functions that determine a specified number of most common colors in the image and 
    //use only those colors to draw the image. 

    //image to contain stepped image
    Mat stepped_adams = adams.clone();

    //step image function, takes an image to steppify, and number of steps
    step_image(stepped_adams, 4);

    //showing image and histogram of stepped images
    show_image(stepped_adams, "Stepped adams.jpg", adams.cols);
    Mat stepped_adamsHistImage = draw_color_histogram(stepped_adams);
    show_image(stepped_adamsHistImage, "Stepped adams Histogram", adams.cols, adams.rows/2);
    save_image(stepped_adams, "stepped_adams.jpg");
    save_image(stepped_adamsHistImage, "stepped_adamsHistImage.jpg");

    Mat stepped_monitor1 = monitor1.clone();

    //step image function, takes an image to steppify, and number of steps
    step_image(stepped_monitor1, 10);

    //showing image and histogram of stepped images
    show_image(stepped_monitor1, "Stepped monitor1.jpg", adams.cols, adams.rows/2);
    Mat stepped_monitor1HistImage = draw_color_histogram(stepped_monitor1);
    show_image(stepped_monitor1HistImage, "Stepped monitor1 Histogram", adams.cols, adams.rows);
    save_image(stepped_monitor1, "stepped_monitor1.jpg");
    save_image(stepped_monitor1HistImage, "stepped_monitor1HistImage.jpg");

/******************************************************* PART 2 ******************************************************/
/******************************************************* 2.1 ******************************************************/
    std::string door_path = "Data\\input\\door.jpg";
    Mat door = imread(door_path);
    std::string monitor2_path = "Data\\input\\monitor2.jpg";
    Mat monitor2 = imread(monitor2_path);
    std::string poker_path = "Data\\input\\poker.jpg";
    Mat poker = imread(poker_path);
    std::string book_path = "Data\\input\\book.jpg";
    Mat book = imread(book_path);
    
    //resizing door because I cant see it on my screen
    resize(door, door, Size(door.cols/4, door.rows/4), INTER_LINEAR);
    
    show_image(door, "Original door.jpg");
    show_image(monitor2, "Original monitor2.jpg");
    show_image(poker, "Original poker.jpg");
    show_image(book, "Original book.jpg");

    //remove monitor perspective
    //grid for finding points on original image
    //also can be done in image editor, so commented out
    //Mat g_monitor2 = monitor2.clone();
    //draw_grid(g_monitor2, 25);
    //show_image(g_monitor2, "Grided Monitor2");

    Mat p_monitor2 = monitor2.clone();
    vector<Point2f> m2corn1, m2corn2;

    //finding four points that should be a square
    m2corn1.push_back(Point2f(161, 173));
    m2corn1.push_back(Point2f(610, 177));
    m2corn1.push_back(Point2f(140, 590));
    m2corn1.push_back(Point2f(555, 450));

    //making a square
    //slightly stretching by picking long x coords
    //averaging the y coords
    m2corn2.push_back(Point2f(140, 175));
    m2corn2.push_back(Point2f(610, 175));
    m2corn2.push_back(Point2f(140, 525));
    m2corn2.push_back(Point2f(610, 525));

    remove_perspective(monitor2, p_monitor2, m2corn1, m2corn2);
    show_image(p_monitor2, "Pers Monitor2");
    save_image(p_monitor2, "p_monitor2.jpg");

    //removing door perspective
    Mat pdoor = door.clone();
    vector<Point2f> doorcorn1, doorcorn2;

    doorcorn1.push_back(Point2f(601, 20));
    doorcorn1.push_back(Point2f(753, 55));
    doorcorn1.push_back(Point2f(580, 670));
    doorcorn1.push_back(Point2f(725, 600));

    doorcorn2.push_back(Point2f(540, 37));
    doorcorn2.push_back(Point2f(753, 37));
    doorcorn2.push_back(Point2f(540, 635));
    doorcorn2.push_back(Point2f(753, 660));

    remove_perspective(door, pdoor, doorcorn1, doorcorn2);
    show_image(pdoor, "Pers Door");
    save_image(pdoor, "pdoor.jpg");

    //removing adams perspective
    Mat padams = adams.clone();
    vector<Point2f> acorn1, acorn2;

    acorn1.push_back(Point2f(27, 185));
    acorn1.push_back(Point2f(145, 175));
    acorn1.push_back(Point2f(47, 430));
    acorn1.push_back(Point2f(152, 453));

    acorn2.push_back(Point2f(20, 177));
    acorn2.push_back(Point2f(160, 177));
    acorn2.push_back(Point2f(20, 441));
    acorn2.push_back(Point2f(160, 441));

    remove_perspective(adams, padams, acorn1, acorn2);
    show_image(padams, "Pers adams");
    save_image(padams, "padams.jpg");

    //removing poker perspective
    Mat ppoker = poker.clone();
    vector<Point2f> pcorn1, pcorn2;

    pcorn1.push_back(Point2f(50, 298));
    pcorn1.push_back(Point2f(150, 250));
    pcorn1.push_back(Point2f(80, 350));
    pcorn1.push_back(Point2f(200, 300));

    pcorn2.push_back(Point2f(50, 274));
    pcorn2.push_back(Point2f(150, 274));
    pcorn2.push_back(Point2f(50, 325));
    pcorn2.push_back(Point2f(150, 325));

    remove_perspective(poker, ppoker, pcorn1, pcorn2);
    show_image(ppoker, "Pers poker");
    save_image(ppoker, "ppoker.jpg");

    //removing book perspective
    Mat pbook = book.clone();
    vector<Point2f> bcorn1, bcorn2;

    bcorn1.push_back(Point2f(170, 120));
    bcorn1.push_back(Point2f(405, 105));
    bcorn1.push_back(Point2f(20, 420));
    bcorn1.push_back(Point2f(225, 550));

    bcorn2.push_back(Point2f(10, 112));
    bcorn2.push_back(Point2f(240, 112));
    bcorn2.push_back(Point2f(10, 485));
    bcorn2.push_back(Point2f(240, 485));

    remove_perspective(book, pbook, bcorn1, bcorn2);
    show_image(pbook, "Pers book");
    save_image(pbook, "pbook.jpg");

/******************************************************* 2.2 ******************************************************/

    Mat sign_door = door.clone();
    Mat sign_door2 = door.clone();
    std::string g1_path = "Data\\input\\graphic1.png";
    Mat g1 = imread(g1_path);
    std::string g2_path = "Data\\input\\graphic2.png";
    Mat g2 = imread(g2_path);
    std::string g3_path = "Data\\input\\graphic3.png";
    Mat g3 = imread(g3_path);

    //creating point vectors for homography
    vector<Point2f> g1corn1, g1corn2, g2corn1, g2corn2, g3corn1, g3corn2;

    //found it better to start a little inside the image to be projected
    //original g1 points
    g1corn1.push_back(Point2f(5, 5));
    g1corn1.push_back(Point2f(g1.cols-6, 5));
    g1corn1.push_back(Point2f(5, g1.rows-6));
    g1corn1.push_back(Point2f(g1.cols-6, g1.rows-6));

    //original g2 points
    g2corn1.push_back(Point2f(5, 5));
    g2corn1.push_back(Point2f(g2.cols - 6, 5));
    g2corn1.push_back(Point2f(5, g2.rows - 6));
    g2corn1.push_back(Point2f(g2.cols - 6, g2.rows - 6));

    //original g3 points
    g3corn1.push_back(Point2f(5, 5));
    g3corn1.push_back(Point2f(g3.cols - 6, 5));
    g3corn1.push_back(Point2f(5, g3.rows - 6));
    g3corn1.push_back(Point2f(g3.cols - 6, g3.rows - 6));
    
    //points to map to on final image need to be used a few times
    //points for g1
    Point2f c1(267, 44);
    Point2f c2(465, 75);
    Point2f c3(298, 684);
    Point2f c4(467, 620);

    //points for g2
    Point2f c5(600, 38);
    Point2f c6(730, 71);
    Point2f c7(587, 452);
    Point2f c8(708, 425);

    //points for g3
    Point2f c9(586, 487);
    Point2f c10(708, 453);
    Point2f c11(582, 645);
    Point2f c12(702, 590);

    g1corn2.push_back(c1);
    g1corn2.push_back(c2);
    g1corn2.push_back(c3);
    g1corn2.push_back(c4);

    g2corn2.push_back(c5);
    g2corn2.push_back(c6);
    g2corn2.push_back(c7);
    g2corn2.push_back(c8);

    g3corn2.push_back(c9);
    g3corn2.push_back(c10);
    g3corn2.push_back(c11);
    g3corn2.push_back(c12);

    Mat h1 = findHomography(g1corn1, g1corn2);

    Mat h2 = findHomography(g2corn1, g2corn2);

    Mat h3 = findHomography(g3corn1, g3corn2);

    //creating image of final size with warped graphics
    Mat g1DoorSize = sign_door.clone();
    warpPerspective(g1, g1DoorSize, h1, sign_door.size());
    show_image(g1DoorSize, "g1 door size");

    Mat g2DoorSize = sign_door.clone();
    warpPerspective(g2, g2DoorSize, h2, sign_door.size());
    show_image(g2DoorSize, "g2 door size");

    Mat g3DoorSize = sign_door.clone();
    warpPerspective(g3, g3DoorSize, h3, sign_door.size());
    show_image(g3DoorSize, "g3 door size");

    //calling function that maps warped graphic onto final image
    pers_map_image(g1DoorSize, sign_door, c1, c2, c3, c4);
    pers_map_image(g2DoorSize, sign_door, c5, c6, c7, c8);
    pers_map_image(g3DoorSize, sign_door, c9, c10, c11, c12);

    show_image(sign_door, "pers sign door");
    save_image(sign_door, "sign_door.jpg");



/******************************************************* 2.3 ******************************************************/
    //remove monitor perspective with more points
    //grid for finding points on original image

    Mat manyp = monitor2.clone();
    vector<Point2f> manyp1, manyp2;

    //finding four points that should be a square
    //screen
    manyp1.push_back(Point2f(164, 171));
    manyp1.push_back(Point2f(418, 175));
    manyp1.push_back(Point2f(609, 179));
    manyp1.push_back(Point2f(143, 593));
    manyp1.push_back(Point2f(401, 507));
    manyp1.push_back(Point2f(558, 449));
    //stand
    manyp1.push_back(Point2f(260, 643));
    manyp1.push_back(Point2f(294, 642));
    manyp1.push_back(Point2f(379, 520));
    manyp1.push_back(Point2f(338, 555));
    manyp1.push_back(Point2f(573, 448));

    //screen
    manyp2.push_back(Point2f(140, 172));
    manyp2.push_back(Point2f(416, 172));
    manyp2.push_back(Point2f(692, 172));
    manyp2.push_back(Point2f(140, 530));
    manyp2.push_back(Point2f(416, 530));
    manyp2.push_back(Point2f(692, 530));
    //stand
    manyp2.push_back(Point2f(238, 578));
    manyp2.push_back(Point2f(253, 590));
    manyp2.push_back(Point2f(450, 538));
    manyp2.push_back(Point2f(415, 564));
    manyp2.push_back(Point2f(655, 544));


    remove_perspective(monitor2, manyp, manyp1, manyp2);
    show_image(manyp, "Many points remove perspective");
    save_image(manyp, "manypoints.jpg");

    Mat manypoker = poker.clone();
    vector<Point2f> manypoker1, manypoker2;

    manypoker1.push_back(Point2f(50, 298));
    manypoker1.push_back(Point2f(150, 250));
    manypoker1.push_back(Point2f(80, 350));
    manypoker1.push_back(Point2f(200, 300));
    manypoker1.push_back(Point2f(193, 19));
    manypoker1.push_back(Point2f(74, 177));

    manypoker2.push_back(Point2f(50, 274));
    manypoker2.push_back(Point2f(150, 274));
    manypoker2.push_back(Point2f(50, 325));
    manypoker2.push_back(Point2f(150, 325));
    manypoker2.push_back(Point2f(364, 14));
    manypoker2.push_back(Point2f(140, 183));

    remove_perspective(poker, manypoker, manypoker1, manypoker2);
    show_image(manypoker, "manypoker");
    save_image(manypoker, "manypoker.jpg");
/*************************************************************************************** END ******************************************/
    //clean up and shut down program on key press
    waitKey(0);
    destroyAllWindows();

    return 0;

}