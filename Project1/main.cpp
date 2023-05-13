#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
using std::max;
using std::vector;
using namespace cv;

//folder defaults for images
//alternatively, command line input
//change this to save images to desired folder
std::string output_folder_path = "Data\\output\\";
//change this to input desired photo
std::string input_file_path = "Data\\input\\KSUSelfie.jpg";

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

//loop over every other pixel of original image, set to every pixel of half image
void halve_color_image(Mat big, Mat small) {
    for (int i = 0; i < big.rows; i += 2) {
        for (int j = 0; j < big.cols; j += 2) {
            small.at<Vec3b>(i / 2, j / 2) = big.at<Vec3b>(i, j);
        }
    }
}

//doubling each pixel in horizontal and vertical direction
//actually making 4 total copies to complete each square. 
void double_color_image(Mat small, Mat big) {
    for (int i = 0; i < big.rows; i++) {
        for (int j = 0; j < big.cols; j++) {
            big.at<Vec3b>(i, j) = small.at<Vec3b>(floor(i / 2), floor(j / 2));
        }
    }
}

//function for blending pixels by averaging surounding pixels
void blend_color_pixels(Mat img, int input_reach=3) {
    int sum0;
    int sum1;
    int sum2;
    int reach = input_reach;
    int reachsq = reach * reach;
    //loop through most pixels, not overreaching the edge of the image
    for (int i = 0; i < img.rows - reach; i++) {
        for (int j = 0; j < img.cols - reach; j++) {
            sum0 = 0;
            sum1 = 0;
            sum2 = 0;

            //sum surrounding channels
            for (int k = i; k < i + reach; k++) {
                for (int l = j; l < j + reach; l++) {
                    sum0 += img.at<Vec3b>(k, j)[0];
                    sum1 += img.at<Vec3b>(k, j)[1];
                    sum2 += img.at<Vec3b>(k, j)[2];
                }
            }
            //average the surrounding channels and set to current pixel
            img.at<Vec3b>(i, j)[0] = sum0 / (reachsq);
            img.at<Vec3b>(i, j)[1] = sum1 / (reachsq);
            img.at<Vec3b>(i, j)[2] = sum2 / (reachsq);
        }
    }
}
void blend_color_pixels_back(Mat img, int input_reach = 3) {
    int sum0;
    int sum1;
    int sum2;
    int reach = input_reach;
    int reachsq = reach * reach;
    //loop through most pixels, not overreaching the edge of the image
    for (int i = 0; i < img.rows - reach; i++) {
        for (int j = img.cols-1; j > reach; j--) {
            sum0 = 0;
            sum1 = 0;
            sum2 = 0;

            //sum surrounding channels
            for (int k = i; k < i + reach; k++) {
                for (int l = j; l > j - reach; l--) {
                    sum0 += img.at<Vec3b>(k, j)[0];
                    sum1 += img.at<Vec3b>(k, j)[1];
                    sum2 += img.at<Vec3b>(k, j)[2];
                }
            }
            //average the surrounding channels and set to current pixel
            img.at<Vec3b>(i, j)[0] = sum0 / (reachsq);
            img.at<Vec3b>(i, j)[1] = sum1 / (reachsq);
            img.at<Vec3b>(i, j)[2] = sum2 / (reachsq);
        }
    }
}

void crude_fit(Mat original, Mat resized) {
    for (int i = 0; i < resized.rows; i++) {
        for (int j = 0; j < resized.cols; j++) {
            if (i < original.rows && j < original.cols) {
                resized.at<Vec3b>(i, j) = original.at<Vec3b>(i, j);
            }
        }
    }
}

struct {
    Vec3b red = Vec3b(8, 8, 189);
    Vec3b blue = Vec3b(255, 0, 3);
    Vec3b yellow = Vec3b(0, 255, 232);
    Vec3b skin = Vec3b(57, 167, 217);
    Vec3b brown = Vec3b(27, 70, 128);
    Vec3b black = Vec3b(0, 0, 0);
} mc;

//the following functions each draw a mario line
//added in a little black outline by hand
//could be reworked to allow for stretching or other features
void draw_mario_1(Mat img, int i, int j) {
    for (int k = j + 3; k < j + 8; k++) {
        img.at<Vec3b>(i, k) = mc.red;
        img.at<Vec3b>(i-1, k) = mc.black;
    }
    img.at<Vec3b>(i, j + 2) = mc.black;
    img.at<Vec3b>(i, j + 8) = mc.black;
}
void draw_mario_2(Mat img, int i, int j) {
    for (int k = j + 2; k < j + 11; k++) {
        img.at<Vec3b>(i, k) = mc.red;
    }
    img.at<Vec3b>(i, j+1) = mc.black;
    img.at<Vec3b>(i, j + 11) = mc.black;
}
void draw_mario_3(Mat img, int i, int j) {
    for (int k = j + 2; k < j + 9; k++) {
        if (k < j + 5) {
            img.at<Vec3b>(i, k) = mc.brown;
        }
        else if(k == j+7){
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j + 1) = mc.black;
    img.at<Vec3b>(i, j + 9) = mc.black;
}
void draw_mario_4(Mat img, int i, int j) {
    for (int k = j + 1; k < j + 11; k++) {
        if (k == j || k == j+3) {
            img.at<Vec3b>(i, k) = mc.brown;
        }
        else if (k == j + 7) {
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j) = mc.black;
    img.at<Vec3b>(i, j + 11) = mc.black;
}
void draw_mario_5(Mat img, int i, int j) {
    for (int k = j + 1; k < j + 12; k++) {
        if (k == j || k == j + 3 || k == j+4) {
            img.at<Vec3b>(i, k) = mc.brown;
        }
        else if (k == j + 8) {
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j) = mc.black;
    img.at<Vec3b>(i, j + 12) = mc.black;
}
void draw_mario_6(Mat img, int i, int j) {
    for (int k = j + 2; k < j + 11; k++) {
        if (k == j) {
            img.at<Vec3b>(i, k) = mc.brown;
        }
        else if (k > j + 6) {
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j + 1) = mc.black;
    img.at<Vec3b>(i, j + 11) = mc.black;
}
void draw_mario_7(Mat img, int i, int j) {
    for (int k = j + 3; k < j + 9; k++) {
        img.at<Vec3b>(i, k) = mc.skin;
    }
    img.at<Vec3b>(i, j + 2) = mc.black;
    img.at<Vec3b>(i, j + 9) = mc.black;
}
void draw_mario_8(Mat img, int i, int j) {
    for (int k = j + 2; k < j + 10; k++) {
        if (k == j+5 || k==j+8) {
            img.at<Vec3b>(i, k) = mc.blue;
        }
        else {
            img.at<Vec3b>(i, k) = mc.red;
        }
    }
    img.at<Vec3b>(i, j + 1) = mc.black;
    img.at<Vec3b>(i, j + 10) = mc.black;
}
void draw_mario_9(Mat img, int i, int j) {
    for (int k = j + 1; k < j + 11; k++) {
        if (k == j + 5 || k == j + 8) {
            img.at<Vec3b>(i, k) = mc.blue;
        }
        else {
            img.at<Vec3b>(i, k) = mc.red;
        }
    }
    img.at<Vec3b>(i, j) = mc.black;
    img.at<Vec3b>(i, j + 11) = mc.black;
}
void draw_mario_10(Mat img, int i, int j) {
    for (int k = j; k < j + 12; k++) {
        if (k > j + 4 && k < j + 9) {
            img.at<Vec3b>(i, k) = mc.blue;
        }
        else {
            img.at<Vec3b>(i, k) = mc.red;
        }
    }
    img.at<Vec3b>(i, j - 1) = mc.black;
    img.at<Vec3b>(i, j + 12) = mc.black;
}
void draw_mario_11(Mat img, int i, int j) {
    for (int k = j; k < j + 12; k++) {
        if (k == j+2 || k == j + 9) {
            img.at<Vec3b>(i, k) = mc.red;
        }
        else if (k == j + 4 || k == j + 6 || k == j + 7 || k == j + 9) {
            img.at<Vec3b>(i, k) = mc.blue;
        }
        else if (k == j + 5 || k == j + 8) {
            img.at<Vec3b>(i, k) = mc.yellow;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j - 1) = mc.black;
    img.at<Vec3b>(i, j + 12) = mc.black;
}
void draw_mario_12(Mat img, int i, int j) {
    for (int k = j; k < j + 12; k++) {
        if (k > j + 2 && k < j + 9) {
            img.at<Vec3b>(i, k) = mc.blue;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j - 1) = mc.black;
    img.at<Vec3b>(i, j + 12) = mc.black;
}
void draw_mario_13(Mat img, int i, int j) {
    for (int k = j; k < j + 12; k++) {
        if (k > j + 1 && k < j + 10) {
            img.at<Vec3b>(i, k) = mc.blue;
        }
        else {
            img.at<Vec3b>(i, k) = mc.skin;
        }
    }
    img.at<Vec3b>(i, j - 1) = mc.black;
    img.at<Vec3b>(i, j + 12) = mc.black;
}
void draw_mario_14(Mat img, int i, int j) {
    for (int k = j + 2; k < j + 10; k++) {
        if (k == j+6 || k == j+7) {
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.blue;
        }
    }
    img.at<Vec3b>(i, j + 1) = mc.black;
    img.at<Vec3b>(i, j + 10) = mc.black;
}
void draw_mario_15(Mat img, int i, int j) {
    for (int k = j + 1; k < j + 11; k++) {
        if (k == j + 6 || k == j + 7) {
            //do nothing
        }else if (k == j + 5 || k == j + 8) {
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.brown;
        }
    }
    img.at<Vec3b>(i, j) = mc.black;
    img.at<Vec3b>(i, j + 11) = mc.black;
}
void draw_mario_16(Mat img, int i, int j) {
    for (int k = j; k < j + 12; k++) {
        if ( k == j + 6 || k == j + 7) {
            //do nothing
        }else if (k == j + 5 || k == j + 8) {
            img.at<Vec3b>(i, k) = mc.black;
        }
        else {
            img.at<Vec3b>(i, k) = mc.brown;
            img.at<Vec3b>(i+1, k) = mc.black;
        }
    }
    img.at<Vec3b>(i, j - 1) = mc.black;
    img.at<Vec3b>(i, j + 12) = mc.black;
}

//this function calls the mario drawing lines on a set of conditions
void marioify(Mat img) {
    //stretch could be implemented in future
    int stretch = 1;
    int rowmin = stretch * 17;
    int colmin = stretch * 13;
    for (int i = 0; i < img.rows - rowmin; i++) {
        for (int j = 0; j < img.cols - colmin; j++) {
            //if function choosing specified intervals
            if (i % 36 == 1 && j % 20 == 1) {
                draw_mario_1(img, i, j);
                draw_mario_2(img, i+1, j);
                draw_mario_3(img, i+2, j);
                draw_mario_4(img, i+3, j);
                draw_mario_5(img, i+4, j);
                draw_mario_6(img, i+5, j);
                draw_mario_7(img, i+6, j);
                draw_mario_8(img, i+7, j);
                draw_mario_9(img, i+8, j);
                draw_mario_10(img, i+9, j);
                draw_mario_11(img, i+10, j);
                draw_mario_12(img, i+11, j);
                draw_mario_13(img, i+12, j);
                draw_mario_14(img, i+13, j);
                draw_mario_15(img, i+14, j);
                draw_mario_16(img, i+15, j);
            }
        }
    }
}

//this function skews the image pink
void pinkify(Mat img) {
    MatIterator_<Vec3b> it, end;
    for (it = img.begin<Vec3b>(), end = img.end<Vec3b>(); it != end; ++it) {
        (*it)[0] = (*it)[0] + (.1 * (255 - (*it)[0]));
        (*it)[1] = (*it)[1] - (.3 * (*it)[1]);
        (*it)[2] = (*it)[2] + .6 * (255 - (*it)[2]);
    }
}

//using the circle function to draw a target on an image
void draw_target(Mat img) {
    Mat iclone = img.clone();
    Mat target(img.size(), img.type());
    Point center(img.cols / 2, img.rows / 2);
    vector<int> rads;
    rads.push_back(30);
    for (int i = 50; i < max(img.cols / 2, img.rows / 2) + 50;i+=50) {
        rads.push_back(i);
    }
   
    Vec3b color;
    for (int i = rads.size()-1; i > 0; i--) {
        color = i % 2 == 0 ? mc.blue : mc.red;
        circle(target, center, rads[i], color, 50);
    }
    circle(target, center, rads[0], mc.red, -1);

    //blend target and original
    double alpha = .7;
    double beta = 1 - alpha;
    Mat combined;

    //addWeighted function to blend two images
    addWeighted(iclone, alpha, target, beta, 0.0, img);
}

//using canny to draw lines on an image
void draw_some_edges(Mat img) {
    Mat edge(img.size(), img.type());
    cvtColor(img, edge, COLOR_BGR2GRAY);
    //using canny with low thresholds
    Canny(edge, edge, 30, 30);
    cvtColor(edge, edge, COLOR_GRAY2BGR);
    //blend edges and original
    double alpha = .4;
    double beta = 1 - alpha;

    //addWeighted function to blend two images
    addWeighted(edge, alpha, img, beta, 0.0, img);
}

int main(int argc, char* argv[])
{
    //enabling command line arguments for file paths if desired
    for (int i = 1; i < argc; i++) {
        std::cout << argv[i] << std::endl;
    }
    if (argc >= 3) {
        input_file_path = argv[1];
        output_folder_path = argv[2];
        std::cout << "File paths set" << std::endl;
    }

/************************************************************************************************************
*                                               PART 1                                                      *
*************************************************************************************************************/

/******************************************* Part 1.1 ********************************************************/

    //loading image to program, including check to see if image is readable
    //if not readable, doing a rudimentary exit. 
    std::string input_img = input_file_path;
    bool readable = haveImageReader(input_img);
    if (!readable) {
        std::cout << "input image not readable" << std::endl;
        std::cout << "must have image file at Data/input/" << std::endl;
        std::cout << "otherwise, specify desired input and output file locations in the command line" << std::endl;
        std::cout << "press enter to end" << std::endl;
        getchar();
        std::cout << "close this window" << std::endl;
        return 1;
    }
    else {
        std::cout << "Image readable" << std::endl;
    }
    Mat selfie = imread(input_img);

    show_image(selfie, "Original Selfie", 45, 45);

 /******************************************* Part 1.2 ********************************************************/
    //(1) Convert image to grayscale
    Mat gray_selfie;
    cvtColor(selfie, gray_selfie, COLOR_BGR2GRAY);

    //show gray selfie
    show_image(gray_selfie, "Gray Selfie", selfie.cols + 45, 45);
    //save gray selfie 
    save_image(gray_selfie, "gray_selfie.jpg");

    //(2) Extracting blue, green, red (BGR) channels from the original
    //create three copies of the original image
    Mat blue_selfie = selfie.clone();
    Mat green_selfie = selfie.clone();
    Mat red_selfie = selfie.clone();

    //for each color, at each pixel, we set the undesired channels to 0. 
    //this seemed to produce less code than splitting and merging. 
    //using iterator method

    MatIterator_<Vec3b> it, end;
    for (it = blue_selfie.begin<Vec3b>(), end = blue_selfie.end<Vec3b>(); it != end; ++it) {
        (*it)[1] = 0;
        (*it)[2] = 0;
    }
    for (it = green_selfie.begin<Vec3b>(), end = green_selfie.end<Vec3b>(); it != end; ++it) {
        (*it)[0] = 0;
        (*it)[2] = 0;
    }
    for (it = red_selfie.begin<Vec3b>(), end = red_selfie.end<Vec3b>(); it != end; ++it) {
        (*it)[0] = 0;
        (*it)[1] = 0;
    }

    //showing and saving color images
    show_image(blue_selfie, "Blue Selfie", selfie.cols /2, selfie.rows/2);
    save_image(blue_selfie, "blue_selfie.jpg");

    show_image(green_selfie, "Green Selfie", selfie.cols * 1.5, selfie.rows / 2);
    save_image(green_selfie, "green_selfie.jpg");

    show_image(red_selfie, "Red Selfie", selfie.cols * 2.5, selfie.rows / 2);
    save_image(red_selfie, "red_selfie.jpg");

/******************************************* Part 1.3 ********************************************************/
   //creating new matrix of half the size and the same type as the original
    Mat half_selfie(selfie.size()/2, selfie.type());
    Mat quarter_selfie(half_selfie.size() / 2, half_selfie.type());

    //call halve function twice
    halve_color_image(selfie, half_selfie);
    halve_color_image(half_selfie, quarter_selfie);

    show_image(half_selfie, "Half Selfie", selfie.cols * 3.5, selfie.rows / 2);
    save_image(half_selfie, "half_selfie.jpg");

    show_image(quarter_selfie, "Quarter Selfie", selfie.cols * 3.5, selfie.rows / 2 + half_selfie.rows);
    save_image(quarter_selfie, "quarter_selfie.jpg");

/******************************************* Part 1.4 ********************************************************/
  //make space for new sizes
    Mat new_half_selfie(half_selfie);
    Mat new_selfie(selfie);

    double_color_image(quarter_selfie, new_half_selfie);
    double_color_image(new_half_selfie, new_selfie);

    show_image(new_half_selfie, "New Half Selfie", selfie.cols, selfie.rows / 4);
    save_image(new_half_selfie, "new_half_selfie.jpg");

    show_image(new_selfie, "New Selfie", 0, 0);
    save_image(new_selfie, "new_selfie.jpg");

/******************************************* Part 1.5 ********************************************************/

    Mat improved_new_selfie = new_selfie.clone();

    //our plan to improve the image is to reduce the pixelation
    //to do this, we average each pixel with its surrounding pixels
    //this should improve the transitions, making the image smoother
    //takes an input Mat object and distance of pixels "reached" in a square
    //This makes the image more blurred instead of pixelated, and stretches it slightly. 
    //can play around with the number of blending passes, as well as how many pixels are averaged. 
    //each round we are going once top left to bottom right, then from top right to bottom left.
    int numBlends = 2;
    int reach = 4;
    for (int i = 0; i < numBlends; i++) {
        blend_color_pixels(improved_new_selfie, reach);
        blend_color_pixels_back(improved_new_selfie, reach);
    }

    show_image(improved_new_selfie, "Improved New Selfie", selfie.cols);
    save_image(improved_new_selfie, "improved_new_selfie.jpg");

/******************************************* Part 1.6 ********************************************************/
   //Using library resize function with different interpolation methods
    //Nearest
    Mat nearest_half(half_selfie);
    Mat nearest_full(selfie);
    
    resize(quarter_selfie, nearest_half, nearest_half.size(), INTER_NEAREST);
    resize(nearest_half, nearest_full, nearest_full.size(), INTER_NEAREST);

    show_image(nearest_full, "Nearest New Selfie", selfie.cols*2);
    save_image(nearest_full, "nearest_new_selfie.jpg");

    //linear
    Mat linear_half(half_selfie);
    Mat linear_full(selfie);

    resize(quarter_selfie, linear_half, linear_half.size(), INTER_LINEAR);
    resize(linear_half, linear_full, linear_full.size(), INTER_LINEAR);

    show_image(linear_full, "Linear New Selfie", selfie.cols * 3);
    save_image(linear_full, "linear_new_selfie.jpg");

    //Cubic
    Mat cubic_half(half_selfie);
    Mat cubic_full(selfie);

    resize(quarter_selfie, cubic_half, cubic_half.size(), INTER_CUBIC);
    resize(cubic_half, cubic_full, cubic_full.size(), INTER_CUBIC);

    show_image(cubic_full, "Cubic New Selfie", 0, selfie.rows/2);
    save_image(cubic_full, "cubic_new_selfie.jpg");

    //Area
    Mat area_half(half_selfie);
    Mat area_full(selfie);

    resize(quarter_selfie, area_half, area_half.size(), INTER_AREA);
    resize(area_half, area_full, area_full.size(), INTER_AREA);

    show_image(area_full, "Area New Selfie", selfie.cols, selfie.rows/2);
    save_image(area_full, "area_new_selfie.jpg");

    //Lanczos4
    Mat lan_half(half_selfie);
    Mat lan_full(selfie);

    resize(quarter_selfie, lan_half, lan_half.size(), INTER_LANCZOS4);
    resize(lan_half, lan_full, lan_full.size(), INTER_LANCZOS4);

    show_image(lan_full, "LANCZOS4 New Selfie", selfie.cols * 2, selfie.rows/2);
    save_image(lan_full, "lan_new_selfie.jpg");


/************************************************************************************************************
*                                               PART 2                                                      *
*************************************************************************************************************/

/******************************************* Part 2.1 ********************************************************/
    //this next part accesses the laptop's webcam and contiuously shows the image
    //the parts to do that was demonstrated by Ginni from tutorialspoint linked in the question. 
    Mat myImage;//Declaring a matrix to load the frames//
    Mat savePic; //matrix to save picture
    namedWindow("Video Player");//Declaring the video to show the video//
    VideoCapture cap(0);//Declaring an object to capture stream of frames from default camera//
    if (!cap.isOpened()) { //This section prompt an error message if no video stream is found//
        std::cout << "No video stream detected" << std::endl;
        system("pause");
        return-1;
    }
    else {
        std::cout << "Press Enter to take picture" << std::endl;
        std::cout << "Press ESC to stop video" << std::endl;
    }
    while (true) { //Taking an everlasting loop to show the video//
        cap >> myImage;
        if (myImage.empty()) { //Breaking the loop if no video frame is detected//
            break;
        }
        imshow("Video Player", myImage);//Showing the video//
        char c = (char)waitKey(25);//Allowing 25 milliseconds frame processing time and initiating break condition//
        if (c == 27) { //If 'Esc' is entered break the loop//
            break;
        }
        if (c == 13) { //when pressing enter, picture is captured using read
            std::cout << "Taking picture" << std::endl;
            bool pic = cap.read(savePic);
            if (pic) {
                std::cout << "Picture captured successfully" << std::endl;
                //saving and showing picture
                show_image(savePic, "Video Captured Image");
                save_image(savePic, "computer_camera_selfie.jpg");
            }
            else {
                std::cout << "Picture not captured" << std::endl;
            }
        }

    }
    cap.release();//Releasing the buffer memory//

/******************************************* Part 2.2 ********************************************************/
    Mat com_selfie = imread(output_folder_path + "computer_camera_selfie.jpg");
    //need to resize one of the images so they're the same size
    //creating matrix of correct size with choice of background color
    auto back_color = Scalar(0,0,0);
    Mat com_selfie_resized(selfie.size(), selfie.type(), back_color);
    //shaping to be the same as the selfie
    //A little crude, but does not stretch image
    crude_fit(com_selfie, com_selfie_resized);

    show_image(com_selfie, "Com Selfie", selfie.cols, 45);
    show_image(com_selfie_resized, "Com Selfie Resized", selfie.cols * 2, 45);

    //from the docs we are modeling the function g(x) = (1-a)f0(x) + af1(x)
    double alpha = .4;
    double beta = 1 - alpha;
    Mat combined;

    //addWeighted function to blend two images
    addWeighted(selfie, alpha, com_selfie_resized, beta, 0.0, combined);

    show_image(combined, "Combined Selfies", selfie.cols * 3, 45);
    save_image(combined, "Combined.jpg");

/******************************************* Part 2.3 ********************************************************/
    //my own effects
    Mat my_effects = com_selfie.clone();

    //the first function makes the image more pink
    //the second one draws a bunch of little marios
    pinkify(my_effects);
    marioify(my_effects);
    show_image(my_effects, "Effected Selfies");
    save_image(my_effects, "capturedImgwEffects.jpg");

/******************************************* Part 2.4 ********************************************************/
    //opencv effects
    Mat cv_effects = com_selfie.clone();

    //this uses the canny function to get some edges, then draws them back on the original
    draw_some_edges(cv_effects);
    //this uses the circle function to draw a target over the image
    draw_target(cv_effects);
    show_image(cv_effects, "CV Effects");
    save_image(cv_effects, "capturedImgwCVEffects.jpg");
/********************************** clean up and shut down program on key press ***********************************/
    waitKey(0);
    destroyAllWindows();
    return 0;
}
