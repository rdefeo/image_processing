// Chet Corcos & Garrett Menghini
// February 25, 2012
// OpenCV Data Collection -Positives
//////////////////////////////////////
// This script will aid in collecting data for boosted haar training in OpenCV. Note that a 
// different script is used for collecting the negatives. 

// included necessary libraries

#include <iostream>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>


int main (int argc, const char * argv[])
{    
    /////////////////////////////// User Defined Inputs //////////////////////////////////
    char *basePath = new char[100];
    sprintf(basePath,"/Users/chet/Desktop/opencv images/");
    
    char *imgName = new char[100];
    sprintf(imgName, "chet_");
    //////////////////////////////////////////////////////////////////////////////////////
    
    // create necessary images
    IplImage* videoImg; // original image
    IplImage* videoImg2; // mirrored image
    IplImage* videoImg3; // copied and saved image
    
    // create a variable to count the number of saved images
    int saved = 0;
    
    // create necessary window to display video
    cvNamedWindow("Video", CV_WINDOW_AUTOSIZE);
    
    // initialize the camera
    CvCapture* capture = cvCreateCameraCapture(-1);
    
    // initialize the string to write in the info.dat file
    std::ofstream myfile;  
    
    // create a new file in which to write the data for the positive training samples
    FILE * pFile;
    
    // the .idx file is saved in the same directory as the images
    char *filePath = new char[100];
    strcpy(filePath, basePath);
    strcat(filePath,  "positives.idx");
    pFile = fopen(filePath, "w");
    
    /* Record for the camera. Whenever the space button is held down, save the image within the box
     */
    
    while(1)
    {   
        // display the mirrored, brightness equalized, grey image
        videoImg = cvQueryFrame(capture);
        
        videoImg2 = cvCreateImage(cvGetSize(videoImg), IPL_DEPTH_8U, 3);
        cvFlip(videoImg, videoImg2, 1);
        
        videoImg3 = cvCreateImage(cvGetSize(videoImg2), IPL_DEPTH_8U, 3);
        cvCopy(videoImg2, videoImg3);
        
        int rectx = (videoImg2 -> width)/3;
        int recty = (videoImg2 -> height)/4;
        int rectwidth = (videoImg2 -> width)/3;
        int rectheight = (videoImg2 -> height)/2;
        
        // set up the rectangle
        CvRect rect = cvRect(rectx, recty, rectwidth, rectheight);
        cvRectangleR(videoImg2, rect, cvScalar(0xff,0x00,0x00));
        
        // display the number of saved image
        char* myString1 = new char[100];
        sprintf(myString1, "%d", saved);
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
        cvPutText(videoImg2, myString1 , cvPoint(50,50), &font, CV_RGB(250, 0, 0));
        cvShowImage("Video", videoImg2);
        
        
        // if "a" is pressed, the current frame from the video is captured, equalized, and saved
        char a = cvWaitKey(1);
        if (a == 97)
        {
            saved = saved + 1;
            
            // generate the file name
            char* filename = new char[100];
            strcpy(filename, basePath);
            
            char* name = new char[100];
            strcpy(name, imgName);
            
            char* name2 = new char[100];
            sprintf(name2, "pos_img%d.jpg", saved);
            
            strcat(name, name2);
            strcat(filename, name);
            
            cvSaveImage(filename, videoImg3);
            
            // update info.dat with filename of new image.
            pFile = fopen(filePath, "a");
            
            char* nextline = new char[100];
            strcpy(nextline, name);
            
            char* params = new char[100];
            sprintf(params, " 1 %d %d %d %d\n", rectx, recty, rectwidth, rectheight);
            
            strcat(nextline, params);
            fputs(nextline, pFile);
            fclose(pFile);
        }
        // if the space bar is pressed, release all the data and end the program
        else if (a == 32)
        {
            // release images
            // this gives me an error for some reason so I commented it out
            //            cvReleaseImage(&videoImg);
            //            cvReleaseImage(&videoImg2);
            //            cvReleaseImage(&videoImg3);
            
            // destroy windows
            cvDestroyWindow("Video");
            
            // release the camera
            cvReleaseCapture(&capture);
            return 0;
        }
    } 
    return 0;
}