###Initial setup

+ Compile the `mergevec.cpp` file in the `./src` directory:

        cp src/mergevec.cpp ~/opencv-*/apps/haartraining
        cd ~/opencv-*/apps/haartraining
        g++ `pkg-config --libs --cflags opencv` -I. -o /src/haar/mergevec mergevec.cpp\
          cvboost.cpp cvcommon.cpp cvsamples.cpp cvhaarclassifier.cpp\
          cvhaartraining.cpp\
          -lopencv_core -lopencv_calib3d -lopencv_imgproc -lopencv_highgui -lopencv_objdetect
          
+ Test running if getting an error running `mergevec` then execute         
        sudo ldconfig


###Steps

+
  run the thing to extra positive images etc

  3. Put your positive images in the `./positive_images` folder and create a list
  of them:

          find ./positive_images -iname "*.jpg" > positives.txt

  4. Put the negative images in the `./negative_images` folder and create a list of them:

          find ./negative_images -iname "*.jpg" > negatives.txt

  + Create positive samples with the `createsamples.pl` script and save them
  to the `./samples` folder:
  
          perl createsamples.pl positives.txt data/bg.txt samples 1500\
            "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
            -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 80 -h 40"

  7. Use the compiled executable `mergevec` to merge the samples in `./samples`
  into one file:

          find ./samples -name '*.vec' > samples.txt
          ./mergevec samples.txt samples.vec

  8. Start training the classifier with `opencv_traincascade`, which comes with
  OpenCV, and save the results to `./classifier`:

          opencv_traincascade -data out/classifier -vec samples.vec -bg data/bg.txt\
            -numStages 20 -minHitRate 0.95 -maxFalseAlarmRate 0.4 -numPos 1000\
            -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 420\
            -precalcIdxBufSize 420 -featureType LBP


###File format
info_x.dat
[filename] [# of objects] [[x y width height] [... 2nd object] ...]
[filename] [# of objects] [[x y width height] [... 2nd object] ...]
[filename] [# of objects] [[x y width height] [... 2nd object] ...]
...
where (x,y) is the left-upper corner of the object where the origin (0,0) is the left-upper corner of the image such as

e.g. 
/getter_data/images/5317e7c8d7f3300000000005.jpg 1 1596 814 179 368

###Query to create negative images



###Query to create positive images