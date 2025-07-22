#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "GMRsaliency.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

    Mat sal,img;

    img = imread(argv[1]);

    GMRsaliency GMRsal;
	sal = GMRsal.GetSal(img);

    imshow("img", img);
    imshow("Sal", sal);

    waitKey();

    
   return 0;
}





