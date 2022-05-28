#define DLLEXPORT extern "C"
#include <iostream>  
#include <Eigen/Eigen>  
#include <stdlib.h>  
#include <Eigen/Geometry>  
#include <Eigen/Core>  
#include <vector>  
#include <math.h>  
#include <vector>
  
using namespace std;  
using namespace Eigen;  
  
DLLEXPORT double* rotationMatrix2Quaterniond(double r)
{   
    r = r / 180 * M_PI;
    Eigen::Vector3d x_axiz,y_axiz,z_axiz;  

    Eigen::Matrix3d M;

    x_axiz << cos(r), sin(r), 0;  
    y_axiz << sin(r), -cos(r), 0;  
    z_axiz << 0,0,-1;

    M << x_axiz,y_axiz,z_axiz;
    
    Eigen::Quaterniond q = Eigen::Quaterniond(M);
    q.normalize();

    static double res[4];
    res[0] = q.x();
    res[1] = q.y();
    res[2] = q.z();
    res[3] = q.w();
    // cout << "RotationMatrix2Quaterniond result is:" <<endl;  
    // cout << "x = " << q.x() <<endl;  
    // cout << "y = " << q.y() <<endl;  
    // cout << "z = " << q.z() <<endl;  
    // cout << "w = " << q.w() <<endl<<endl;  
    return res;
}