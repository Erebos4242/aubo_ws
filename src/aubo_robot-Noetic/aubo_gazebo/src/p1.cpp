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

int main()
{
    double r = 45;
    r = r / 180;
    Eigen::Vector3d x_axiz,y_axiz,z_axiz;
    x_axiz << 1,0,0;
    y_axiz << 0,-1,0;
    z_axiz << 0,0,-1;

    Eigen::Matrix3d M1, M2;
    M1 << x_axiz,y_axiz,z_axiz;

    x_axiz << cos(r), sin(r), 0;
    y_axiz << sin(r), cos(r), 0;
    z_axiz << 0,0,1;

    M2 << x_axiz,y_axiz,z_axiz;

    Eigen::Quaterniond q = Eigen::Quaterniond(M1 * M2);
    q.normalize();

    static double res[4];
    res[0] = q.x();
    res[1] = q.y();
    res[2] = q.z();
    res[3] = q.w();
    cout << "x = " << q.x() <<endl;
    cout << "y = " << q.y() <<endl;
    cout << "z = " << q.z() <<endl;
    cout << "w = " << q.w() <<endl<<endl;
    return 0;
}
