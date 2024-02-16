/**************************************************************************
 *
 * BLOCH SIMULATOR MEX FILE
 * Stores the spins for all time not just at the end of the pulse
 * Has the option for an offset along Bz
 *if( tt >= acq_start)
 * {
 * Very limited built in consistency checks.
 * Ensure all inputs (gradients, RF, phase, etc... are the same length.
 * To parallelize, uncomment the openMP lines.
 *                          Author:         Taylor Froelich
 *                          CREATED:        Feb. 10, 2018
 *
 *
 *Arrays are 0 - indexed in C
 *mex bloch_2d_AT.c CFLAGS="$CFLAGS -std=c99"
 *
 *
 *mex bloch_2d_AT.c CFLAGS="$CFLAGS -fopenmp -std=c99" LDFLAGS="$LDFLAGS -fopenmp -std=c99"
 *
 *************************************************************************/

#include "mex.h"
#include "math.h"
#include "omp.h"

void blochsim(double *resPosX,double *resPosY,long int num_pts_posX,long int num_pts_posY,
        long int num_pts,long int acq_start,long int acq_pts, double tstep,
        double *b1x,double *b1y,double *gradx,double *grady, double *offset,
        double *Mx,double *My,double *Mz,double *Mxout_2d,double *Myout_2d,double *Mzout_2d)
{
    long int h,m;
    double const mex_pi = 4*atan(1.0);
    double const gamma = 2*mex_pi*4258;
    
    for (m = 0; m <= num_pts_posY-1; ++m)
    {
#pragma omp parallel for
        for (h = 0; h <= num_pts_posX-1; ++h)
        {
            long int tt;
            double phi;
            double crs_prd[3];
            double dot;
            double Weff[3];
            double abs_weff;
            
            double b1x_tt,b1y_tt;
            
            double Mxyz[6];
            
            Mxyz[0] = Mx[m*num_pts_posX + h];
            Mxyz[2] = My[m*num_pts_posX + h];
            Mxyz[4] = Mz[m*num_pts_posX + h];
           
            for(tt = 0; tt <= num_pts-1; tt++)
            {
                
                b1x_tt = b1x[tt];
                b1y_tt = b1y[tt];
                /*
                if (select >= 1)
                {
                    Weff[0] = b1x_tt;
                    Weff[1] = b1y_tt;
                    Weff[2] = gamma*resPosX[h]*gradx[tt] + gamma*resPosY[m]*grady[tt] + offset[m*num_pts_posX*num_pts + h*num_pts + tt];
                }
                else
                {*/
                    
                    Weff[0] = b1x_tt;
                    Weff[1] = b1y_tt;
                    Weff[2] = gamma*resPosX[h]*gradx[tt] + gamma*resPosY[m]*grady[tt] + offset[m*num_pts_posX + h];
                
                
                
                abs_weff = sqrt(pow(Weff[0],2)+pow(Weff[1],2)+pow(Weff[2],2));
                phi = -abs_weff*tstep;
                
                if(abs_weff > 0.0000001)
                {
                    Weff[0] = Weff[0]/abs_weff;
                    Weff[1] = Weff[1]/abs_weff;
                    Weff[2] = Weff[2]/abs_weff;
                    
                    crs_prd[0] = Weff[1]*Mxyz[4]-Weff[2]*Mxyz[2];
                    crs_prd[1] = Weff[2]*Mxyz[0]-Weff[0]*Mxyz[4];
                    crs_prd[2] = Weff[0]*Mxyz[2]-Weff[1]*Mxyz[0];
                    
                    dot = Weff[0]*Mxyz[0]+Weff[1]*Mxyz[2]+Weff[2]*Mxyz[4];
                    
                    Mxyz[1] = (cos(phi)*Mxyz[0]+sin(phi)*crs_prd[0]+(1-cos(phi))*dot*Weff[0]);
                    Mxyz[3] = (cos(phi)*Mxyz[2]+sin(phi)*crs_prd[1]+(1-cos(phi))*dot*Weff[1]);
                    Mxyz[5] = (cos(phi)*Mxyz[4]+sin(phi)*crs_prd[2]+(1-cos(phi))*dot*Weff[2]);
                }
                else
                {
                    Mxyz[1] = Mxyz[0];
                    Mxyz[3] = Mxyz[2];
                    Mxyz[5] = Mxyz[4];
                }
                
                if( tt >= acq_start)
                {
#pragma omp atomic
                    Mxout_2d[m*num_pts_posX*acq_pts + h*acq_pts + tt - acq_start] += Mxyz[1];
#pragma omp atomic
                    Myout_2d[m*num_pts_posX*acq_pts + h*acq_pts + tt - acq_start] += Mxyz[3];
#pragma omp atomic
                    Mzout_2d[m*num_pts_posX*acq_pts + h*acq_pts + tt - acq_start] += Mxyz[5];
                }
                
                Mxyz[0] = Mxyz[1];
                Mxyz[2] = Mxyz[3];
                Mxyz[4] = Mxyz[5];
                
            }
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nlhs != 3)
    {
        mexErrMsgTxt("ERROR:  Number of outputs does not equal 3. \n");
    }
    if (nrhs != 16)
    {
        mexErrMsgTxt("ERROR:  Number of RHS objects is incorrect. \n");
    }
    
    double *b1x;
    double *b1y;
    
    double *offset;
    
    double *gradx;
    double *grady;
    
    double *resPosX;
    double *resPosY;
    
    double tstep;
    int long num_pts;
    
    int long num_pts_posX;
    int long num_pts_posY;
    
    int long acq_start;
    int long acq_pts;
    
    double *Mx;
    double *My;
    double *Mz;
    
    double *Mxout_2d;
    double *Myout_2d;
    double *Mzout_2d;
    
    resPosX = mxGetPr(prhs[0]);
    resPosY = mxGetPr(prhs[1]);
    
    num_pts_posX = (long int)mxGetScalar(prhs[2]);
    num_pts_posY = (long int)mxGetScalar(prhs[3]);
    
    num_pts = (long int)mxGetScalar(prhs[4]);
    
    acq_start = (long int)mxGetScalar(prhs[5]);
    acq_pts = (long int)mxGetScalar(prhs[6]);
    
    tstep = mxGetScalar(prhs[7]);
    
    b1x = mxGetPr(prhs[8]);
    b1y = mxGetPr(prhs[9]);
    
    gradx = mxGetPr(prhs[10]);
    grady = mxGetPr(prhs[11]);
    
    offset = mxGetPr(prhs[12]);
    
    Mx = mxGetPr(prhs[13]);
    My = mxGetPr(prhs[14]);
    Mz = mxGetPr(prhs[15]);
    
    /*num_pts_posX*num_pts_posY*acq_pts*/
    
    plhs[0] = mxCreateDoubleMatrix(num_pts_posY*num_pts_posX*acq_pts,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(num_pts_posY*num_pts_posX*acq_pts,1,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(num_pts_posY*num_pts_posX*acq_pts,1,mxREAL);
    
    Mxout_2d = mxGetPr(plhs[0]);
    Myout_2d = mxGetPr(plhs[1]);
    Mzout_2d = mxGetPr(plhs[2]);
    
    blochsim(resPosX,resPosY,num_pts_posX,num_pts_posY,num_pts,acq_start,
            acq_pts,tstep,b1x,b1y,gradx,grady,offset,Mx,My,Mz,Mxout_2d,Myout_2d,Mzout_2d);
    
}