#include "magnetization.hpp"
#include "math.h"
#include "mex.h"

CUDA_CALLABLE_MEMBER double myatomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__host__ CUDA_CALLABLE_MEMBER magnetization::magnetization(double mx0, double my0, double mz0, double x,
            double y, double z, int index, double offres, int volume, int avg) 
            : mx(mx0),my(my0),mz(mz0),xpos(x),ypos(y),zpos(z),bin(index),
                    offres(offres), volume(volume), avg(avg) {}

CUDA_CALLABLE_MEMBER void magnetization::rotate(double bx,double by, double bz, double tstep) {
    double xprod[3], tempm[3];
    double dot, phi, weff;
    
    weff = sqrt(bx*bx + by*by + bz*bz);
    phi = -2*M_PI*weff * tstep;
    
    if (weff != 0.0) {
        
        xprod[0] = (by*mz - bz*my)/weff;
        xprod[1] = (bz*mx - bx*mz)/weff;
        xprod[2] = (bx*my - by*mx)/weff;
        
        dot = (bx*mx + by*my + bz*mz)/weff;
        
        tempm[0] = cos(phi)*mx + sin(phi)*xprod[0] + (1-cos(phi))*dot*bx/weff;
        tempm[1] = cos(phi)*my + sin(phi)*xprod[1] + (1-cos(phi))*dot*by/weff;
        tempm[2] = cos(phi)*mz + sin(phi)*xprod[2] + (1-cos(phi))*dot*bz/weff;
        
        mx = tempm[0];
        my = tempm[1];
        mz = tempm[2];
    }
}

CUDA_CALLABLE_MEMBER void magnetization::display() {
    
}

CUDA_CALLABLE_MEMBER int magnetization::getBin() {
    return bin;
}

__host__ CUDA_CALLABLE_MEMBER void magnetization::setBin(int index,int numCols, int numRows,
        int numPages) {
   
    int z = (index + 1) % (avg * avg * numRows * numCols);
    int jj, ii, kk;
    
    if (z != 0) {
        jj = z % (avg * numCols);
        if (jj == 0) {
            jj = avg * numCols;
        }
        ii = ((z - jj)/(avg * numCols)) + 1;
    }
    else{
        jj = avg * numCols;
        ii = avg * numRows;
    }
    ii = ii - 1;
    jj = jj - 1;
    kk = (index - avg*numCols*(ii)-jj)/(avg * avg * numRows * numCols);

    int flag_jj = 1;
    int count_jj = 0;
    int avg_from_jj = 0;
    
    int flag_ii = 1;
    int count_ii = 0;
    int avg_from_ii = 0;
    
    int flag_kk = 1;
    int count_kk = 0;
    int avg_from_kk = 0;
    
    while (flag_jj == 1) {
        if (jj < avg) {
            flag_jj = 0;
        }
        else{
            count_jj = count_jj + 1;
            if (count_jj >= 1){
                avg_from_jj = avg_from_jj + 1;
                count_jj = 0;
            }
            jj = jj - avg;
        }
    }
    
   
    if (numRows > 1){
        while (flag_ii == 1) {
            if (ii < avg) {
                flag_ii = 0;
            }
            else{
                count_ii = count_ii + 1;
                if (count_ii >= 1){
                    avg_from_ii = avg_from_ii + 1;
                    count_ii = 0;
                }
                ii = ii - avg;
            }
        }
    }
    
    if (numPages > 1) {
        while (flag_kk == 1) {
            if (kk < avg) {
                flag_kk = 0;
            }
            else{
                count_kk = count_kk + 1;
                if (count_kk >= 1){
                    avg_from_kk = avg_from_kk + 1;
                    count_kk = 0;
                }
                kk = kk - avg;
            }
        }
    }
    
    bin = avg_from_jj + (numCols)*avg_from_ii + ((numRows)*(numCols))*avg_from_kk;
};

CUDA_CALLABLE_MEMBER void magnetization::acquire(double* mxout, double* myout, double* mzout, 
        int ndims, int time) {
    int outputBin = bin + time*volume;
    myatomicAdd(&mxout[outputBin],mx/pow(avg,ndims));
    myatomicAdd(&myout[outputBin],my/pow(avg,ndims));
    myatomicAdd(&mzout[outputBin],mz/pow(avg,ndims));
};

__host__ CUDA_CALLABLE_MEMBER void magnetization::setVolume(int numRows, int numCols, int numPages){
    volume = numRows * numCols * numPages;
};

CUDA_CALLABLE_MEMBER double magnetization::getX() {
    return xpos;
};

CUDA_CALLABLE_MEMBER double magnetization::getY() {
    return ypos;
};

CUDA_CALLABLE_MEMBER double magnetization::getZ() {
    return zpos;
};

__host__ CUDA_CALLABLE_MEMBER void magnetization::setpos(int index,double* xgrid, double* ygrid, double* zgrid) {
    xpos = xgrid[index];
    ypos = ygrid[index];
    zpos = zgrid[index];
};

CUDA_CALLABLE_MEMBER void magnetization::set2eq(){
    mx = 0;
    my = 0;
    mz = 1;
};

CUDA_CALLABLE_MEMBER void magnetization::refocusM(){
    mx = -mx;
    mz = -mz;
};

__host__ void magnetization::setobj(double objmz){
    mz = objmz;
};

__host__ void magnetization::setOffset(double offset){
    offres = offset;
};  

CUDA_CALLABLE_MEMBER double magnetization::getOffset(){
    return offres;
};