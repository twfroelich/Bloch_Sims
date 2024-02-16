/*==========================================================
 * mex_blochsim.cpp - C++ Bloch Simulator
 *========================================================*/

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <iostream>
#include "mex.h"
#include "math.h"
#include "gpu/mxGPUArray.h"
#include "magnetization.hpp"
#include "mexsimulator.hpp"
#include "event_manager.hpp"
#include <string>
#include <float.h>

#define mymax(a,b) a > b ? a : b

/* The gateway function. */ 
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {

    int usrDefNRHS = 2;
    int usrDefNLHS = 3;
    
    std::string RHSerrormsg; //strings to store error messages
    std::string LHSerrormsg;
    std::string sNRHS = std::to_string(usrDefNRHS); //add number of arguments to string
    std::string sNLHS = std::to_string(usrDefNLHS);
    RHSerrormsg = "MEX_BLOCHSIM requires " + sNRHS + " input arguments."; //concat. strings
    LHSerrormsg = "MEX_BLOCHSIM requires " + sNLHS + " output arguments.";
    if(nrhs != usrDefNRHS) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                          RHSerrormsg.c_str()); //.c_str returns pointer to first element.
    }
    if(nlhs != usrDefNLHS) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
                          LHSerrormsg.c_str());
    }

    int cudaDevNum; //user specified device number
    cudaDevNum = (int)mxGetScalar(prhs[1]); //get that number from matlab
    int maxDevNum; // number of available devices. Must have cudaDevNum < maxDevNum
    
    cudaGetDeviceCount( &maxDevNum ); //returns number of devices.

    if(cudaDevNum > maxDevNum || cudaDevNum < 0) {

        std::string cudaDevErrMsg; //string to store error messages
        std::string snumdev = std::to_string(maxDevNum); //add number of devices to string
        std::string snumdevmaxval = std::to_string(maxDevNum-1); //add number of devices to string

        cudaDevErrMsg = "Specified Device Number exceeds number of GPU devices: "
                             + snumdev + ". Enter number 0 - " + snumdevmaxval + "\n"; //concat. strings
   
        mexPrintf(cudaDevErrMsg.c_str());
        mexPrintf("Defaulting to device 0 \n");
        cudaSetDevice(0);
    }
    else{
        cudaSetDevice(cudaDevNum);
    }

    int nfields = mxGetNumberOfFields(prhs[0]);
    int ifield;
    char* fname = NULL;
    fname = new char[256]; //don't have fieldnames longer than 256 characters

    //Spatial grids
    double* xgrid;
    double* ygrid;
    double* zgrid;


    int FLAGoffres; //for determinig if off resonance map was input
    /* Acquire pointers to the input data */
    int index,avg,numRows,numCols,numPages,nelements;
    //set some default values
    numRows = 1;
    numPages = 1;
    numCols = 1;
    avg = 1;

    const size_t* dims;
    int ndims;

    double* Gx;
    double* Gy;
    double* Gz;
    size_t nGx;
    size_t nGy;
    size_t nGz;

    double* rfamp;
    size_t nrfamp;
    double* rfphase;
    size_t nrfphase;

    //Offresonance
    double* offres;
    
    double* events;
    size_t numelemEvents;
    int nEvents;

    //Get user-defined object. Need an error check on the size of it still 
    double *usrObj;

    for (ifield=0; ifield< nfields; ifield++){
          fname = (char*)mxGetFieldNameByNumber(prhs[0],ifield);

          if (strcmp(fname,"avg") == 0){ //strcmp() returns 0 if they match
	      // second argument is 0 since prhs[0] points to a 1x1 struct
              avg = (int)(*mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield))); 
          }
	  else if (strcmp(fname,"xgrid") == 0){
		xgrid = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		dims = mxGetDimensions(mxGetFieldByNumber(prhs[0],0,ifield));
		ndims = mxGetNumberOfDimensions(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"ygrid") == 0){
		ygrid = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
          else if (strcmp(fname,"zgrid") == 0){
		zgrid = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"Gx") == 0){
		Gx = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		nGx = mxGetNumberOfElements(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"Gy") == 0){
		Gy = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		nGy = mxGetNumberOfElements(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"Gz") == 0){
		Gz = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		nGz = mxGetNumberOfElements(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"rfamp") == 0){
		rfamp = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		nrfamp = mxGetNumberOfElements(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"rfphase") == 0){
		rfphase = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		nrfphase = mxGetNumberOfElements(mxGetFieldByNumber(prhs[0],0,ifield));
	  }	
	  else if (strcmp(fname,"offres") == 0){
		offres = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
        FLAGoffres = 1;
	  }	
	  else if (strcmp(fname,"events") == 0){
		events = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
		numelemEvents = mxGetNumberOfElements(mxGetFieldByNumber(prhs[0],0,ifield));
		nEvents = mxGetM(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
	  else if (strcmp(fname,"usrObj") == 0){
		usrObj = mxGetPr(mxGetFieldByNumber(prhs[0],0,ifield));
	  }
    }

    /* This is the number of magnetization vectors being simulated */
    nelements = 1;
    for (index = 0; index < ndims; index++){
        nelements = dims[index] * nelements;
        mexPrintf("index: %d, dims[index]: %d, ndims: %d, nelements: %d \n",
                    index, dims[index], ndims,nelements);
    }
    

    if (nrfamp - nrfphase != 0){
        mexErrMsgIdAndTxt("MATLAB:mex_blochsim:typeargin",
                          "RF amp and phase must be same size.");
    }
    
    size_t nTime = 0;
    
    int dummy,dummyindex;
    for (dummy = 0; dummy < nEvents; dummy++){
        if (events[dummy] == 4 || events[dummy] == 5){ //these are acquisition events
            dummyindex = dummy + 2*nEvents; //accesses number of points in event
            nTime += (size_t)events[dummyindex];
        }
    }

    size_t* dimsOut = new size_t[ndims+1];
    switch (ndims) {
        case 1:
            numCols = mymax(dims[0]/avg,1);
            dimsOut[0] = numCols;
            dimsOut[1] = nTime;
            break;
        case 2:
            numCols = mymax(dims[0]/avg,1);
            numRows = mymax(dims[1]/avg,1);
            dimsOut[0] = numCols;
            dimsOut[1] = numRows;
            dimsOut[2] = nTime;
            
            break;
        case 3:
            numCols = mymax(dims[0]/avg,1);
            numRows = mymax(dims[1]/avg,1);
            numPages = mymax(dims[2]/avg,1);
            dimsOut[0] = numCols;
            dimsOut[1] = numRows;
            dimsOut[2] = numPages;
            dimsOut[3] = nTime;
            
            break;
        default:
            mexErrMsgIdAndTxt("MATLAB:mex_blochsim:ndims",
                          "Number of dimensions of xgrid must be 1, 2, or 3.");
    };
   
    /*Initialize magnetization array*/

    magnetization* magn = NULL;
    magn = new magnetization[nelements];
    
    for (index = 0; index < nelements; index++){
        magn[index].avg = avg;
        magn[index].setBin(index,numCols,numRows,numPages);
        magn[index].setVolume(numCols,numRows,numPages);
        magn[index].setpos(index,xgrid,ygrid,zgrid);
        if (FLAGoffres == 1){
            magn[index].setOffset(offres[index]);
        }
        else{
            magn[index].setOffset(0.0);
        }
        magn[index].setobj(usrObj[index]);
    }
    
    plhs[0] = mxCreateNumericArray(ndims+1,dimsOut,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericArray(ndims+1,dimsOut,mxDOUBLE_CLASS,mxREAL);
    plhs[2] = mxCreateNumericArray(ndims+1,dimsOut,mxDOUBLE_CLASS,mxREAL);
    
    double* mxout = mxGetPr(plhs[0]);
    double* myout = mxGetPr(plhs[1]);
    double* mzout = mxGetPr(plhs[2]);

    //Allocate memory for all the inputs on the GPU device
    magnetization* d_magn; //device copy of magnetization;
    double* d_mxout; //device copies of output magnetization
    double* d_myout;
    double* d_mzout; 

    cudaMalloc((void **)&d_magn,nelements*sizeof(magnetization));
    cudaMemcpy(d_magn, magn, nelements*sizeof(magnetization),cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&d_mxout,numRows*numCols*numPages*nTime*sizeof(double));
    cudaMalloc((void **)&d_myout,numRows*numCols*numPages*nTime*sizeof(double));
    cudaMalloc((void **)&d_mzout,numRows*numCols*numPages*nTime*sizeof(double));
    cudaMemcpy(d_mxout, mxout, numRows*numCols*numPages*nTime*sizeof(double),
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_myout, myout, numRows*numCols*numPages*nTime*sizeof(double),
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_mzout, mzout, numRows*numCols*numPages*nTime*sizeof(double),
                cudaMemcpyHostToDevice);
    //Do the gradients, rf, and event list now
    double* d_Gx;
    double* d_Gy;
    double* d_Gz;
    double* d_rfamp;
    double* d_rfphase;
    double* d_events;

    cudaMalloc((void **)&d_Gx,nGx*sizeof(double));
    cudaMemcpy(d_Gx,Gx,nGx*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_Gy,nGy*sizeof(double));
    cudaMemcpy(d_Gy,Gy,nGy*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_Gz,nGz*sizeof(double));
    cudaMemcpy(d_Gz,Gz,nGz*sizeof(double),cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_rfamp,nrfamp*sizeof(double));
    cudaMemcpy(d_rfamp,rfamp,nrfamp*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_rfphase,nrfamp*sizeof(double));
    cudaMemcpy(d_rfphase,rfphase,nrfamp*sizeof(double),cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&d_events,numelemEvents*sizeof(double));
    cudaMemcpy(d_events,events,numelemEvents*sizeof(double),cudaMemcpyHostToDevice);
    
    int* d_nelements;
    int* d_numCols;
    int* d_numRows;
    int* d_numPages;
    int* d_ndims;
    int* d_nEvents;

    cudaMalloc((void **)&d_nelements,sizeof(int));
    cudaMalloc((void **)&d_numCols,sizeof(int));
    cudaMalloc((void **)&d_numRows,sizeof(int));
    cudaMalloc((void **)&d_numPages,sizeof(int));
    cudaMalloc((void **)&d_ndims,sizeof(int));
    cudaMalloc((void **)&d_nEvents,sizeof(int));

    cudaMemcpy(d_nelements,&nelements,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_numCols,&numCols,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_numRows,&numRows,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_numPages,&numPages,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_ndims,&ndims,sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_nEvents,&nEvents,sizeof(int),cudaMemcpyHostToDevice);
    int threadPerBlock = 512;

    mexsimulator<<<((nelements + threadPerBlock-1)/threadPerBlock),threadPerBlock>>>(d_magn,d_mxout,d_myout,d_mzout,d_nelements,d_numCols,
d_numRows,d_numPages,d_ndims,d_Gx,d_Gy,d_Gz,d_rfamp,d_rfphase,d_events,d_nEvents);


//Transfer output arrays from GPU back to CPU
    cudaMemcpy(mxout,d_mxout,numRows*numCols*numPages*nTime*sizeof(double),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(myout,d_myout,numRows*numCols*numPages*nTime*sizeof(double),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(mzout,d_mzout,numRows*numCols*numPages*nTime*sizeof(double),
                cudaMemcpyDeviceToHost);

//Free all the allocated memory
    cudaFree(d_nEvents);
    cudaFree(d_ndims);
    cudaFree(d_numPages);
    cudaFree(d_numRows);
    cudaFree(d_numCols);
    cudaFree(d_nelements);

    cudaFree(d_events);
    cudaFree(d_rfphase);
    cudaFree(d_rfamp);
    cudaFree(d_Gz);
    cudaFree(d_Gy);
    cudaFree(d_Gx);
    cudaFree(d_mzout);
    cudaFree(d_myout);
    cudaFree(d_mxout);
    cudaFree(d_magn);
    
    
    delete [] magn;
    delete [] dimsOut;
    delete [] fname;
    dimsOut = NULL;
    magn = NULL;
    fname = NULL;

    mexPrintf("Simulation Complete! \n");
}

