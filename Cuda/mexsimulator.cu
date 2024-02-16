#include "magnetization.hpp"
#include "math.h"
#include "mexsimulator.hpp"
#include "event_manager.hpp"
#include "mex.h"


__global__ void mexsimulator(magnetization* magn, double* mxout, double* myout,
        double* mzout, int* ptr_nelements,  int* ptr_numCols, int* ptr_numRows,
        int* ptr_numPages,int* ptr_ndims, double* gradx, double* grady, double* gradz,
        double* rfpulse, double* rfphase, double* events, int* ptr_nEvents) {

    /* Index for voxel loop*/
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nelements = *ptr_nelements;

    if (index < nelements){ //make sure index is within bounds of magnetization array
    int xstart = 0, ystart = 0, zstart = 0, rfstart = 0;
    int ndims = *ptr_ndims;
    int nEvents = *ptr_nEvents;

    /* Index for event loop */
    int eventIndex;
    
    /* index to access values in *events */
    int eventdurationIndex;
    int eventstepsIndex;
    
    int acqstart = 0;
    
    for (eventIndex = 0; eventIndex < nEvents; eventIndex++){
        eventdurationIndex = eventIndex + nEvents;
        eventstepsIndex = eventIndex + 2*nEvents;
        
        mrEvent myEvent((typeEvent)events[eventIndex],events[eventdurationIndex],
                (int)events[eventstepsIndex]);
        
            
            switch( myEvent.getEvent() ) {
                
                case pulse:
                    myEvent.pulseEvent(&magn[index], rfstart, rfpulse,rfphase);
                    break;
                    
                case gradient:
                    myEvent.gradEvent(&magn[index], xstart, ystart, zstart, gradx,
                            grady, gradz);
                    break;
                case pulseAndgrad:
                    myEvent.pulsegradEvent(&magn[index], xstart, ystart, zstart, gradx,
                            grady, gradz, rfpulse, rfphase, rfstart);
                    break;
                    
                case delay:
                    myEvent.delayEvent(&magn[index]);
                    
                    break;
                    
                case acquire:
                    
                    myEvent.acquireEvent(&magn[index], xstart, ystart, zstart,
                            gradx, grady, gradz, mxout, myout, mzout,
                            ndims, acqstart);
                    
                    break;
                case pulseGradAcq:
                    
                    myEvent.pulseGradAcqEvent(&magn[index], xstart, ystart, zstart,
                            gradx, grady, gradz, mxout, myout, mzout,
                            ndims, acqstart, rfpulse, rfphase, rfstart);
                    break;
                case thermaleq:
                    myEvent.thermaleqEvent(&magn[index]);
                    break;
                 
                case refocus:
                    myEvent.refocusEvent(&magn[index]);
                    break;

                }
        /*Update starting point in the various waveform arrays */
    
        if (rfpulse[rfstart] == rfNull){
            myEvent.indexUpdate(&rfstart,1);
        }
        else{
            myEvent.indexUpdate(&rfstart,myEvent.getnSteps());
        };
        if (gradx[xstart] == gradNull){
            myEvent.indexUpdate(&xstart,1);
        }
        else{
            myEvent.indexUpdate(&xstart,myEvent.getnSteps());
        };
        if (grady[ystart] == gradNull){
            myEvent.indexUpdate(&ystart,1);
        }
        else{
            myEvent.indexUpdate(&ystart,myEvent.getnSteps());
        };
        if (gradz[zstart] == gradNull){
            myEvent.indexUpdate(&zstart,1);
        }
        else{
            myEvent.indexUpdate(&zstart,myEvent.getnSteps());
        };
        
        if (myEvent.getEvent() == acquire){
            myEvent.indexUpdate(&acqstart,myEvent.getnSteps());
        };
    };
    };
    //__syncthreads();
};