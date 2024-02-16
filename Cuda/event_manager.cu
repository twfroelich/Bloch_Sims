#include "event_manager.hpp"
#include "math.h"

/*
 * Gradients, pulses, delays, acquisition are input as vectors of fixed 
 * length, which need not all be the same length. There is a separate event 
 * array, structured as (going across columns):
 *      event type, duration, number of steps 
 * The Bloch simulator steps through the events, and updates the 
 * magnetization vectors accordingly. The counters (ctr) below update the
 * correct indices of the input vectors for the current event.
 */
CUDA_CALLABLE_MEMBER mrEvent::mrEvent(typeEvent eventtype1, double duration1, int nSteps1) :
    eventtype(eventtype1), duration(duration1), nSteps(nSteps1) {};
    
CUDA_CALLABLE_MEMBER typeEvent mrEvent::getEvent() {
    return eventtype;
}; 

CUDA_CALLABLE_MEMBER double mrEvent::gettstep() { 
    return tstep;
}

CUDA_CALLABLE_MEMBER void mrEvent::indexUpdate(int* start, int nSteps) {
    *start = *start + nSteps;
}

/* Don't need to define this until off-resonance map is included */
CUDA_CALLABLE_MEMBER void mrEvent::delayEvent(magnetization* magn) {
    
};

CUDA_CALLABLE_MEMBER void mrEvent::pulseEvent(magnetization* magn, int rfstart, double* rfpulse, double* rfphase){
    double rfx, rfy;
    int index;
    for (index = rfstart; index < rfstart+nSteps; index++){
        if (rfpulse[rfstart] != rfNull){
            rfx = cos(rfphase[index+rfstart]) * rfpulse[index+rfstart];
            rfy = sin(rfphase[index+rfstart]) * rfpulse[index+rfstart];
        }
        else{
            rfx = 0;
            rfy = 0;
        }
        
        magn->rotate(rfx,rfy,magn->getOffset(),tstep);
    }

};

CUDA_CALLABLE_MEMBER void mrEvent::gradEvent(magnetization* magn, int xstart, int ystart,
        int zstart, double* gradx, double* grady,
        double* gradz) {
    int index;
    double bx, by, bz;
    for (index = 0; index < nSteps; index++){
        /* No pretty way to implement this. Could do a bunch of cases, but 
         * there would be 2^3 = 8 possibilities, since any number of them 
         * could be null. 
         */
        if (gradx[xstart] != gradNull){
            bx = gauss2Hz * gradx[index+xstart] * magn->getX();
        }
        else{
            bx = 0;
        }
        if (grady[ystart] != gradNull){
            by = gauss2Hz * grady[index+ystart] * magn->getY();
        }
        else{
            by = 0;
        }
        if (gradz[zstart] != gradNull){
            bz = gauss2Hz * gradz[index+zstart] * magn->getZ();
        }
        else{
            bz = 0;
        }

        magn->rotate(0,0,bx+by+bz+magn->getOffset(),tstep);
    }
    
};

CUDA_CALLABLE_MEMBER void mrEvent::pulsegradEvent(magnetization* magn, int xstart, int ystart,
        int zstart, double* gradx, double* grady,
        double* gradz, double* rfpulse, double* rfphase,int rfstart) {
    int index;
    double bx, by, bz, rfx, rfy;
    
    for (index = 0; index < nSteps; index++){
        /* No pretty way to implement this. Could do a bunch of cases, but 
         * there would be 2^3 = 8 possibilities, since any number of them 
         * could be null. 
         */
        if (rfpulse[rfstart] != rfNull){
            rfx = cos(rfphase[index+rfstart]) * rfpulse[index+rfstart];
            rfy = sin(rfphase[index+rfstart]) * rfpulse[index+rfstart];
        }
        else{
            rfx = 0;
            rfy = 0;
        }
        
        if (gradx[xstart] != gradNull){
            bx = gauss2Hz * gradx[index+xstart] * magn->getX();
        }
        else{
            bx = 0;
        }
        if (grady[ystart] != gradNull){
            by = gauss2Hz * grady[index+ystart] * magn->getY();
        }
        else{
            by = 0;
        }
        if (gradz[zstart] != gradNull){
            bz = gauss2Hz * gradz[index+zstart] * magn->getZ();
        }
        else{
            bz = 0;
        }

        magn->rotate(rfx,rfy,bx+by+bz+magn->getOffset(),tstep);
    };
    
};

CUDA_CALLABLE_MEMBER void mrEvent::pulseGradAcqEvent(magnetization* magn, int xstart, int ystart,
        int zstart, double* gradx, double* grady,
        double* gradz, double* mxout, double* myout, double* mzout,
        int ndims, int timestart,double* rfpulse, 
        double* rfphase,int rfstart) {
    
    int index;
    double bx, by, bz, rfx, rfy;
    for (index = 0; index < nSteps; index++){
        /* No pretty way to implement this. Could do a bunch of cases, but 
         * there would be 2^3 = 8 possibilities, since any number of them 
         * could be null. 
         */
        if (rfpulse[rfstart] != rfNull){
            rfx = cos(rfphase[index+rfstart]) * rfpulse[index+rfstart];
            rfy = sin(rfphase[index+rfstart]) * rfpulse[index+rfstart];
        }
        else{
            rfx = 0;
            rfy = 0;
        }
        if (gradx[xstart] != gradNull){
            bx = gauss2Hz * gradx[index+xstart] * magn->getX();
        }
        else{
            bx = 0;
        }
        if (grady[ystart] != gradNull){
            by = gauss2Hz * grady[index+ystart] * magn->getY();
        }
        else{
            by = 0;
        }
        if (gradz[zstart] != gradNull){
            bz = gauss2Hz * gradz[index+zstart] * magn->getZ();
        }
        else{
            bz = 0;
        }

        magn->rotate(rfx,rfy,bx+by+bz+magn->getOffset(),tstep);
        magn->acquire(mxout, myout, mzout, ndims, (index+timestart));
        
    };
    
};

CUDA_CALLABLE_MEMBER void mrEvent::acquireEvent(magnetization* magn, int xstart, int ystart,
        int zstart, double* gradx, double* grady,
        double* gradz, double* mxout, double* myout, double* mzout,
        int ndims, int timestart) {
    int index;
    double bx, by, bz;
    for (index = 0; index < nSteps; index++){
        /* No pretty way to implement this. Could do a bunch of cases, but 
         * there would be 2^3 = 8 possibilities, since any number of them 
         * could be null. 
         */
        if (gradx[xstart] != gradNull){
            bx = gauss2Hz * gradx[index+xstart] * magn->getX();
        }
        else{
            bx = 0;
        }
        if (grady[ystart] != gradNull){
            by = gauss2Hz * grady[index+ystart] * magn->getY();
        }
        else{
            by = 0;
        }
        if (gradz[zstart] != gradNull){
            bz = gauss2Hz * gradz[index+zstart] * magn->getZ();
        }
        else{
            bz = 0;
        }

        magn->rotate(0,0,bx+by+bz+magn->getOffset(),tstep);
        magn->acquire(mxout, myout, mzout, ndims, (index+timestart));
        
    };
    
};

CUDA_CALLABLE_MEMBER int mrEvent::getnSteps(){
    return nSteps;
};

CUDA_CALLABLE_MEMBER void mrEvent::thermaleqEvent(magnetization* magn){
    magn->set2eq();
};
CUDA_CALLABLE_MEMBER void mrEvent::refocusEvent(magnetization* magn){
    magn->refocusM();
};