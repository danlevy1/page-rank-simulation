#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

int main (int argc, char *argv[]) {
    // Initialize MPI and get number of processes and my number or rank
    int numProcs, myID;                                                                 // MPI information
    MPI_Init(&argc,&argv);                                                              // Initialize MPI
    MPI_Comm_size(MPI_COMM_WORLD,&numProcs);                                            // Get number of processes
    MPI_Comm_rank(MPI_COMM_WORLD,&myID);                                                // Get my ID

    // Initialize variables
    int i, j, k, iStart, iMax;                                                          // Loop variables
    int numPages = 16;                                                                  // Number of pages on the WWW
    double dampingFactor = 0.15;                                                        // Probablility that the surfer picks any page at random
    int numMatvec = 1000;                                                               // Number of multiplications
    double matVecStartTime;                                                             // Start time of timer
    double matVecTime;                                                                  // Total matvec time
    int maxPagesPerProc = numPages / numProcs;                                          // Maximum number of pages (dense rows) per proc
    int maxLinksPerProc = (maxPagesPerProc) * 2;                                        // Maximum number of links per process
    int *rowIndexVector = (int *)malloc(sizeof(int) * (maxPagesPerProc) + 1);  ;        // Row index vector
    double *xVectorGlobal = (double *)malloc(sizeof(double) * numPages);                // Global multiply vector
    double *yVector = (double *)malloc(sizeof(double) * numPages);                      // Page rank vector
    double *rankMatrix;                                                                 // Sparse rank matrix
    int *columnIndexVector;                                                             // Column index vector
    if (myID == 0) {
        rankMatrix = (double *)malloc(sizeof(double) * (maxLinksPerProc - 1));          // rankMatrix size for process 0 is 1 smaller than all other processes
        columnIndexVector = (int *)malloc(sizeof(int) * (maxLinksPerProc - 1));         // columnIndexVector size for process 0 is 1 smaller than all other processes
    } else {
        rankMatrix = (double *)malloc(sizeof(double) * maxLinksPerProc);
        columnIndexVector = (int *)malloc(sizeof(int) * maxLinksPerProc);
    }

    // Checks if any memory allocation failed
    if (rankMatrix == NULL || rowIndexVector == NULL || columnIndexVector == NULL || xVectorGlobal == NULL || yVector == NULL) {
        printf("Memory allocation for rankMatrix, rowIndexVector, columnIndexVector, xVectorGlobal, or yVector failed");
        return 1;
    }

    // Initializes xVectorGlobal elements to 1 / numPages
    for (i = 0; i < numPages; i ++) {
        xVectorGlobal[i] = (double) 1 / (double) numPages;
    }

    // Initializes all rankRatrix elements to 0.5 with damping factor
    iMax = maxLinksPerProc;
    if (myID == 0) {                                                                    // Process 0 holds 1 less rankMatrix element
        iMax --;
    }
    for (i = 0; i < iMax; i ++) {                                                       // Initializes rankMatrix elements
        rankMatrix[i] = (1 - dampingFactor) * 0.5;
    }
    // Updates rankMatrix with damping factor for link from page numPages to page numPages - 1
    if (numPages == numProcs) {                                                         // If there is 1 page / process, indexing is different for last element
        if (myID == numProcs - 2) {                                                     // Updates element on second-to-last process
            rankMatrix[1] = (1 - dampingFactor) * 1.0;
        }
    } else if (myID == numProcs - 1) {                                                  // Otherwise, wait update on last process
        if (myID == 0) {                                                                // If only running on 1 process, indexing is different for last element
            rankMatrix[maxLinksPerProc - 4] = (1 - dampingFactor) * 1.0;
        } else {                                                                        // Updates element in all other cases
            rankMatrix[maxLinksPerProc - 3] = (1 - dampingFactor) * 1.0;
        }
    }

    // Sets up rowIndexVector
    iStart = 0;
    if (myID == 0) {                                                                    // Process 0 has odd-numbered row indices (except for first element)
        // First two elements are "custom"
        rowIndexVector[0] = 0;
        rowIndexVector[1] = 1;
        j = 3;                                                                          // Start row indexing at 3
        iStart = 2;                                                                     // Start loop at 2 (elements 0 and 1 were just initialized)
    } else {                                                                            // Start row indexing at 0 if myID is not 0
        j = 0;
    }
    for (i = iStart; i < (maxPagesPerProc) + 1; i ++) {                                 // Initializes row index elements
        rowIndexVector[i] = j;
        j += 2;                                                                         // Increment row index by 2
    }

    // Sets up columnIndexVector
    iMax = maxPagesPerProc;                                                             // Maximum i value for the following loop
    if (myID == numProcs - 1) {                                                         // Process 0 holds 1 less columnIndexVector element
        iMax --;
    }
    j = 0;                                                                              // Column index counter
    for (i = 0; i < iMax; i ++) {                                                       // Initializes column index elements
        int diagonalIndex = (maxPagesPerProc * myID) + i;                               // Computes equivalent diagonal index if the matrix were dense
        if (diagonalIndex == 0) {                                                       // If diagonal index is 0 (first row), initialize one element to 1
            columnIndexVector[0] = 1;
            j ++;
        } else {                                                                        // Initialize next two elements (left and right of diagonal)
            columnIndexVector[j] = diagonalIndex - 1;
            j ++;
            columnIndexVector[j] = diagonalIndex + 1; 
            j ++;
        }
    }
    if (myID == numProcs - 1) {                                                         // The last process has two custom last values
        if (myID == 0) {                                                                // If this is process 0, columnIndexVector indices are different
            columnIndexVector[maxLinksPerProc - 3] = 0;
            columnIndexVector[maxLinksPerProc - 2] = numPages - 2;
        } else {
            columnIndexVector[maxLinksPerProc - 2] = 0;
            columnIndexVector[maxLinksPerProc - 1] = numPages - 2;
        }
    }

    // Starts matvec timer
    if (myID == 0) {
        matVecStartTime = MPI_Wtime();
    }

    // Multiplies rankMatrix by xVector to get yVector (loops numMatvec times)
    for (i = 0; i < numMatvec; i ++) {
        for (j = 0; j < maxPagesPerProc; j ++) {
            yVector[j] = 0.0;
            for (k = rowIndexVector[j]; k < rowIndexVector[j + 1]; k ++) {
                yVector[j] += rankMatrix[k] * xVectorGlobal[columnIndexVector[k]];
                // yVector[j] += rankMatrix[k] * xVector[columnIndexVector[k]];
                // printf("myID = %d, xVector index needed = %d, xVector index computed = %d\n", myID, columnIndexVector[k], j);
            }
        }
        // Adds damping factor to yVector
        for (j = 0; j < numPages; j ++) {
            yVector[j] += dampingFactor / numPages;
        }
        // Copies yVector elements to xVectorGlobal
        for (j = 0; j < maxPagesPerProc; j ++) {
            xVectorGlobal[(myID * maxPagesPerProc) + j] = yVector[j];
        }

        // Send and receive elements needed for next matvec
        if (numProcs > 1) {                                                                                                                 // No communication needed with just 1 process
            if (myID == 0) {                                                                                                                // Process 0 has custom sends and receives
                MPI_Send(&xVectorGlobal[0], 1, MPI_DOUBLE, numProcs - 1, 0, MPI_COMM_WORLD);                                                // Sends xVectorGlobal[0] element to process numProcs - 1
                MPI_Send(&xVectorGlobal[maxPagesPerProc - 1], 1, MPI_DOUBLE, myID + 1, 0, MPI_COMM_WORLD);                                  // Sends xVectorGlobal[maxPagesPerProc - 1] element to process myID + 1
                MPI_Recv(&xVectorGlobal[maxPagesPerProc], 1, MPI_DOUBLE, myID + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                   // Receives xVectorGlobal[maxPagesPerProc] element to process myID + 1
            } else if (myID == numProcs - 1) {                                                                                              // Process numProcs - 1 has custom sends and receives
                MPI_Send(&xVectorGlobal[myID * maxPagesPerProc], 1, MPI_DOUBLE, myID - 1, 0, MPI_COMM_WORLD);                               // Sends xVectorGlobal[myID * maxPagesPerProc] element to process myID - 1
                MPI_Recv(&xVectorGlobal[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);                                        // Receives xVectorGlobal[0] element from process 0
                MPI_Recv(&xVectorGlobal[(myID * maxPagesPerProc) - 1], 1, MPI_DOUBLE, myID - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      // Receives xVectorGlobal[(myID * maxPagesPerProc) - 1] element from process myID - 1
            } else {                                                                                                                        // Sends and receives for all other processes
                MPI_Send(&xVectorGlobal[myID * maxPagesPerProc], 1, MPI_DOUBLE, myID - 1, 0, MPI_COMM_WORLD);                               // Sends xVectorGlobal[myID * maxPagesPerProc] element to process myID - 1
                MPI_Send(&xVectorGlobal[maxPagesPerProc * (myID + 1) - 1], 1, MPI_DOUBLE, myID + 1, 0, MPI_COMM_WORLD);                     // Sends xVectorGlobal[maxPagesPerProc * (myID + 1) - 1] element to process myID + 1
                MPI_Recv(&xVectorGlobal[(myID * maxPagesPerProc) - 1], 1, MPI_DOUBLE, myID - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      // Receives xVectorGlobal[(myID * maxPagesPerProc) - 1] element from myID - 1
                MPI_Recv(&xVectorGlobal[maxPagesPerProc * (myID + 1)], 1, MPI_DOUBLE, myID + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      // Receives xVectorGlobal[maxPagesPerProc * (myID + 1)] element from myID + 1
            }
        }
    }

    // Stops matvec timer
    if (myID == 0) {
        matVecTime = MPI_Wtime() - matVecStartTime;
    }

    // Gathers xVectorGlobal elements from each process and puts them into process 0's xVectorGlobal
    if (numProcs > 1) {                                                                                                                     // No communication needed with just 1 process
        if (myID == 0) {                                                                                                                    // Process 0 receives all elements
            for (i = 1; i < numProcs; i ++) {
                MPI_Recv(&xVectorGlobal[(i * maxPagesPerProc)], maxPagesPerProc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      // Receives maxPagesPerProc elements from processes 1 through numProcs - 1
            }
        } else {                                                                                                                            // All other processes send their elements
            MPI_Send(&xVectorGlobal[(myID * maxPagesPerProc)], maxPagesPerProc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);                          // Sends maxPagesPerProc elements to process 0
        }
    }

    // Process 0 prints final results
    if (myID == 0) {
        // Prints final page ranks
        for (i = 0; i < numPages; i ++) {
            printf("Page %d Rank: %.20f\n", i + 1, xVectorGlobal[i]);
        }

        // Prints minimum and maximum page ranks
        double min = DBL_MAX;
        double max = -DBL_MAX;
        for (i = 0; i < numPages; i ++) {
            if (xVectorGlobal[i] < min) {
                min = xVectorGlobal[i];
            }
            if (xVectorGlobal[i] > max) {
                max = xVectorGlobal[i];
            }
        }
        printf("Min Page Rank: %f\n", min);
        printf("Max Page Rank: %f\n", max);

        printf("Time: %f Seconds\n", matVecTime);   // Prints the matvec computational time
    }
    if (myID == 0) {
         printf("\n\n\nDone.");
    }
    MPI_Finalize();
}
