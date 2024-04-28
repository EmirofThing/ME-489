#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<string.h> 
#include <omp.h>


/* ************************************************************************** */
int Nd, Nc, Np;
double TOL;  

#define BUFSIZE 512

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Define maximum number of iterations
#define MAX_ITER 10000
#define PAD 8

/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  // Error Check
  if (fp == NULL) {
    printf("Error opening the input file\n");
  }

  int sk = 0; double result; 
  char buffer[BUFSIZE], fileTag[BUFSIZE]; 
  
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      sk++;
      return result;
    }
  }

  if(sk==0){
    printf("ERROR! Could not find the tag: [%s] in the file [%s]\n", tag, fileName);
    exit(EXIT_FAILURE); 
  }
}

/* ************************************************************************** */
void readDataFile(char *fileName, double *data){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
  }

  int sk = 0;  
  char buffer[BUFSIZE], fileTag[BUFSIZE]; 
  
  int shift = Nd; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
      if(Nd==2)
        sscanf(buffer, "%lf %lf", &data[sk*shift + 0], &data[sk*shift+1]);
      if(Nd==3)
        sscanf(buffer, "%lf %lf %lf", &data[sk*shift + 0], &data[sk*shift+1], &data[sk*shift+2]);
      if(Nd==4)
        sscanf(buffer, "%lf %lf %lf %lf", &data[sk*shift+0],&data[sk*shift+1], &data[sk*shift+2], &data[sk*shift+3]);
      sk++; 
  }
}

/* ************************************************************************** */
void writeDataToFile(char *fileName, double *data, int *Ci) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
        return;
    }

    // İlk satırı yazdırmadan direkt veri noktalarını yazdırıyoruz
    for (int p = 0; p < Np; p++) {
        fprintf(fp, "%d %d", p, Ci[p]);  // İlk iki sütun: nokta numarası ve küme numarası
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, " %.4f", data[p * Nd + dim]);  // Koordinatlar
        }
        fprintf(fp, "\n");  // Her nokta için yeni satır
    }

    fclose(fp);
}

/* ************************************************************************** */
void writeCentroidToFile(char *fileName, double *Cm){
  FILE *fp = fopen(fileName, "w");
  if (fp == NULL) {
    printf("Error opening the output file\n");
  }

  for(int n=0; n<Nc; n++){
    for(int dim=0; dim<Nd; dim++){
      fprintf(fp, "%.4f ", Cm[n*Nd + dim]);
    }
    fprintf(fp, "\n"); 
  }
  fclose(fp); 
}

/*************************************************************************** */
// Function to calculate Euclidean distance between two points
double distance(const double *a, const double *b, int length) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < length; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

/*************************************************************************** */
void assignPoints(double *data, int *Ci, int *Ck, double *Cm) {
    int *allLocalCk = calloc(omp_get_max_threads() * Nc, sizeof(int));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *localCk = &allLocalCk[tid * Nc * PAD];

        #pragma omp for schedule(static)
        for (int p = 0; p < Np; p++) {
            double min_distance = INFINITY;
            int cluster_index = 0;
            for (int n = 0; n < Nc; n++) {
                double d = distance(&data[p * Nd], &Cm[n * Nd], Nd);
                if (d < min_distance) {
                    min_distance = d;
                    cluster_index = n;
                }
            }
            Ci[p] = cluster_index;
            localCk[cluster_index]++;
        }

        #pragma omp critical
        for (int n = 0; n < Nc; n++) {
            Ck[n] += localCk[n];
        }
    }
    free(allLocalCk);
}



/*************************************************************************** */
double updateCentroids(double *data, int *Ci, int *Ck, double *Cm) {
    double *newCm = calloc(Nc * Nd, sizeof(double));

    #pragma omp parallel
    {
        double *localNewCm = calloc(Nc * Nd, sizeof(double));

        #pragma omp for nowait
        for (int p = 0; p < Np; p++) {
            int cluster_index = Ci[p];
            for (int dim = 0; dim < Nd; dim++) {
                localNewCm[cluster_index * Nd + dim] += data[p * Nd + dim];
            }
        }

        #pragma omp critical
        for (int n = 0; n < Nc; n++) {
            for (int dim = 0; dim < Nd; dim++) {
                newCm[n * Nd + dim] += localNewCm[n * Nd + dim];
            }
        }
        free(localNewCm);
    }

    double err = 0.0;
    for (int n = 0; n < Nc; n++) {
        for (int dim = 0; dim < Nd; dim++) {
            double oldVal = Cm[n * Nd + dim];
            if (Ck[n] > 0) {
                Cm[n * Nd + dim] = newCm[n * Nd + dim] / Ck[n];
            }
            double diff = fabs(Cm[n * Nd + dim] - oldVal);
            if (diff > err) err = diff;
        }
    }

    free(newCm);
    return err;
}


/*************************************************************************** */
// Function to perform k-means clustering
void kMeans(double *data, int *Ci, int *Ck, double *Cm) {
    // İş parçacığı başlatma maliyetini azaltmak için tek bir paralel bölge
    #pragma omp parallel
    {
        // Noktaların atanması
        #pragma omp for schedule(dynamic, 10)
        for (int p = 0; p < Np; p++) {
            double min_distance = INFINITY;
            int cluster_index = 0;
            for (int n = 0; n < Nc; n++) {
                double d = distance(&data[p * Nd], &Cm[n * Nd]);
                if (d < min_distance) {
                    min_distance = d;
                    cluster_index = n;
                }
            }
            Ci[p] = cluster_index;
            #pragma omp atomic
            Ck[cluster_index]++;
        }

        // Yerel merkez güncellemeleri
        double *localCm = calloc(Nc * Nd, sizeof(double));
        #pragma omp for schedule(static)
        for (int p = 0; p < Np; p++) {
            int cluster_index = Ci[p];
            for (int dim = 0; dim < Nd; dim++) {
                localCm[cluster_index * Nd + dim] += data[p * Nd + dim];
            }
        }

        // Küresel merkezlerin güncellenmesi
        #pragma omp critical
        for (int n = 0; n < Nc; n++) {
            for (int dim = 0; dim < Nd; dim++) {
                Cm[n * Nd + dim] += localCm[n * Nd + dim];
            }
        }
        free(localCm);
    }

    // Küresel merkezleri normalize et
    for (int n = 0; n < Nc; n++) {
        if (Ck[n] > 0) {
            for (int dim = 0; dim < Nd; dim++) {
                Cm[n * Nd + dim] /= Ck[n];
            }
        }
    }
}



/*************************************************************************** */

int main(int argc, char *argv[]) {
    if(argc != 3) {
        printf("Usage: %s input.dat data.dat\n", argv[0]);
        return -1;
    }

    // Read input parameters
    Np = (int) readInputFile(argv[1], "NUMBER_OF_POINTS");
    Nc = (int) readInputFile(argv[1], "NUMBER_OF_CLUSTERS");
    Nd = (int) readInputFile(argv[1], "DATA_DIMENSION");
    TOL = readInputFile(argv[1], "TOLERANCE");

    // Allocate memory for data points and clusters
    double *data = (double*) malloc(Np * Nd * sizeof(double));
    int *Ci = (int *) calloc(Np, sizeof(int)); // Cluster index for each point
    int *Ck = (int *) calloc(Nc, sizeof(int)); // Number of points in each cluster
    double *Cm = (double*) calloc(Nc * Nd, sizeof(double)); // Centroids

    // Load data points from file
    readDataFile(argv[2], data);

    // Measure the start time
    double startTime = omp_get_wtime();

    // Perform k-means clustering
    kMeans(data, Ci, Ck, Cm);

    // Measure the end time and compute the total time
    double endTime = omp_get_wtime();
    double totalTime = endTime - startTime;

    // Get the number of threads used
    int numThreads = omp_get_max_threads();

    // Print total compute time and thread usage
    printf("Total Compute Time: %.8f using %d threads\n", totalTime, numThreads);

    // Print cluster details
    for(int n = 0; n < Nc; n++) {
        printf("(%d of %d) points are in the cluster %d with centroid(", Ck[n], Np, n);
        for(int dim = 0; dim < Nd; dim++) {
            printf(" %f", Cm[n * Nd + dim]);
            if (dim < Nd - 1) printf(", ");
        }
        printf(")\n");
    }

    // Write output to files
    writeDataToFile("output.dat", data, Ci);
    writeCentroidToFile("centroids.dat", Cm);

    // Clean up
    free(data);
    free(Ci);
    free(Ck);
    free(Cm);

    return 0;
}