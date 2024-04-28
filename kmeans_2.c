#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>


// Function prototypes
void readInput(const char *filename, int *N, int *K, int *d, double *tol);
void readData(const char *filename, double **data, int N, int d);
void kmeans(double **data, int N, int K, int d, double tol, int *labels, double **centroids);
double euclideanDistance(const double *point1, const double *point2, int d);
int assignPointsToClusters(double **data, double **centroids, int *labels, int N, int K, int d);
void computeCentroids(double **data, double **centroids, int *labels, int N, int K, int d, int *clusterSizes);
double centroidChange(double **oldCentroids, double **newCentroids, int K, int d);
void printResults(double **data, int *labels, int N, int d);
void printClusterInfo(double **centroids, int *labels, int N, int K, int d);
void writeResultsToCSV(const char* filename, double **data, int *labels, int N, int d);
void writeCentroidsToCSV(const char* filename, double **centroids, int K, int d);


int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input file> <data file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N, K, d;
    double tol;
    readInput(argv[1], &N, &K, &d, &tol);

    double **data = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; ++i) {
        data[i] = (double *)malloc(d * sizeof(double));
    }
    int *labels = (int *)malloc(N * sizeof(int));

    // Allocate memory for centroids
    double **centroids = (double **)malloc(K * sizeof(double *));
    for (int k = 0; k < K; ++k) {
        centroids[k] = (double *)malloc(d * sizeof(double));
    }

    readData(argv[2], data, N, d);

    // Execute k-means algorithm
    kmeans(data, N, K, d, tol, labels, centroids);

    // Print results to the console (if needed)
    // printResults(data, labels, N, d);
    printClusterInfo(centroids, labels, N, K, d);

    // Write clustering results to CSV
    writeResultsToCSV("clustering_results.csv", data, labels, N, d);

    // Write centroids to CSV
    writeCentroidsToCSV("centroids.csv", centroids, K, d);

    // Free dynamically allocated memory
    for (int i = 0; i < N; ++i) {
        free(data[i]);
    }
    free(data);
    free(labels);

    for (int k = 0; k < K; ++k) {
        free(centroids[k]);
    }
    free(centroids);

    return EXIT_SUCCESS;
}



void skipLine(FILE *file) {
    char c;
    while ((c = fgetc(file)) != '\n' && c != EOF) { /* Satır sonuna kadar oku */ }
}

void readInput(const char *filename, int *N, int *K, int *d, double *tol) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening input file");
        exit(EXIT_FAILURE);
    }

    skipLine(file); // [NUMBER_OF_POINTS] başlığını atla
    fscanf(file, "%d", N);
    skipLine(file); skipLine(file); // [NUMBER_OF_CLUSTERS] başlığını atla
    fscanf(file, "%d", K);
    skipLine(file); skipLine(file); // [DATA_DIMENSION] başlığını atla
    fscanf(file, "%d", d);
    skipLine(file); skipLine(file); // [TOLERANCE] başlığını atla
    fscanf(file, "%lf", tol);

    fclose(file);
    printf("Successfully read input parameters: N=%d, K=%d, d=%d, tol=%lf\n", *N, *K, *d, *tol);
}




void readData(const char *filename, double **data, int N, int d) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening data file");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            if (fscanf(file, "%lf", &data[i][j]) != 1) {
                fprintf(stderr, "Failed to read data for point %d\n", i);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void kmeans(double **data, int N, int K, int d, double tol, int *labels, double **centroids) {
    double start_time = omp_get_wtime();
    double **oldCentroids = (double **)malloc(K * sizeof(double *));
    int *clusterSizes = (int *)calloc(K, sizeof(int));
    for (int k = 0; k < K; ++k) {
        oldCentroids[k] = (double *)malloc(d * sizeof(double));
    }

    // Initialize centroids randomly or by some strategy
    for (int k = 0; k < K; ++k) {
        for (int dim = 0; dim < d; ++dim) {
            centroids[k][dim] = data[rand() % N][dim];
        }
    }

    double maxChange;
    do {
        assignPointsToClusters(data, centroids, labels, N, K, d);
        computeCentroids(data, centroids, labels, N, K, d, clusterSizes);

        // Calculate the maximum change among all centroids
        maxChange = centroidChange(oldCentroids, centroids, K, d);

        // Prepare for the next iteration
        for (int k = 0; k < K; ++k) {
            for (int dim = 0; dim < d; ++dim) {
                oldCentroids[k][dim] = centroids[k][dim];
            }
        }
    } while (maxChange > tol);

    // Cleanup
    for (int k = 0; k < K; ++k) {
        free(oldCentroids[k]);
    }
    free(oldCentroids);
    free(clusterSizes);

    double end_time = omp_get_wtime();
    printf("Elapsed time : %f seconds\n", end_time - start_time);
}



double euclideanDistance(const double *point1, const double *point2, int d) {
    double sum = 0.0;
    for (int i = 0; i < d; ++i) {
        sum += pow(point1[i] - point2[i], 2);
    }
    return sqrt(sum);
}

int assignPointsToClusters(double **data, double **centroids, int *labels, int N, int K, int d) {
    int changed = 0;
    #pragma omp parallel for reduction(+:changed)
    for (int i = 0; i < N; ++i) {
        double minDist = DBL_MAX;
        int bestCluster = 0;
        for (int k = 0; k < K; ++k) {
            double dist = euclideanDistance(data[i], centroids[k], d);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = k;
            }
        }
        int oldLabel = labels[i];
        if (oldLabel != bestCluster) {
            labels[i] = bestCluster;
            changed = 1;
        }
    }
    return changed;
}


void computeCentroids(double **data, double **centroids, int *labels, int N, int K, int d, int *clusterSizes) {
    int *count = (int *)calloc(K, sizeof(int));  // Bellek ayırma işlemi count dizisi için de gerekebilir

    // Reset centroids and cluster sizes
    #pragma omp parallel for
    for (int k = 0; k < K; ++k) {
        for (int dim = 0; dim < d; ++dim) {
            centroids[k][dim] = 0.0;
        }
        clusterSizes[k] = 0;
    }

    // Sum up and count data points for each cluster
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int cluster = labels[i];
        #pragma omp atomic
        clusterSizes[cluster]++;  // Ensure atomic update to avoid race condition
        for (int dim = 0; dim < d; ++dim) {
            #pragma omp atomic
            centroids[cluster][dim] += data[i][dim];
        }
    }

    // Divide by count to get the average (centroid)
    #pragma omp parallel for
    for (int k = 0; k < K; ++k) {
        if (clusterSizes[k] > 0) {
            for (int dim = 0; dim < d; ++dim) {
                centroids[k][dim] /= clusterSizes[k];
            }
        }
    }

    free(count);
}




double centroidChange(double **oldCentroids, double **newCentroids, int K, int d) {
    double maxChange = 0.0;
    for (int k = 0; k < K; ++k) {
        double dist = euclideanDistance(oldCentroids[k], newCentroids[k], d);
        if (dist > maxChange) {
            maxChange = dist;
        }
    }
    return maxChange;
}


void printResults(double **data, int *labels, int N, int d) {
    // If you have printing of each point's cluster assignment here, comment it out or remove it.
    /*
    printf("Point, Cluster\n");
    for (int i = 0; i < N; ++i) {
        printf("%d, %d\n", i, labels[i]);
    }
    */
}


void printClusterInfo(double **centroids, int *labels, int N, int K, int d) {
    int *clusterSizes = calloc(K, sizeof(int));
    if (!clusterSizes) {
        perror("Memory allocation for cluster sizes failed");
        exit(EXIT_FAILURE);
    }

    // Her kümedeki nokta sayısını say
    for (int i = 0; i < N; ++i) {
        clusterSizes[labels[i]]++;
    }

    // Her küme için bilgiyi yazdır
    for (int k = 0; k < K; ++k) {
        printf("(%d of %d) points are in the cluster %d with centroid(", clusterSizes[k], N, k);
        for (int dim = 0; dim < d; ++dim) {
            printf(" %f", centroids[k][dim]);
            if (dim < d - 1) printf(",");
        }
        printf(" )\n");
    }

    free(clusterSizes);
}

void writeResultsToCSV(const char* filename, double **data, int *labels, int N, int d) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    // Assuming 2D data for simplicity
    for (int i = 0; i < N; ++i) {
        fprintf(file, "%f,%f,%d\n", data[i][0], data[i][1], labels[i]);
    }

    fclose(file);
}


void writeCentroidsToCSV(const char* filename, double **centroids, int K, int d) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file to write centroids!\n");
        exit(EXIT_FAILURE);
    }

    for (int k = 0; k < K; ++k) {
        for (int dim = 0; dim < d; ++dim) {
            fprintf(file, "%f", centroids[k][dim]);
            if (dim < d - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}
