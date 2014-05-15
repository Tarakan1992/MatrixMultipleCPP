#include "stdio.h"
#include <mpi.h>
#include "stdlib.h"
#include <sys/time.h>

#define FROM_MASTER 10
#define FROM_SLAVE 20

struct TMatrix
{
	int rows;
	int columns;
	double* data;
};

TMatrix createMatrix(int rows, int columns)
{
	TMatrix temp;
	temp.data = (double *) malloc(rows * columns * sizeof(double));
	temp.rows = rows;
	temp.columns = columns;
	return temp;
}


void printMatrix(TMatrix matrix)
{
	for(int i = 0; i < matrix.rows; i++)
	{
		for(int j = 0; j < matrix.columns; j++)
		{
			printf("%.2f ", matrix.data[i * matrix.columns + j]);
		}
		printf("\n");
	}
}

void fillSimpleMatrix(double* data, int rows, int columns)
{
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < columns; j++)
		{
			data[i * columns + j] = 1.0;
		}
	}
}



int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("I am process number - %d\n", rank);

	int rowsA = 9;
	int columnA = 4;
	int columnB = 9;
	int rowsB = columnA;
	TMatrix matrixC = createMatrix(rowsA, columnB);

	if(rank == 0)
	{
		double start, end;
		TMatrix matrixA = createMatrix(rowsA, columnA);
		TMatrix matrixB = createMatrix(rowsB, columnB);

		fillSimpleMatrix(matrixA.data, matrixA.rows, matrixA.columns);
		fillSimpleMatrix(matrixB.data, matrixB.rows, matrixB.columns);
		
		printMatrix(matrixA);
		printf("\n");
		printMatrix(matrixB);
		printf("\n");

		int numberOfSlaves = size - 1;
		start = MPI_Wtime();

		int rowsPerWorker = rowsA / numberOfSlaves;
		int remainingRows = rowsA % numberOfSlaves;
		int offsetRow = 0;
		int messageType = FROM_MASTER;
		
		for(int destination = 1; destination <= numberOfSlaves; destination++)
		{
			int rows = (destination <= remainingRows) ? rowsPerWorker + 1 : rowsPerWorker;
			MPI_Send((void *)&offsetRow, 1, MPI_INT, destination, messageType, MPI_COMM_WORLD);
			MPI_Send((void *)&rows, 1, MPI_INT, destination, messageType, MPI_COMM_WORLD);

			double* temp = matrixA.data + offsetRow;


			MPI_Send((void *)temp, rows * columnA, MPI_DOUBLE, destination, messageType, MPI_COMM_WORLD);
			MPI_Send((void *)matrixB.data, rowsB * columnB, MPI_DOUBLE, destination, messageType, MPI_COMM_WORLD);
			offsetRow += rows;
		}

		messageType = FROM_SLAVE;

		for(int source = 1; source <= numberOfSlaves; source++)
		{
			int rowOffset;
			MPI_Status status;

			MPI_Recv((void*)&rowOffset, 1, MPI_INT, source, messageType, MPI_COMM_WORLD, &status);
			
			int rows;
			MPI_Recv((void*)&rows, 1, MPI_INT, source, messageType, MPI_COMM_WORLD, &status);

			double* temp = (double*)malloc(rows * columnB * sizeof(double));

			MPI_Recv((void*)temp, rows * columnB, MPI_DOUBLE, source, FROM_SLAVE, MPI_COMM_WORLD, &status);

			for(int j = 0; j < rows * columnB; j++)
			{
				matrixC.data[rowOffset * columnB + j] = temp[j];
			}
		}

		printMatrix(matrixC);
		printf("\n");
		end = MPI_Wtime();
		printf("time: %.4f\n", end - start);
	}
	else
	{
		int offsetRow;
		MPI_Status status;
		MPI_Recv((void*)&offsetRow, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, &status);

		int rows;
		MPI_Recv((void*)&rows, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, &status);
		printf("Process - %d, Offset - %d, Rows - %d\n", rank, offsetRow, rows);

		TMatrix aMatrix = createMatrix(rows, columnA);
		MPI_Recv((void*)aMatrix.data, rows * columnA, MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, &status);


		TMatrix bMatrix = createMatrix(rowsB, columnB);
		MPI_Recv((void*)bMatrix.data, rowsB * columnB, MPI_DOUBLE, 0, FROM_MASTER, MPI_COMM_WORLD, &status);

		double* c =(double*)malloc(rows * columnB * sizeof(double));

		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < columnB; j++)
			{
				c[i * columnB + j] = 0.0;
				for(int k = 0; k < columnA; k++)
				{
					c[i * columnB + j] += aMatrix.data[i * columnA + k] * bMatrix.data[j * columnA + k];
				}

			}
		}

		MPI_Send((void*)&offsetRow, 1, MPI_INT, 0, FROM_SLAVE, MPI_COMM_WORLD);
		MPI_Send((void*)&rows, 1, MPI_INT, 0, FROM_SLAVE, MPI_COMM_WORLD);
		MPI_Send((void*)c, rows * columnB, MPI_DOUBLE, 0, FROM_SLAVE, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	
	return 0;	
}