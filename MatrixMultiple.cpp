#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>

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

void MatrixMulitple(MPI_Comm comm, int group)
{
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	//printf("size - %d, rank - %d", size, rank);

	int rowsA = 3;
	int columnA = 7;
	int columnB = 4;
	int rowsB = columnA;
	TMatrix matrixC = createMatrix(rowsA, columnB);

	if(rank == 0)
	{
		double start, end;
		TMatrix matrixA = createMatrix(rowsA, columnA);
		TMatrix matrixB = createMatrix(rowsB, columnB);

		fillSimpleMatrix(matrixA.data, matrixA.rows, matrixA.columns);
		fillSimpleMatrix(matrixB.data, matrixB.rows, matrixB.columns);
		
		// printMatrix(matrixA);
		// printf("\n");
		// printMatrix(matrixB);
		// printf("\n");

		int numberOfSlaves = size - 1;
		start = MPI_Wtime();

		int rowsPerWorker = rowsA / numberOfSlaves;
		int remainingRows = rowsA % numberOfSlaves;
		int offsetRow = 0;
		int messageType = FROM_MASTER;
		
		for(int destination = 1; destination <= numberOfSlaves; destination++)
		{
			int rows = (destination <= remainingRows) ? rowsPerWorker + 1 : rowsPerWorker;
			MPI_Send((void *)&offsetRow, 1, MPI_INT, destination, messageType, comm);
			MPI_Send((void *)&rows, 1, MPI_INT, destination, messageType, comm);

			double* temp = matrixA.data + offsetRow;


			MPI_Send((void *)temp, rows * columnA, MPI_DOUBLE, destination, messageType, comm);
			MPI_Send((void *)matrixB.data, rowsB * columnB, MPI_DOUBLE, destination, messageType, comm);
			offsetRow += rows;
		}

		messageType = FROM_SLAVE;

		for(int source = 1; source <= numberOfSlaves; source++)
		{
			int rowOffset;
			MPI_Status status;

			MPI_Recv((void*)&rowOffset, 1, MPI_INT, source, messageType, comm, &status);
			
			int rows;
			MPI_Recv((void*)&rows, 1, MPI_INT, source, messageType, comm, &status);

			double* temp = (double*)malloc(rows * columnB * sizeof(double));

			MPI_Recv((void*)temp, rows * columnB, MPI_DOUBLE, source, FROM_SLAVE, comm, &status);

			for(int j = 0; j < rows * columnB; j++)
			{
				matrixC.data[rowOffset * columnB + j] = temp[j];
			}
		}

		printMatrix(matrixC);
		end = MPI_Wtime();
		printf("Group: %d; time: %.4f\n",group, end - start);
	}
	else
	{
		int offsetRow;
		MPI_Status status;
		MPI_Recv((void*)&offsetRow, 1, MPI_INT, 0, FROM_MASTER, comm, &status);

		int rows;
		MPI_Recv((void*)&rows, 1, MPI_INT, 0, FROM_MASTER, comm, &status);
		//printf("Process - %d, Offset - %d, Rows - %d\n", rank, offsetRow, rows);

		TMatrix aMatrix = createMatrix(rows, columnA);
		MPI_Recv((void*)aMatrix.data, rows * columnA, MPI_DOUBLE, 0, FROM_MASTER, comm, &status);


		TMatrix bMatrix = createMatrix(rowsB, columnB);
		MPI_Recv((void*)bMatrix.data, rowsB * columnB, MPI_DOUBLE, 0, FROM_MASTER, comm, &status);

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

		MPI_Send((void*)&offsetRow, 1, MPI_INT, 0, FROM_SLAVE, comm);
		MPI_Send((void*)&rows, 1, MPI_INT, 0, FROM_SLAVE, comm);
		MPI_Send((void*)c, rows * columnB, MPI_DOUBLE, 0, FROM_SLAVE, comm);
	}

	
}

bool isContains(int *ms, int n, int key)
{
	for(int i = 0; i < n; i++)
	{
		if(ms[i] == key)
		{
			return true;
		}
	}
	return false;
}

int main(int argc, char** argv)
{
	srand(time(NULL));

	int groupCount = atoi(argv[1]);

	MPI_Init(&argc, &argv);

	int remainingProcess, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &remainingProcess);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(groupCount > remainingProcess / 2)
	{
		printf("Need more process or less group!");
		exit(-1);
	}

	int currentRank = 0;

	MPI_Group originalGroup, newGroup;
	MPI_Comm newComm;

	MPI_Comm_group(MPI_COMM_WORLD, &originalGroup);

	for(int i = 0; i < groupCount; i++)
	{
		int groupInProcess = 2;

		if(i + 1 == groupCount)
		{
			groupInProcess = remainingProcess;
		}
		else
		{
			if(rank == 0)
			{
				int t = remainingProcess - (groupCount - i - 1) * 2 - 2;
				groupInProcess = t == 0 ? 2 : 2 + rand() % t;
				//printf("%d\n",groupInProcess);
			}

			MPI_Bcast(&groupInProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);

			//printf("rank - %d, group in process - %d\n", rank, groupInProcess);
		}

		remainingProcess -= groupInProcess;
		int* temp = (int*)malloc(groupInProcess * sizeof(int));

		for(int j = 0; j < groupInProcess; j++)
		{
			temp[j] = currentRank++;
		}

		MPI_Group_incl(originalGroup, groupInProcess, temp, &newGroup);

		MPI_Comm_create(MPI_COMM_WORLD, newGroup, &newComm);

		if(isContains(temp, groupInProcess, rank))
		{
			//printf("Group - %d, rank - %d\n",i, rank);
			MatrixMulitple(newComm, i);
		}
	}

	MPI_Finalize();
	return 0;	
}