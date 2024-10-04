
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "Particle.h"
#include "Vector.h"
#include <fstream>
#include <filesystem>

/*Решение задачи N - тел методом Particle - Particle(пр¤мого интегрировани¤).
моделируетс¤ трёхмерное пространство,
обезразмеривание осуществлено с соображением G = 1,
уравнения движения решаются методом Эйлера.
используется симметрично-противоположна¤ матрица гравитационных взаимодействий force,
благодар¤ которой количество необходимых вычислений уменьшаетс¤ в два раза,
в соответствии с третьим законом Ќьютона (Fij = -Fji)*/

/*Solution of N-body problem with Particle-Particle (direct sum) method.
This program calculates in 3D space,
with nondimensialization (G = 1),
equations of motion are solved with Euler's method.
Forces Fij and Fji are treated as equal with different signs, due to Newton's 3rd law. That halves the amount of calculations.*/


Particle* InitializeNBodySystem(const std::string path, int& n);

double Cube(double number);

Vector Sum(Vector* sequence, int size);
Vector Sum(Vector* sequence, int first, int size);

__global__ void calculateForce(Vector* force, Particle* particles, const size_t size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < size && col < size && row < col)
	{
		double distanceX = particles[col].position.x - particles[row].position.x;
		double distanceY = particles[col].position.y - particles[row].position.y;
		double distanceZ = particles[col].position.z - particles[row].position.z;

		double vector = sqrt(distanceX * distanceX + distanceY * distanceY + distanceZ * distanceZ);
		double denominator = vector * vector * vector;

		force[row * size + col].x = distanceX * particles[row].mass * particles[col].mass / denominator;
		force[row * size + col].y = distanceY * particles[row].mass * particles[col].mass / denominator;
		force[row * size + col].z = distanceZ * particles[row].mass * particles[col].mass / denominator;

		force[col * size + row].x = -force[row * size + col].x;
		force[col * size + row].y = -force[row * size + col].y;
		force[col * size + row].z = -force[row * size + col].z;
	}
}

int main()
{
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	int n;
	double timeStep = 0.01;

	Particle* particles = InitializeNBodySystem("Particles.txt", n);

	Vector* force = new Vector[n * n];

	const size_t sizeForceBytes = n * n * sizeof(Vector);
	const size_t sizeParticlesBytes = n * sizeof(Particle);

	Particle* particlesDevice;
	Vector* forceDevice;

	const int blockSize = 32;

	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(n / blockSize + 1, n / blockSize + 1);
	

	std::filesystem::path path = L"coordinates";
	if (std::filesystem::exists(path))
	{
		std::filesystem::remove_all(path);
	}

	if (!std::filesystem::create_directory(path))
	{
		printf("Error making a directory\n");
		return 1;
	}

	double time = 0.0;
	for (;;)
	{
		std::ofstream fileCoordinates;
		std::string timeStr = std::to_string(time);
		fileCoordinates.open("coordinates\\" + timeStr + ".csv");
		fileCoordinates << "x;y;z\n";

		cudaStatus = cudaMalloc((void**)&particlesDevice, sizeParticlesBytes);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return 1;
		}

		cudaStatus = cudaMalloc((void**)&forceDevice, sizeForceBytes);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			return 1;
		}

		cudaStatus = cudaMemcpy(particlesDevice, particles, sizeParticlesBytes, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}


		for (int i = 0; i < n; ++i)
		{
			force[i * n + i].SetZeroVector();
		}

		calculateForce <<<dimGrid, dimBlock>>> (forceDevice, particlesDevice, n);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return -1;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			return 1;
		}

		cudaStatus = cudaMemcpy(force, forceDevice, sizeForceBytes, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return 1;
		}



		for (int i = 0; i < n; ++i)
		{
			fileCoordinates << particles[i].position << "\n";

			particles[i].acceleration = Sum(force, i * n, i * n + n) / particles[i].mass;

			particles[i].velocity = particles[i].velocity + particles[i].acceleration * timeStep;

			particles[i].position = particles[i].position + particles[i].velocity * timeStep;
		}

		time += timeStep;

		cudaFree(particlesDevice);
		cudaFree(forceDevice);

		fileCoordinates.close();
	}


	delete[] force;
	delete[] particles;
	return 0;
}

Particle* InitializeNBodySystem(const std::string path, int& n)
{
	std::ifstream fileParticles;
	fileParticles.open(path);

	char tempString[256];
	fileParticles.getline(tempString, 256, ':');

	fileParticles >> n;
	Particle* particles = new Particle[n];

	fileParticles.get();
	fileParticles.get();

	fileParticles.getline(tempString, 256);

	for (int i = 0; i < n; ++i)
	{
		fileParticles >> particles[i].mass;
		fileParticles.get();
		fileParticles >> particles[i].velocity.x >> particles[i].velocity.y >> particles[i].velocity.z;
		fileParticles.get();
		fileParticles >> particles[i].position.x >> particles[i].position.y >> particles[i].position.z;
	}

	fileParticles.close();
	return particles;
}

double Cube(double number)
{
	return number * number * number;
}

Vector Sum(Vector* sequence, int size)
{
	Vector sum;
	sum.x = .0;
	sum.y = .0;
	sum.z = .0;

	for (int i = 0; i < size; ++i)
	{
		sum = sum + sequence[i];
	}

	return sum;
}

Vector Sum(Vector* sequence, int first, int last)
{
	Vector sum;
	sum.x = .0;
	sum.y = .0;
	sum.z = .0;

	for (int i = first; i < last; ++i)
	{
		sum = sum + sequence[i];
	}

	return sum;
}

