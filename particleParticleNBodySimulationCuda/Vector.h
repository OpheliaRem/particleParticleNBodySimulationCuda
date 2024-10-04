#pragma once
#include <cmath>
#include <fstream>

struct Vector
{
	double x;
	double y;
	double z;

	double Abs() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	Vector operator+(const Vector& other) const
	{
		Vector newVector;
		newVector.x = x + other.x;
		newVector.y = y + other.y;
		newVector.z = z + other.z;
		return newVector;
	}

	Vector operator-(const Vector& other) const
	{
		Vector newVector;
		newVector.x = x - other.x;
		newVector.y = y - other.y;
		newVector.z = z - other.z;
		return newVector;
	}

	Vector operator*(double scalar) const
	{
		Vector newVector;
		newVector.x = x * scalar;
		newVector.y = y * scalar;
		newVector.z = z * scalar;
		return newVector;
	}

	Vector operator/(double scalar) const
	{
		Vector newVector;
		newVector.x = x / scalar;
		newVector.y = y / scalar;
		newVector.z = z / scalar;
		return newVector;
	}

	Vector operator-() const
	{
		Vector newVector;
		newVector.x = -x;
		newVector.y = -y;
		newVector.z = -z;
		return newVector;
	}

	friend std::ofstream& operator<<(std::ofstream& file, const Vector& vector)
	{
		file << vector.x << ";" << vector.y << ";" << vector.z;
		return file;
	}

	void SetZeroVector()
	{
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}
};