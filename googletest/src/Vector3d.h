#pragma once

#include <cmath>
#include <ostream>

class Vector3d
{
	double x, y, z;

public:
	Vector3d(){};
	Vector3d(double xi, double yi, double zi) : x(xi), y(yi), z(zi){};
	void set(double xi, double yi, double zi);
	Vector3d get();

	double magnitude();
	Vector3d unitVector();

	friend std::ostream& operator<<(std::ostream& out, const Vector3d& object);
	friend Vector3d operator * (double alpha, const Vector3d& object);
	friend Vector3d operator * (const Vector3d& object, double alpha);
	friend Vector3d operator / (const Vector3d& object, double alpha);
	friend double operator * (const Vector3d& object1, const Vector3d& object2);
};