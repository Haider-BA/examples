#include "Vector3d.h"

void Vector3d::set(double xi, double yi, double zi)
{
	x = xi;
	y = yi;
	z = zi;
}

Vector3d Vector3d::get()
{
	return *this;
}

double Vector3d::magnitude()
{
	return sqrt(x*x + y*y + z*z);
}

Vector3d Vector3d::unitVector()
{
	return *this/magnitude();
}

std::ostream& operator<<(std::ostream& out, const Vector3d& object)
{
	out << "(" << object.x << ", " << object.y << ", " << object.z << ")";
	return out;
}

Vector3d operator * (double alpha, const Vector3d& object)
{
	return Vector3d(object.x*alpha, object.y*alpha, object.z*alpha);
}

Vector3d operator * (const Vector3d& object, double alpha)
{
	return Vector3d(object.x*alpha, object.y*alpha, object.z*alpha);
}

Vector3d operator / (const Vector3d& object, double alpha)
{
	return Vector3d(object.x/alpha, object.y/alpha, object.z/alpha);
}

double operator * (const Vector3d& object1, const Vector3d& object2)
{
	return object1.x*object2.x + object1.y*object2.y + object1.z*object2.z;
}
