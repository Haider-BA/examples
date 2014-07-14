#include <iostream>
#include "Vector3d.h"

int main()
{
	Vector3d a(1.0, 1.0, 0.0);
	Vector3d b(1.0, 0.0, 0.0);
	std::cout << a << std::endl;
	std::cout << a.unitVector() << std::endl;
	std::cout << a*3.0 << std::endl;
	std::cout << a*b << std::endl;
	return 0;
}
