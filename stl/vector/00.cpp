#include<iostream>
int main()
{
	std::cout << "Sizes of data types\n";
	std::cout << "-------------------\n";
	std::cout << "size_t: " << sizeof(size_t) << "\nint: " << sizeof(int) << "\nunsigned int: " << sizeof(unsigned int) << std::endl;
	std::cout << "long: " << sizeof(long) << '\n';
	std::cout << "long int: " << sizeof(long int) << '\n';
	std::cout << "long long: " << sizeof(long long) << '\n';
	std::cout << "float: " << sizeof(float) << '\n';
	std::cout << "double: " << sizeof(double) << '\n';
	std::cout << "long double: " << sizeof(long double) << '\n';
	std::cout << "char: " << sizeof(char) << '\n';
	return 0;
}
