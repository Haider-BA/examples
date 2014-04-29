#include<iostream>
#include<vector>
int main()
{
	std::vector<int> v;

	v.push_back(-1);
	v.push_back(23);
	v.push_back(5);

	std::cout << "Size of vector: " << v.size() << std::endl;

	std::cout << "\nPrint vector using a loop index:\n";
	for(size_t i=0; i<v.size(); i++)
	{	
		std::cout << v[i] << ", ";
	}
	std::cout << std::endl;

	std::cout << "\nPrint vector using an iterator:\n";

	// in c++11, we can replace std::vector<int>::iterator with auto
	for(std::vector<int>::iterator j=v.begin(); j!=v.end(); ++j)
	{
		std::cout << *j << ", ";
	}
	std::cout << std::endl;

	return 0;
}
