#include <iostream>
#include <cmath>
using namespace std;

double integrate(double (*f)(double x), double a, double b)
{
	int    N = 10;
	double h = (b-a)/N;
	double integral = 0.0;
	for(int i=0; i<N; i++)
	{
		double x = a + (i+0.5)*h;
		integral += (*f)(x)*h;
	}
	return integral;
}

int main()
{
	cout << integrate(exp, 0.0, 1.0) << endl;
	cout << integrate(sin, 0.0, M_PI/4.0) << endl;
}
