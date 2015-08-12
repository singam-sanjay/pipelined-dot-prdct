#include<iostream>
#include<cstdlib>
#include"cublas_v2.h"

using namespace std;
typedef float TYPE;

int main()
{
	size_t N;
	cout << "N:";cin >> N;

	TYPE *arr,*d_arr;

	try
	{
		arr = new TYPE[N];
	}
	catch(exception e)
	{
		cerr << "Unable to allocate mem for arr: " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	if( cudaMalloc( &d_arr,sizeof(TYPE)*N )!=cudaSuccess )
	{
		cerr << "Unable to allocate d_arr :" << cudaGetErrorString( cudaGetLastError() ) << '\n';
		return EXIT_FAILURE;
	}

	if( cudaFree( d_arr )!=cudaSuccess )
	{
		cerr << "Unable to deallocate mem for d_arr: " << cudaGetErrorString( cudaGetLastError() ) << '\n';
		return EXIT_FAILURE;
	}

	try
	{
		delete [] arr;
	}
	catch(exception e)
	{
		cerr << "Unable to deallocate mem for arr: " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
