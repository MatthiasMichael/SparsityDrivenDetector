#pragma once

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

//#define RELEASE_MODE

#ifndef RELEASE_MODE

#define cudaSafeCall(error) __cudaSafeCall(error, __FILE__, __LINE__)
#define cudaCheckError()    __cudaCheckError(__FILE__, __LINE__)
#define cudaCheckMemory(request) __cudaCheckMemory(request, __FILE__, __LINE__)

#else

#define cudaSafeCall(error) error
#define cudaCheckError()    
#define cudaCheckMemory(request) request

#endif

#define STRONG_CHECK


inline void printMemoryCheck(){
	size_t avail;
	size_t total;

	cudaMemGetInfo( &avail, &total );

	size_t used = total - avail;

	std::cout << "Device memory statistics: " << std::endl;
	std::cout << "\tTotal: "     << (total / 1000000)  << "MB (" << total << "B" << std::endl;
	std::cout << "\tUsed: "      << (used  / 1000000)  << "MB (" << used  << "B" << std::endl;
	std::cout << "\tAvailable: " << (avail / 1000000)  << "MB (" << avail << "B" << std::endl;
}


inline void __cudaCheckMemory(size_t request, const char *file, const int line){
	size_t avail;
	size_t total;

	cudaMemGetInfo( &avail, &total );

	if(avail < request){
		std::cerr << "Die Speicheranforderung kann nicht bearbeitet werden:" << std::endl;
		std::cerr << "\tDatei: " << file << ",\nZeile: " << line << ":" << std::endl;
		std::cerr << "\t" << request << "Bytes (" << (request / 1000000) << "MB angefordert."  << std::endl;
		std::cerr << "\t" << avail   << "Bytes (" << (avail   / 1000000) << "MB verfügbar." << std::endl;

		exit(-1);
	}
}


inline void __cudaSafeCall(cudaError error, const char *file, const int line){
	if (error != cudaSuccess){
		std::cerr << "cudaSafeCall() hat folgenden Fehler ergeben:" << std::endl;
		std::cerr << "\tDatei: " << file << ",\nZeile: " << line << " - CudaError " << error << ":" << std::endl;
		std::cerr << "\t" << cudaGetErrorString(error) << std::endl;

		system("PAUSE");
		exit( -1 );
	}
}


inline void __cudaCheckError(const char *file, const int line){

	cudaError error = cudaGetLastError();
	if (error != cudaSuccess){
		std::cerr << "cudaCheckError() hat folgenden Fehler ergeben:" << std::endl;
		std::cerr << "\tDatei: " << file << ",\tZeile: " << line << " - CudaError " << error << ":" << std::endl;
		std::cerr << "\t" << cudaGetErrorString(error) << std::endl;

		system("PAUSE");
		exit( -1 );
	}

#ifdef STRONG_CHECK
	//Wenn besonders stark geprüft werden soll, so wird auf die Fertigstellung 
	//des Device gewartet und die Fehlerprüfung erneut durchgeführt.
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess){
		std::cerr << "cudaCheckError() hat folgenden Fehler ergeben:" << std::endl;
		std::cerr << "\tDatei: " << file << ",\tZeile: " << line << " - CudaError " << error << ":" << std::endl;
		std::cerr << "\t" << cudaGetErrorString(error) << std::endl;
		system("PAUSE");
		exit( -1 );
	}
#endif //STRONG_CHECK
}


template<typename T>
bool checkVectorCopyOkay(const std::vector<T> & hostVector, const T * deviceVector)
{
	const size_t numElements = hostVector.size();

	T * copyTest = new T[numElements];

	cudaSafeCall( cudaMemcpy(copyTest, deviceVector, numElements * sizeof(T), cudaMemcpyDeviceToHost) );

	bool okay = true;
	for(int i = 0; i < numElements; ++i)
	{
		if(copyTest[i] != hostVector[i])
		{
			std::cout << "Copy Error on (" << i << "):" << copyTest[i] << " instead of " << hostVector[i] << std::endl;
			okay = false;
		}
	}
	
	return okay;
}