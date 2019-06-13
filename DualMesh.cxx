#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <list>
#include <math.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkKdTree.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <omp.h>
//#include "HelperFunctions.h"

using namespace std;
//using namespace OPS;

typedef Eigen::Vector3d Vector3d;

//! A struct to store a vtkIdType and an angle
struct neighbors{
    vtkIdType _id;
    double_t _angle;
    neighbors(vtkIdType i, double_t a):_id(i), _angle(a){}
    bool operator <(const neighbors &n) const{
	return _angle < n._angle;
    }
};

int main(int argc, char* argv[])
{
    if( argc != 5 ) {
	cout << "argc = " << argc << endl
	    << "Usage: dualMesh baseName numStart numEnd numOPENMPThreads"
	    << endl;
	return(0);
    }

    ////////////////////////////////////////////////////////////////////
    // Input section
    ////////////////////////////////////////////////////////////////////

    // read in vtk file
    string baseFileName = argv[1];
    size_t numStart = stoi(argv[2]);
    size_t numEnd = stoi(argv[3]);
    size_t numThreads = stoi(argv[4]);

    //Set the number of threads
    omp_set_num_threads(numThreads);

#pragma omp parallel for
    for(size_t bigI=numStart; bigI <= numEnd; bigI++){
	stringstream sstm;
	string inputFileName, outFileName;
	sstm << baseFileName << "-" << bigI <<".vtk";
	inputFileName = sstm.str();
	sstm.str("");
	sstm.clear();

	auto reader = vtkSmartPointer<vtkPolyDataReader>::New();
	reader->SetFileName( inputFileName.c_str() );
	reader->Update();
	vtkSmartPointer<vtkPolyData> inputMesh = reader->GetOutput();
	inputMesh->BuildLinks();
	size_t npts = inputMesh->GetNumberOfPoints();

	//get displacement vectors, if they exist
	string vectorName="displacements";
	vtkSmartPointer<vtkDoubleArray> displacements =
	    vtkDoubleArray::SafeDownCast(inputMesh->GetPointData()->
		    GetVectors(vectorName.c_str()));

	// get vertex positions
	std::vector< Vector3d > points( npts, Vector3d::Zero() );

	// Calculate centroid of each triangle while updating points vector
	auto newPts = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> cells = inputMesh->GetPolys();
	auto cellPointIds = vtkSmartPointer<vtkIdList>::New();
	cells->InitTraversal();
	while( cells->GetNextCell( cellPointIds ) ){
	    size_t numCellPoints = cellPointIds->GetNumberOfIds();
	    Vector3d centroid(0.0,0.0,0.0);
	    for(size_t i=0; i < numCellPoints; i++){
		vtkIdType currCellPoint = cellPointIds->GetId(i);
		inputMesh->GetPoint( currCellPoint, &points[currCellPoint][0] );
		if( displacements.GetPointer() != NULL){
		    Vector3d currDisp(0.0,0.0,0.0);
		    displacements->GetTuple( currCellPoint, &currDisp[0] );
		    points[currCellPoint] += currDisp;
		}
		centroid += points[currCellPoint];
	    }
	    centroid /= numCellPoints;
	    newPts->InsertNextPoint( &centroid[0] );
	}

	// Prepare valence Cell Data array
	auto valence = vtkSmartPointer<vtkIntArray>::New();
	valence->SetName("Valence");
	valence->SetNumberOfComponents(1);

	// Prepare new cell array for polygons
	auto newPolys = vtkSmartPointer<vtkCellArray>::New();

	for(size_t a=0; a < npts; a++) {

	    auto currPolyPtIds = vtkSmartPointer<vtkIdList>::New();
	    std::list< neighbors > currPoly;
	    Vector3d vec0, vecj, currCross, axis, centroid(0.0,0.0,0.0);
	    double vec0_norm, vecj_norm, sign, currSin, currCos, currAngle;

	    inputMesh->GetPointCells( a, currPolyPtIds );
	    size_t numCellPoints = currPolyPtIds->GetNumberOfIds();

	    // Get coordinates of first cell's centroid
	    vtkIdType currId = currPolyPtIds->GetId(0);
	    newPts->GetPoint( currId, &centroid[0] );
	    vec0 = (centroid - points[a]).normalized();
	    neighbors pt0(currId, 0.0);
	    currPoly.push_back( pt0 );

	    // For remaining centroids
	    for(auto j=1; j < numCellPoints; j++){
		currId = currPolyPtIds->GetId(j);
		newPts->GetPoint( currId, &centroid[0] );
		vecj = (centroid - points[a]).normalized();
		currSin = (vec0.cross( vecj )).norm();
		axis = (vec0.cross( vecj )).normalized();
		sign = axis.dot( points[a] );
		currSin = (sign > 0.0) ? currSin : -1.0 * currSin;
		currCos = vec0.dot( vecj );
		currAngle = (180 / M_PI) * atan2(currSin, currCos);
		currAngle = (currAngle < 0) ? (360 + currAngle) : currAngle;
		neighbors ptj(currId, currAngle);
		currPoly.push_back(ptj);
	    }

	    //Sort the list of neigbors and make a polygon
	    currPoly.sort();
	    newPolys->InsertNextCell( numCellPoints );
	    for(auto t = currPoly.begin(); t != currPoly.end(); ++t){
		neighbors n = *t;
		newPolys->InsertCellPoint( n._id );
	    }
	    valence->InsertNextTuple1( numCellPoints );
	}

	//Assign points and polygons to a new polydata and write it out
	auto newPolyData = vtkSmartPointer<vtkPolyData>::New();
	auto writer = vtkSmartPointer<vtkPolyDataWriter>::New();

	newPolyData->SetPoints( newPts );
	newPolyData->SetPolys( newPolys );
	newPolyData->GetCellData()->AddArray( valence );
	writer->SetInputData( newPolyData );
	sstm << baseFileName << "-dual-" << bigI <<".vtk";
	outFileName = sstm.str();
	sstm.str("");
	sstm.clear();
	writer->SetFileName(outFileName.c_str());
	writer->Write();
    }
    return 0;
}
