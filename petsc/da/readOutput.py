import sys, os
sys.path.append(os.path.join(os.environ['PETSC_DIR'],'bin','pythonscripts'))
import PetscBinaryIO
petsc_objs = PetscBinaryIO.PetscBinaryIO().readVec('U.dat')[1:]
print petsc_objs
