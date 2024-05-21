#!/bin/bash

p_mpi=/usr/bin/
p_scorep=${SCOREP_PATH}/bin/

for x in $(find . -maxdepth 1 -name 'local*');
do 
  echo "replacing $x"
  sed -i -e "s|${p_mpi}mpicc|${p_scorep}scorep-mpicc|g" $x
  sed -i -e "s|${p_mpi}mpic++|${p_scorep}scorep-mpicxx|g" $x
  sed -i -e "s|${p_mpi}mpifort|${p_scorep}scorep-mpif90|g" $x
done
