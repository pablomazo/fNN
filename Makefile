FC=gfortran
FFLAGS=-O3

ROOT_DIR=./src/
FMODS=-I $(ROOT_DIR)/nn
fNN_src=$(addprefix $(ROOT_DIR), nn/nn_types.f90 nn/nn_activations.f90 nn/nn_layers.f90 nn/nn_network.f90 nn.f90)
fNNo=$(fNN_src:%.f90=%.o)

fnn: $(fNNo)
	rm -f ./lib/libnn.a
	ar cvr ./lib/libnn.a $^

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $< $(FMODS)

