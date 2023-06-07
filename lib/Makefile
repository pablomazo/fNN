FC=gfortran
FFLAGS=-O3

ROOT_DIR=../src/
FMODS=-I $(ROOT_DIR)/nn
fNN_src=$(addprefix $(ROOT_DIR), nn/nn_types.f90 nn/nn_activations.f90 nn/nn_layers.f90 nn/nn_network.f90 nn.f90)
fNNo=$(fNN_src:%.f90=%.o)

fnn: $(fNNo)
	rm -f libnn.a
	ar cvr libnn.a $^

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $< $(FMODS)

clean:
	rm -rf $(ROOT_DIR)/*.o $(ROOT_DIR)/*.mod $(ROOT_DIR)/nn/*.o $(ROOT_DIR)/nn/*.mod
